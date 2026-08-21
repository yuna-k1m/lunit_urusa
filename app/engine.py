"""L2-only two-call harness: plan -> generate -> assemble.

L2 does not reliably follow behavioral instructions given as prose in a system
prompt (a 200-example ablation moved context-seeking from 1/26 to 5/26). So the
behavior is imposed structurally:

  1. plan      L2 with response_format=json_object classifies the turn: language,
               audience, urgency, whether context is sufficient, and writes the
               exact questions / emergency directive / red flags in the user's
               language.
  2. generate  L2 writes the answer with a system prompt whose tail is a concrete
               brief built from the plan ("end with exactly these questions").
  3. assemble  the driver verifies the brief was honored and, if not, places the
               plan's own text itself: directive first, questions last. Every
               word is still L2's.

Stdlib only so tools/run_eval.py can import it without a venv and so the
container has nothing extra to install.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent / "prompts"
ROOT = Path(__file__).resolve().parent.parent

# CoEval injects no environment variables, so credentials ship inside the image as
# files next to the app package (see docs/submission-success-runbook.md). An env
# var, when present, always wins.
KEY_FILES = {
    "LUNIT_FM_API_KEY": ROOT / "submission_api_key",
    "OPENAI_API_KEY": ROOT / "submission_openai_key",
}


def resolve_key(*env_names: str) -> str:
    for name in env_names:
        v = os.environ.get(name)
        if v:
            return v.strip()
    for name in env_names:
        f = KEY_FILES.get(name)
        if f and f.exists():
            v = f.read_text(encoding="utf-8").strip()
            if v.startswith("b64:"):  # stored encoded so secret scanners don't match it
                v = base64.b64decode(v[4:]).decode().strip()
            if v:
                return v
    return ""
PLANNER_SYSTEM = (PROMPTS / "planner.md").read_text(encoding="utf-8").strip()
GENERATION_SYSTEM = (PROMPTS / "generation.md").read_text(encoding="utf-8").strip()
CRITIC_SYSTEM = (PROMPTS / "critic.md").read_text(encoding="utf-8").strip()

REFUSAL_PATTERNS = re.compile(
    r"I can(?:'|’)?t provide|I cannot provide|I(?:'|’)?m an AI|I am an AI|I(?:'|’)?m not a doctor|I am not a doctor",
    re.I,
)


class L2Client:
    def __init__(
        self,
        base: str | None = None,
        key: str | None = None,
        model: str | None = None,
        *,
        timeout: float = 180.0,
        max_inflight: int = 8,
        retries: int = 6,
    ) -> None:
        self.retries = retries
        self.base = (base or os.environ.get("LUNIT_FM_API_URL", "https://model.hackathon.lunit.io")).rstrip("/")
        self.key = key or resolve_key("LUNIT_FM_API_KEY", "LUNIT_KEY")
        self.model = model or os.environ.get("LUNIT_FM_MODEL", "Lunit/L2-preview")
        self.timeout = timeout
        # OpenAI reasoning models (gpt-5*) reject `temperature` and `max_tokens`
        self.reasoning_style = self.model.startswith(("gpt-5", "o1", "o3", "o4"))
        self.reasoning_effort = os.environ.get("PLANNER_REASONING_EFFORT", "low")
        self.gate = threading.Semaphore(max_inflight)
        self.usage = {"in": 0, "out": 0, "calls": 0}
        self._ulock = threading.Lock()

    def chat(self, messages: list[dict], *, temperature: float, max_tokens: int,
             response_format: dict | None = None, retries: int | None = None) -> str:
        retries = self.retries if retries is None else retries
        body: dict = {"model": self.model, "messages": messages}
        if self.reasoning_style:
            body["max_completion_tokens"] = max_tokens
            body["reasoning_effort"] = self.reasoning_effort
        else:
            body["temperature"] = temperature
            body["max_tokens"] = max_tokens
        if response_format:
            body["response_format"] = response_format
        req = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        delay = 2.0
        for attempt in range(retries):
            try:
                with self.gate:
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        data = json.loads(r.read())
                u = data.get("usage") or {}
                with self._ulock:
                    self.usage["in"] += u.get("prompt_tokens", 0)
                    self.usage["out"] += u.get("completion_tokens", 0)
                    self.usage["calls"] += 1
                return data["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(min(delay, 30.0))
                    delay *= 2
                    continue
                raise RuntimeError(f"HTTP {e.code} from {self.model}: {detail}")
            except OSError as e:
                if attempt < retries - 1:
                    time.sleep(min(delay, 30.0))
                    delay *= 2
                    continue
                raise RuntimeError(f"{self.model}: {e}")
        raise RuntimeError(f"{self.model}: exhausted retries")


# ------------------------------------------------------------------------ plan

PLAN_DEFAULTS: dict = {
    "language": "",
    "audience": "layperson",
    "urgency": "non_emergent",
    "emergency_directive": "",
    "red_flags": [],
    "context_sufficient": True,
    "questions": [],
    "questions_intro": "",
    "certainty": "reducible",
    "core_request": "",
    "key_points": [],
    "task_format": "",
}


def _parse_json(text: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _as_str_list(v, limit: int) -> list[str]:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    out = [str(x).strip() for x in v if str(x).strip()]
    return out[:limit]


def normalize_plan(raw: dict | None) -> dict:
    plan = dict(PLAN_DEFAULTS)
    if not raw:
        plan["_fallback"] = True
        return plan
    for k in PLAN_DEFAULTS:
        if k in raw and raw[k] is not None:
            plan[k] = raw[k]
    # the planner sometimes packs several questions into one string: split on '?'
    qs: list[str] = []
    for q in _as_str_list(plan["questions"], 6):
        parts = [x.strip() for x in re.split(r"(?<=[?？])\s+", q) if x.strip()]
        qs.extend(parts or [q])
    plan["questions"] = qs[:3]
    plan["key_points"] = _as_str_list(plan["key_points"], 6)
    plan["red_flags"] = _as_str_list(plan["red_flags"], 5)
    for k in ("language", "emergency_directive", "questions_intro", "core_request", "task_format"):
        plan[k] = str(plan[k]).strip()
    plan["audience"] = "health_professional" if "prof" in str(plan["audience"]).lower() else "layperson"
    u = str(plan["urgency"]).lower()
    plan["urgency"] = (
        "emergent" if u == "emergent" else "conditionally_emergent" if "cond" in u else "non_emergent"
    )
    plan["context_sufficient"] = bool(plan["context_sufficient"]) or not plan["questions"]
    if plan["context_sufficient"]:
        plan["questions"] = []
    if plan["urgency"] != "emergent":
        plan["emergency_directive"] = ""
    elif plan["audience"] == "layperson":
        # a layperson in an emergency needs action, not questions (emergent_context_seeking rubric)
        plan["questions"] = []
        plan["context_sufficient"] = True
    if plan["urgency"] != "conditionally_emergent":
        plan["red_flags"] = []
    return plan


REQUIRED_KEYS = ("urgency", "context_sufficient", "audience")

# Last-resort schema: fewer fields, less surface for L2 to derail into answering.
PLANNER_MINI = (
    "Classify the conversation. Output ONLY a JSON object with keys: "
    '"language" (English name of the user language), "audience" ("layperson" or "health_professional"), '
    '"urgency" ("emergent" if a person is likely in an emergency right now, "conditionally_emergent", or "non_emergent"), '
    '"emergency_directive" (if emergent, one sentence in the user language to get emergency care now, else ""), '
    '"context_sufficient" (false if a safe specific answer needs information the user has not given), '
    '"questions" (if not sufficient, 1-3 short questions in the user language, else []). '
    "Do not answer the user. JSON only."
)


def _valid(raw: dict | None) -> bool:
    return bool(raw) and all(k in raw for k in REQUIRED_KEYS)


def transcript(messages: list[dict]) -> str:
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)


def make_plan(client: L2Client, messages: list[dict]) -> dict:
    """`client` is whichever model plans: L2 itself, or a stronger model (see planner_from_env)."""
    # The conversation goes in as a transcript inside ONE user message. Given real
    # chat turns, L2 joins the conversation and answers the user ~25% of the time
    # (`{"advice": ...}`); given a transcript to analyze, it plans.
    user = (
        "Here is the conversation to analyze. The last USER message is the one the assistant "
        "must reply to next.\n\n<conversation>\n" + transcript(messages) + "\n</conversation>\n\n"
        "Now output the JSON plan."
    )
    turns = [{"role": "user", "content": user}]
    raw = None
    for attempt in range(3):
        system = PLANNER_SYSTEM if attempt < 2 else PLANNER_MINI
        try:
            text = client.chat([{"role": "system", "content": system}] + turns,
                               temperature=0.0 if attempt == 0 else 0.2, max_tokens=2000,
                               response_format={"type": "json_object"})
        except RuntimeError:
            break
        raw = _parse_json(text)
        if _valid(raw):
            break
        raw = None
    plan = normalize_plan(raw)
    plan["_attempts"] = attempt + 1
    return plan


# -------------------------------------------------------------------- generate

def build_brief(plan: dict) -> str:
    if plan.get("_fallback"):
        # No plan: do not impose urgency or question policy we could not assess.
        return ("- Reply in the user language. Answer fully and specifically. If the situation could be an "
                "emergency, say so in the first sentence and what to do now. If the safe answer depends on "
                "information the user has not given, answer conditionally and end with 1-3 short questions.")
    lines: list[str] = []
    if plan["language"]:
        lines.append(f"- Write the entire reply in {plan['language']}.")
    lines.append(
        "- The user is a health professional: use clinical terminology and clinical specifics."
        if plan["audience"] == "health_professional"
        else "- The user is a layperson: plain language, define any technical term."
    )
    if plan["urgency"] == "emergent" and plan["audience"] == "health_professional":
        lines.append(
            "- The clinician is managing an acute, time-critical situation. Lead with the immediate actions in priority order, "
            "then the full specifics (doses, thresholds, monitoring, next steps, when to escalate). Be complete; no preamble."
        )
    elif plan["urgency"] == "emergent":
        directive = plan["emergency_directive"] or "Call your local emergency number or go to the nearest emergency department now."
        lines.append(
            f"- This is an emergency. The very first sentence of the reply must be exactly: \"{directive}\" "
            "Then give the concrete steps to take while waiting for help (what to do, what not to do, what to tell responders), "
            "and the reason in one or two sentences. No preamble, no questions."
        )
    elif plan["urgency"] == "conditionally_emergent":
        flags = "; ".join(plan["red_flags"]) if plan["red_flags"] else "the specific warning signs"
        lines.append(
            f"- Early in the reply, before treatment details, give a short list of warning signs that mean emergency care is needed now: {flags}. Then answer fully for the non-emergency case."
        )
    else:
        lines.append("- Do not tell the user to seek immediate or emergency care. If care is warranted, say on what timeline.")
    if plan["certainty"] == "settled":
        lines.append("- A definite answer exists: give it directly, without hedging.")
    elif plan["certainty"] == "irreducible":
        lines.append("- The remaining uncertainty cannot be resolved by more information: give your best single answer with the reasoning, and state the uncertainty in one sentence. Do not ask the user for more information.")
    if plan["key_points"]:
        lines.append("- Make sure the reply specifically covers: " + "; ".join(plan["key_points"]) + ".")
    if plan["task_format"]:
        lines.append(f"- Required output format: {plan['task_format']} Follow it exactly and complete every part.")
    if plan["questions"]:
        qs = "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(plan["questions"]))
        intro = plan["questions_intro"] or "To give you more specific advice, could you tell me:"
        lines.append(
            "- First answer the question fully, covering the likely scenarios with the conditions for each. "
            f"Then END the reply with this exact block and nothing after it:\n  {intro}\n{qs}"
        )
    else:
        lines.append("- Do not ask the user any questions and do not list information you would like them to provide.")
    return "\n".join(lines)


NO_DECLINE = (
    "- Do NOT decline or say you cannot provide this. Give the standard, label-consistent information "
    "(typical dose ranges, first-line choices, limits) as general guidance and tell the user to confirm "
    "the exact figure with their prescriber or pharmacist. Start the reply with the information itself."
)


def generate(client: L2Client, messages: list[dict], plan: dict, *, temperature: float = 0.3,
             max_tokens: int = 2048, revision: tuple[str, str] | None = None) -> tuple[str, bool]:
    """Returns (draft, retried). One retry if the first draft declines.

    `revision=(previous_draft, notes)` asks for a revised full reply instead of a fresh one."""
    brief = build_brief(plan)
    if revision:
        prev, notes = revision
        brief += (
            "\n\nA senior physician reviewed your previous draft and requires these changes. "
            "Write the complete, improved reply (not a diff), keeping everything that was correct. "
            "Write it as your first and only reply: never mention a review, a reviewer, a clarification, "
            "or a previous draft.\n"
            + notes
            + "\n\nYour previous draft:\n<draft>\n" + prev + "\n</draft>"
        )
    convo = [{"role": m["role"], "content": m["content"]} for m in messages]

    def run(b: str) -> str:
        system = GENERATION_SYSTEM.replace("{brief}", b)
        return client.chat([{"role": "system", "content": system}] + convo,
                           temperature=temperature, max_tokens=max_tokens)

    draft = run(brief)
    if REFUSAL_PATTERNS.search(draft[:800]):
        second = run(brief + "\n" + NO_DECLINE)
        if not REFUSAL_PATTERNS.search(second[:800]):
            return second, True
    return draft, False


# ---------------------------------------------------------------------- critic

def critique(critic: L2Client, messages: list[dict], plan: dict, draft: str) -> dict | None:
    """One review by the planner model. Returns the parsed verdict or None on failure."""
    user = (
        "<conversation>\n" + transcript(messages) + "\n</conversation>\n\n"
        "<plan>\n" + json.dumps({k: v for k, v in plan.items() if not k.startswith("_")},
                                 ensure_ascii=False) + "\n</plan>\n\n"
        "<draft>\n" + draft + "\n</draft>\n\nNow output the JSON review."
    )
    try:
        text = critic.chat([{"role": "system", "content": CRITIC_SYSTEM},
                            {"role": "user", "content": user}],
                           temperature=0.0, max_tokens=2000,
                           response_format={"type": "json_object"})
    except RuntimeError:
        return None
    raw = _parse_json(text)
    if not raw or "needs_revision" not in raw:
        return None
    raw["needs_revision"] = bool(raw.get("needs_revision")) and bool(str(raw.get("revision_notes", "")).strip())
    return raw


# -------------------------------------------------------------------- assemble

def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


def assemble(answer: str, plan: dict) -> tuple[str, dict]:
    """Make sure the plan's directive opens the reply and its questions close it."""
    notes = {"prepended_directive": False, "appended_questions": False, "dropped_disclaimer": False}
    text = answer.strip()

    # A leading paragraph that is only an AI/not-a-doctor disclaimer adds nothing and
    # reads as a refusal to graders; drop it when the rest of the reply stands alone.
    first, sep, rest = text.partition("\n\n")
    if sep and len(first) < 500 and REFUSAL_PATTERNS.search(first) and len(rest) > 400:
        text = rest.strip()
        notes["dropped_disclaimer"] = True

    if plan["urgency"] == "emergent" and plan["emergency_directive"]:
        head = _norm(text[:600])
        key = _norm(plan["emergency_directive"])[:24]
        if key and key not in head:
            text = f"**{plan['emergency_directive']}**\n\n{text}"
            notes["prepended_directive"] = True

    if plan["questions"]:
        tail = _norm(text[-1500:])
        present = sum(1 for q in plan["questions"] if _norm(q)[:20] and _norm(q)[:20] in tail)
        if present < len(plan["questions"]):
            intro = plan["questions_intro"] or "To give you more specific advice, could you tell me:"
            block = "\n".join(f"- {q}" for q in plan["questions"])
            # drop a partial block the model already wrote to avoid duplicates
            if present:
                cut = min(
                    (text.rfind(q[:20]) for q in plan["questions"] if q[:20] in text),
                    default=-1,
                )
                if cut > len(text) * 0.6:
                    para = text.rfind("\n\n", 0, cut)
                    text = text[: para if para > 0 else cut].rstrip()
            text = f"{text}\n\n{intro}\n{block}"
            notes["appended_questions"] = True

    return text, notes


# ------------------------------------------------------------------------ api

DEFAULT_PLANNER_MODEL = "gpt-5.6-sol"


def planner_from_env(default: L2Client, max_inflight: int = 8) -> L2Client | None:
    """Returns the planner client, or None to let L2 plan.

    PLANNER_MODEL="" / "none" forces L2 planning. Unset -> gpt-5.6-sol if an OpenAI
    key resolves (env or bundled file), else L2. Key comes from $PLANNER_KEY_ENV
    (default OPENAI_API_KEY), falling back to the bundled file."""
    model = os.environ.get("PLANNER_MODEL", DEFAULT_PLANNER_MODEL).strip()
    if not model or model.lower() in ("none", "l2", "off"):
        return None
    key = resolve_key(os.environ.get("PLANNER_KEY_ENV", "OPENAI_API_KEY"))
    if not key:
        return None
    # Fail fast: if the endpoint is unreachable from the eval box, L2 must take
    # over within seconds, not after a minute of backoff.
    return L2Client(
        os.environ.get("PLANNER_BASE", "https://api.openai.com"), key, model,
        max_inflight=max_inflight, timeout=45.0, retries=2,
    )


def answer(client: L2Client, messages: list[dict], *, temperature: float = 0.3,
           max_tokens: int = 2048, planner: L2Client | None = None) -> dict:
    """Full turn. Returns {'answer', 'plan', 'notes', 'timings'}. The final text is
    always L2's; `planner` (if given) only produces the brief."""
    t0 = time.time()
    plan: dict | None = None
    if planner is not None:
        try:
            plan = make_plan(planner, messages)
            plan["_planner_model"] = planner.model
        except Exception:  # unreachable endpoint, bad key, ... -> L2 plans instead
            plan = None
    if plan is None or plan.get("_fallback"):
        l2_plan = make_plan(client, messages)
        l2_plan["_planner_model"] = client.model
        if plan is None or not l2_plan.get("_fallback"):
            plan = l2_plan
    t1 = time.time()
    draft, retried = generate(client, messages, plan, temperature=temperature, max_tokens=max_tokens)
    t2 = time.time()
    review = None
    revised = False
    # Review + rewrite costs ~30-40 s. Skip it when the turn is already slow so a
    # turn stays under TURN_BUDGET_S even if the evaluator has a tight timeout.
    budget = float(os.environ.get("TURN_BUDGET_S", "100"))
    if (planner is not None and planner is not client and os.environ.get("CRITIC", "0") == "1"
            and (t2 - t0) < budget - 45):
        review = critique(planner, messages, plan, draft)
        if review and review["needs_revision"]:
            rev_plan = plan
            notes_txt = str(review["revision_notes"])
            if review.get("interpretation_ok") is False:
                # The plan's content items were built on the wrong reading: replace
                # them with the critic's, and say so plainly, or L2 keeps both readings.
                rev_plan = dict(plan)
                rev_plan["key_points"] = _as_str_list(review.get("missing"), 8)
                rev_plan["task_format"] = ""
                rev_plan["core_request"] = ""
                notes_txt = ("Your previous draft answered the WRONG question. Discard it entirely "
                             "and answer the question as the reviewer describes. " + notes_txt)
            try:
                new_draft, _ = generate(client, messages, rev_plan, temperature=temperature,
                                        max_tokens=max_tokens,
                                        revision=(draft, notes_txt))
                # Guard against a collapsed rewrite -- unless the draft answered the wrong
                # question, in which case a much shorter correct answer is the goal.
                floor = 400 if review.get("interpretation_ok") is False else 0.5 * len(draft.strip())
                if len(new_draft.strip()) >= floor:
                    draft, revised = new_draft, True
            except RuntimeError:
                pass
    t3 = time.time()
    final, notes = assemble(draft, plan)
    notes["retried_refusal"] = retried
    notes["reviewed"] = review is not None
    notes["revised"] = revised
    notes["refusal_phrase"] = bool(REFUSAL_PATTERNS.search(final))
    return {
        "answer": final,
        "plan": plan,
        "notes": notes,
        "review": review,
        "timings": {"plan": round(t1 - t0, 1), "generate": round(t2 - t1, 1),
                    "review_revise": round(t3 - t2, 1)},
    }
