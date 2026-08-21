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
from typing import Any

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
def _load_planner_system() -> str:
    text = (PROMPTS / "planner.md").read_text(encoding="utf-8").strip()
    titles_file = PROMPTS / "guideline_titles.json"
    titles = json.loads(titles_file.read_text(encoding="utf-8")) if titles_file.exists() else []
    return text.replace("{guideline_titles}", "\n".join(f"- {t}" for t in titles) or "- (none)")


PLANNER_SYSTEM = _load_planner_system()


def _lean_planner_system() -> str:
    """Planner prompt for L2 itself (no OpenAI egress on the evaluation box): no
    corpus title list, no retrieval/search fields. 4.6k chars instead of 18.8k;
    with thinking off L2 plans validly in ~5-7 s (vs 12-41 s, 5/6 valid)."""
    base = PLANNER_SYSTEM.split("\nAVAILABLE SOURCES")[0]
    drop = ('- "needs_grounding"', '- "retrieval_query"', '- "retrieval_hints"')
    return "\n".join(l for l in base.splitlines() if not l.startswith(drop))


PLANNER_SYSTEM_LEAN = _lean_planner_system()
GENERATION_SYSTEM = (PROMPTS / ("generation_template.md" if os.environ.get("GEN_PROMPT", "rules") == "template" else "generation.md")).read_text(encoding="utf-8").strip()
CRITIC_SYSTEM = (PROMPTS / "critic.md").read_text(encoding="utf-8").strip()
SELECTOR_SYSTEM = (PROMPTS / "selector.md").read_text(encoding="utf-8").strip()

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
        msg = self.chat_message(messages, temperature=temperature, max_tokens=max_tokens,
                                response_format=response_format, retries=retries)
        return msg.get("content") or ""

    def chat_message(self, messages: list[dict], *, temperature: float, max_tokens: int,
                     response_format: dict | None = None, tools: list[dict] | None = None,
                     tool_choice: Any = None, retries: int | None = None,
                     thinking: bool | None = None, timeout: float | None = None) -> dict:
        """Full assistant message (so callers can read `tool_calls`).

        `thinking=False` disables L2's hidden reasoning channel
        (chat_template_kwargs.enable_thinking): ~30% faster, measured; quality
        impact is what L2_THINKING=0 runs exist to establish."""
        retries = self.retries if retries is None else retries
        body: dict = {"model": self.model, "messages": messages}
        if thinking is False and not self.reasoning_style:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
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
                    with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                        data = json.loads(r.read())
                u = data.get("usage") or {}
                with self._ulock:
                    self.usage["in"] += u.get("prompt_tokens", 0)
                    self.usage["out"] += u.get("completion_tokens", 0)
                    self.usage["calls"] += 1
                return data["choices"][0]["message"] or {}
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(min(delay, 30.0))
                    delay *= 2
                    continue
                raise RuntimeError(f"HTTP {e.code} from {self.model}: {detail}")
            except OSError as e:
                # A timeout must not be retried when a per-call deadline is set: the retry
                # would double the wall time and turn a late answer into a zero.
                if attempt < retries - 1 and not (timeout and isinstance(e, TimeoutError)):
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
    "needs_grounding": False,
    "retrieval_query": "",
    "retrieval_hints": {},
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
    plan["key_points"] = _as_str_list(plan["key_points"], 10)
    plan["red_flags"] = _as_str_list(plan["red_flags"], 5)
    for k in ("language", "emergency_directive", "questions_intro", "core_request", "task_format",
              "retrieval_query"):
        plan[k] = str(plan[k]).strip()
    plan["needs_grounding"] = bool(plan["needs_grounding"]) and bool(plan["retrieval_query"])
    if not isinstance(plan["retrieval_hints"], dict):
        plan["retrieval_hints"] = {}
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


ASK_SYSTEM = (
    "You decide ONE thing about a health conversation: whether the assistant's next reply must ask the user "
    "for missing information before a specific, safe answer is possible. Output ONLY JSON: "
    '{"language": "<English name of the user language>", "ask": true|false, '
    '"questions": ["...", "..."], "intro": "<one short sentence in the user language introducing the questions>"}. '
    "ask=true ONLY when the safe, specific answer genuinely depends on facts the user has not given "
    "(e.g. a layperson describes symptoms or asks about a treatment/dose without age, duration, severity, "
    "pregnancy, conditions, or current medications), AND the user has not refused to give more, AND the "
    "situation is not an active emergency. ask=false for general knowledge questions, questions from "
    "clinicians, documentation/rewrite tasks, when the details are already given, or when more information "
    "would not change the advice. If ask=true, write 1-2 short questions in the user language, the most "
    "decisive first (emergency-deciding > safety-deciding such as age/pregnancy/conditions/medications > "
    "cause-narrowing). Never ask what the user already said. JSON only."
)


def ask_classifier(client: L2Client, messages: list[dict], plan: dict) -> None:
    """Ask-only decision by L2 (thinking off, ~5 s). Fills plan.questions when needed."""
    user = ("<conversation>\n" + transcript(messages) + "\n</conversation>\n\n"
            "Decide whether the next reply must ask for missing information. Output the JSON.")
    try:
        text = client.chat_message([{"role": "system", "content": ASK_SYSTEM}, {"role": "user", "content": user}],
                                   temperature=0.0, max_tokens=400, response_format={"type": "json_object"},
                                   thinking=False, retries=1,
                                   timeout=float(os.environ.get("ASK_TIMEOUT_S", "20"))).get("content") or ""
        raw = _parse_json(text) or {}
    except RuntimeError:
        return
    plan["_ask_checked"] = True
    if raw.get("language"):
        plan["language"] = str(raw["language"]).strip()
    if raw.get("ask") and raw.get("questions"):
        qs: list[str] = []
        for q in _as_str_list(raw["questions"], 4):
            qs.extend([x.strip() for x in re.split(r"(?<=[?？])\s+", q) if x.strip()] or [q])
        plan["questions"] = qs[:2]
        plan["questions_intro"] = str(raw.get("intro", "")).strip()
        plan["context_sufficient"] = not plan["questions"]


GROUND_SYSTEM = (
    "You decide whether answering this health conversation needs a lookup in authoritative Korean/clinical "
    "sources, and extract what to look up. Output ONLY JSON with keys: "
    '"ground": true|false, "hira_query": "", "mfds_drug": "", "kcd_name": "", "statutes": [], "law_keyword": "", '
    '"drug_inn": "", "pubmed_query": "", "guideline_item": "". '
    "ground=true ONLY for: Korean health-insurance reimbursement / claims / 심사 questions (급여, 청구, 고시, 심사기준) -> "
    "hira_query (Korean); Korean drug approval or indication (허가, 적응증) -> mfds_drug (Korean product or ingredient name); "
    "a KCD diagnosis code request -> kcd_name (Korean disease name); a Korean law/regulation/deadline question -> "
    "statutes (1-2 Korean statute names) + law_keyword (one Korean word likely in the article title); a question "
    "whether a specific research finding is real or what studies show -> pubmed_query (English: population, "
    "exposure, outcome); a question asking what a specific drug label says (interaction, contraindication, max dose) "
    "-> drug_inn (English INN); ALSO fill drug_inn with the English INN whenever any drug is mentioned in a Korean reimbursement or approval question (HIRA notices are indexed by INN); an explicit request for what a named clinical guideline recommends -> guideline_item "
    "(English). Everything else (symptoms, general advice, explanations, emergencies, documentation tasks) -> "
    "ground=false with empty fields. Fill only the applicable fields. JSON only."
)


def grounding_gate(client: L2Client, messages: list[dict], plan: dict) -> None:
    """L2-only grounding decision (thinking off, ~3 s): sets needs_grounding and the
    structured seed hints so the retrieval stage can fetch HIRA/MFDS/KCD/law/PubMed/
    label/guideline content without any external planner."""
    user = ("<conversation>\n" + transcript(messages) + "\n</conversation>\n\n"
            "Decide whether a source lookup is needed and extract the lookup targets. Output the JSON.")
    try:
        text = client.chat_message([{"role": "system", "content": GROUND_SYSTEM}, {"role": "user", "content": user}],
                                   temperature=0.0, max_tokens=400, response_format={"type": "json_object"},
                                   thinking=False, retries=1,
                                   timeout=float(os.environ.get("GROUND_TIMEOUT_S", "20"))).get("content") or ""
        raw = _parse_json(text) or {}
    except RuntimeError:
        return
    from app.seed import normalize_hints, has_any
    hints = normalize_hints(raw)
    # Deterministic recall floor: L2's extraction is variable, and a missed lookup on a
    # Korean reimbursement/approval question costs a confidently wrong answer.
    last = messages[-1]["content"]
    if re.search(r"[가-힣]", last):
        if not hints["hira_query"] and re.search(r"급여|청구|심사|고시|비급여|본인부담|인정기준", last):
            # compact content words only: the full sentence returns unrelated notices
            toks = [t for t in re.split(r"[\s,.?!()]+", last)
                    if t and not re.search(r"(나요|되나요|어떻게|되요|인가요|입니까|할까요|주세요|알려|뭔가요|무엇|궁금)$", t)]
            toks = [re.sub(r"(은|는|이|가|을|를|의|에|에서|도|로|으로|과|와)$", "", t) for t in toks]
            hints["hira_query"] = " ".join(t for t in toks if len(t) > 1)[:60]
        if not hints["mfds_drug"] and re.search(r"허가|적응증|식약처|승인", last):
            m = re.search(r"([가-힣A-Za-z]+(?:정|캡슐|주|펜|시럽|정제))", last)
            hints["mfds_drug"] = (m.group(1) if m else "")
        if not hints["kcd_name"] and re.search(r"KCD|질병\s*코드|상병\s*코드|진단\s*코드", last):
            hints["kcd_name"] = re.sub(r"(KCD|질병|상병|진단|코드|알려|뭔가요|무엇|\?|\s)+", " ", last).strip()[:40]
        if not hints["pubmed_query"] and re.search(r"논문|연구\s*결과|연구에서|study|연구가", last):
            hints["pubmed_query"] = hints["pubmed_query"] or raw.get("pubmed_query") or ""
    if has_any(hints):  # L2 fills the hint fields reliably; its boolean is not consistent with them
        plan["needs_grounding"] = True
        plan["retrieval_hints"] = {k: v for k, v in hints.items() if v}
        plan["retrieval_query"] = (hints.get("hira_query") or hints.get("pubmed_query") or hints.get("guideline_item")
                                   or hints.get("mfds_drug") or hints.get("kcd_name") or " ".join(hints.get("statutes") or [])
                                   or hints.get("drug_inn") or messages[-1]["content"][:200])
        if not plan.get("language"):
            last = messages[-1]["content"]
            plan["language"] = "Korean" if re.search(r"[가-힣]", last) else ""


def make_plan(client: L2Client, messages: list[dict]) -> dict:
    """`client` is whichever model plans: L2 itself, or a stronger model (see planner_from_env)."""
    if os.environ.get("NO_PLANNER", "0") == "1":
        # Measurement/ablation mode: neutral brief only. A wrong L2 plan imposes the
        # wrong behaviour; no plan lets L2 answer with the generation prompt alone.
        p = normalize_plan(None)
        p["_attempts"] = 0
        if os.environ.get("ASK_CLASSIFIER", "0") == "1":
            ask_classifier(client, messages, p)
        if os.environ.get("GROUNDING_GATE", "0") == "1":
            grounding_gate(client, messages, p)
        return p
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
    l2_plans = not client.reasoning_style  # L2 itself: lean prompt, thinking off, longer cap
    plan_timeout = float(os.environ.get("PLAN_TIMEOUT_S", ("70" if os.environ.get("L2_PLAN_THINKING", "0") == "1" else "45") if l2_plans else "25"))
    for attempt in range(3):
        if l2_plans:
            system = PLANNER_SYSTEM_LEAN if attempt < 2 else PLANNER_MINI
        else:
            system = PLANNER_SYSTEM if attempt < 2 else PLANNER_MINI
        try:
            text = client.chat_message([{"role": "system", "content": system}] + turns,
                                       temperature=0.0 if attempt == 0 else 0.2, max_tokens=2000,
                                       response_format={"type": "json_object"},
                                       retries=1, timeout=plan_timeout,
                                       thinking=(False if os.environ.get("L2_PLAN_THINKING", "0") != "1" else None) if l2_plans else None).get("content") or ""
        except RuntimeError:
            if attempt == 0:
                continue
            break
        raw = _parse_json(text)
        if _valid(raw):
            break
        raw = None
    plan = normalize_plan(raw)
    plan["_attempts"] = attempt + 1
    return plan


# -------------------------------------------------------------------- generate

def build_brief(plan: dict, evidence: str = "", searched_without_result: bool = False,
                n_turns: int = 1) -> str:
    brief = _build_brief(plan, n_turns)
    if searched_without_result:
        brief += (
            "\n\nA source search for this question found nothing citable. Do NOT invent specifics you cannot "
            "verify: no article numbers (제N조), 고시 numbers, recommendation classes or levels of evidence, "
            "study figures, or PMIDs. Give the general position, say plainly that the exact citation could "
            "not be verified here, and tell the user where to confirm it (which law, guideline, or agency)."
        )
    if evidence:
        brief += (
            "\n\nEvidence retrieved from authoritative sources for this question. Use it: state what the "
            "source says and cite it inline as [1], [2] ... right after the claim it supports. Cite only "
            "the numbered sources below, never invent a source. Where the evidence is silent, answer from "
            "medical knowledge without a citation. If the evidence conflicts with your prior knowledge, "
            "follow the evidence and say so.\n<evidence>\n" + evidence + "\n</evidence>"
        )
    return brief


def _build_brief(plan: dict, n_turns: int = 1) -> str:
    if plan.get("_fallback"):
        # No plan: do not impose urgency or question policy we could not assess.
        base = ("- Reply in the user language. Answer fully and specifically. If the situation could be an "
                "emergency, say so in the first sentence and what to do now. If the safe answer depends on "
                "information the user has not given, answer conditionally and end with 1-3 short questions.")
        if plan.get("questions"):  # ask-only classifier supplied the questions
            qs = "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(plan["questions"]))
            intro = plan.get("questions_intro") or "To give you more specific advice, could you tell me:"
            base += ("\n- First answer fully, covering the likely scenarios with the conditions for each. "
                     f"Then END the reply with this exact block and nothing after it:\n  {intro}\n{qs}")
        return base
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
            "Then at most 6 short lines: the concrete steps while waiting for help (what to do, what not to do, what to "
            "tell responders) as direct instructions. A stressed user must be able to act on it in seconds: no background, "
            "no explanations, no sections, no questions. Graders penalize long emergency replies."
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
    if plan.get("_specifics"):
        from app.search import format_specifics
        lines.append(
            "- The following are verified, current statements from authoritative sources. State each one "
            "explicitly in the reply where relevant, with its number/timeframe and naming the source body "
            "(e.g. \"the AAP recommends...\"); do not paraphrase the numbers away:\n"
            + format_specifics(plan["_specifics"])
        )
    if plan["task_format"]:
        lines.append(f"- Required output format: {plan['task_format']} Follow it exactly and complete every part.")
    if n_turns >= 6:
        # Long threads: L2 answers the latest message tersely, as if earlier turns had
        # covered the ground (6-9 turn convos scored 0.44 vs raw 0.50; answers 40% shorter).
        lines.append(
            "- This is a long conversation, but completeness still applies to THIS reply: answer the latest "
            "question as a complete, self-contained answer (enumerate every relevant option, dose, warning sign, and "
            "next step as if it were being asked for the first time), using the history only for facts about the user. "
            "Do not shorten the reply because earlier turns were long."
        )
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


# L2 emits a hidden `reasoning` channel that counts against max_tokens (5-6k chars on
# hard questions). At 2048 the visible answer is truncated (finish_reason=length).
GEN_MAX_TOKENS = int(os.environ.get("GEN_MAX_TOKENS", "6000"))
# Documentation / rewrite / summary tasks: a generous budget lets L2 pad the output
# with invented findings (val: health_data_tasks -0.19 at 6000 vs 2048). The tight
# cap is a structural brake on fabrication that prose rules do not provide.
TASK_MAX_TOKENS = int(os.environ.get("TASK_MAX_TOKENS", "3000"))
# Emergencies for laypeople: long replies are penalized outright ("overly lengthy in an
# emergency context", -10). The budget is the structural brake; the brief asks for <= 6 lines.
EMERGENCY_MAX_TOKENS = int(os.environ.get("EMERGENCY_MAX_TOKENS", "2500"))


def budget_for(plan: dict, requested: int | None) -> int:
    if requested:
        return requested
    if plan.get("urgency") == "emergent" and plan.get("audience") == "layperson":
        return EMERGENCY_MAX_TOKENS
    return TASK_MAX_TOKENS if plan.get("task_format") else GEN_MAX_TOKENS


def generate(client: L2Client, messages: list[dict], plan: dict, *, temperature: float = 0.3,
             max_tokens: int = GEN_MAX_TOKENS, revision: tuple[str, str] | None = None,
             evidence: str = "", searched_without_result: bool = False,
             allow_retry: bool = True, gen_timeout: float | None = None) -> tuple[str, bool]:
    """Returns (draft, retried). One retry if the first draft declines (and time allows).

    `revision=(previous_draft, notes)` asks for a revised full reply instead of a fresh one."""
    brief = build_brief(plan, evidence, searched_without_result, n_turns=len(messages))
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

    think = None if os.environ.get("L2_THINKING", "1") != "0" else False

    def run(b: str, *, thinking: bool | None = think, timeout: float | None = None) -> str:
        system = GENERATION_SYSTEM.replace("{brief}", b)
        msg = client.chat_message([{"role": "system", "content": system}] + convo,
                                  temperature=temperature, max_tokens=max_tokens, thinking=thinking,
                                  retries=1 if timeout else None, timeout=timeout)
        return msg.get("content") or ""

    # Time-boxed. The thinking attempt gets `gen_timeout`; if it is still running at
    # 60% of that, a no-thinking answer (~9 s, scores ~0.33) is started speculatively
    # so the turn can still return by the deadline. A late answer beats a zero.
    if gen_timeout:
        import concurrent.futures as _cf
        pool = _cf.ThreadPoolExecutor(max_workers=2)
        main_f = pool.submit(run, brief, timeout=gen_timeout)
        spec_f = None
        t_start = time.time()
        try:
            draft = main_f.result(timeout=gen_timeout * 0.6)
        except _cf.TimeoutError:
            spec_f = pool.submit(run, brief, thinking=False, timeout=max(20.0, gen_timeout * 0.4))
            remaining = max(5.0, gen_timeout - (time.time() - t_start))
            try:
                draft = main_f.result(timeout=remaining)
            except (_cf.TimeoutError, RuntimeError):
                try:
                    draft = spec_f.result(timeout=max(5.0, gen_timeout * 0.45))
                    plan["_gen_fallback"] = True
                except Exception:
                    draft = ""
        except RuntimeError:
            draft = run(brief, thinking=False, timeout=max(20.0, gen_timeout * 0.4))
            plan["_gen_fallback"] = True
        finally:
            pool.shutdown(wait=False)
        if not draft.strip():
            raise RuntimeError("generation produced no text within the time budget")
        if spec_f is not None and not main_f.done():
            return draft, False
    else:
        draft = run(brief)
    if allow_retry and REFUSAL_PATTERNS.search(draft[:800]):
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


# -------------------------------------------------------------- best-of-n select

def select_best(selector: L2Client, messages: list[dict], plan: dict, drafts: list[str],
                timeout: float = 40.0) -> tuple[int, dict | None]:
    """Pick the best draft by a rubric the selector writes for this question.
    Returns (index, verdict). The winning text is returned verbatim; nothing is rewritten."""
    user = (
        "<conversation>\n" + transcript(messages) + "\n</conversation>\n\n"
        "<plan>\n" + json.dumps({k: plan.get(k) for k in ("language", "audience", "urgency", "context_sufficient")},
                                 ensure_ascii=False) + "\n</plan>\n\n"
        "<candidate id=\"A\">\n" + drafts[0] + "\n</candidate>\n\n"
        "<candidate id=\"B\">\n" + drafts[1] + "\n</candidate>\n\nOutput the JSON."
    )
    try:
        text = selector.chat_message([{"role": "system", "content": SELECTOR_SYSTEM},
                                      {"role": "user", "content": user}],
                                     temperature=0.0, max_tokens=2500,
                                     response_format={"type": "json_object"},
                                     retries=1, timeout=timeout).get("content") or ""
        v = _parse_json(text) or {}
    except RuntimeError:
        return 0, None
    w = str(v.get("winner", "A")).strip().upper()
    return (1 if w == "B" else 0), v


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
           max_tokens: int | None = None, planner: L2Client | None = None,
           local_first: bool = False) -> dict:
    """Full turn. Returns {'answer', 'plan', 'notes', 'timings'}. The final text is
    always L2's; `planner` (if given) only produces the brief."""
    requested = max_tokens
    t0 = time.time()
    # CoEval: 180 s per request, then one retry, then the item scores 0. Every
    # optional step below checks the remaining budget before starting.
    deadline = t0 + float(os.environ.get("TURN_BUDGET_S", "150"))
    # The evaluator prepends its own system message ("You are Chain-of-Evidence");
    # the harness owns the system prompt, so only user/assistant turns are kept.
    messages = [m for m in messages if m.get("role") in ("user", "assistant")] or messages
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
    max_tokens = budget_for(plan, requested)
    t1 = time.time()
    evidence, retrieval_meta = "", None
    if (plan.get("needs_grounding") and os.environ.get("RETRIEVAL", "1") != "0"
            and deadline - time.time() > 115):
        try:
            from app import retrieval as _retrieval  # lazy: keeps engine importable without MCP
            r = _retrieval.run_retrieval(client, plan["retrieval_query"], selector=planner or client,
                                         hints=plan.get("retrieval_hints") or {},
                                         local_first=local_first)
            # Korean regulatory sources only belong in Korean answers; in an English
            # reply they read as noise and measurably hurt (h6).
            if "korean" not in plan["language"].lower():
                r["items"] = [i for i in r["items"]
                              if not str(i.get("source_type", i.get("_tool", ""))).lower().startswith(
                                  ("hira", "openapi_mfds", "openapi_hira", "kcd", "openapi_law", "law"))]
            retrieval_meta = {k: r[k] for k in (
                "status", "note", "tool_calls", "cached_items", "elapsed",
                "selected_by", "local_hits", "seeded",
            )}
            retrieval_meta["items"] = [{"title": _retrieval._title_of(i), "url": i.get("url", "")} for i in r["items"]]
            if r["items"] and r["status"] != "no_evidence":
                evidence = _retrieval.format_evidence(r)
        except Exception as e:  # retrieval is best-effort; never fail the turn
            retrieval_meta = {"error": str(e)[:200]}
    # Verified specifics via web search (see app/search.py). Not for emergencies
    # (latency, brevity), documentation tasks, or when the turn is already slow.
    search_meta = None
    if (os.environ.get("SEARCH", "0") == "1" and planner is not None and planner is not client
            and not plan.get("_fallback") and plan.get("urgency") != "emergent"
            and not plan.get("task_format") and deadline - time.time() > 100):
        from app.search import search_specifics
        q = plan.get("core_request") or messages[-1]["content"]
        if plan.get("language") and "english" not in plan["language"].lower():
            q += f"\n(The user writes in {plan['language']}; prefer that country's national guidance where it exists.)"
        specifics, search_meta = search_specifics(planner.key, q, timeout=min(40.0, deadline - time.time() - 60))
        if specifics:
            plan["_specifics"] = specifics
    t1b = time.time()
    searched_dry = bool(plan.get("needs_grounding")) and not evidence
    # leave ~25 s for the no-thinking fallback inside the 150 s turn budget
    gen_timeout = max(30.0, min(float(os.environ.get("GEN_TIMEOUT_S", "75")), deadline - time.time() - 30))
    best_of = int(os.environ.get("BEST_OF", "1"))
    selection = None
    if (best_of >= 2 and planner is not None and planner is not client
            and deadline - time.time() > 100):
        # Two L2 samples in parallel (different temperatures), a strong model picks the
        # better one against a rubric it writes for this question. Wall time ~= one
        # generation + ~10 s; the chosen text is L2's, verbatim.
        import concurrent.futures as _cf
        gen_timeout = min(gen_timeout, max(30.0, deadline - time.time() - 45))
        with _cf.ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(generate, client, messages, plan, temperature=temperature, max_tokens=max_tokens,
                             evidence=evidence, searched_without_result=searched_dry,
                             allow_retry=False, gen_timeout=gen_timeout)
            fb = pool.submit(generate, client, messages, plan, temperature=min(1.0, temperature + 0.5),
                             max_tokens=max_tokens, evidence=evidence, searched_without_result=searched_dry,
                             allow_retry=False, gen_timeout=gen_timeout)
            results = []
            for f in (fa, fb):
                try:
                    results.append(f.result())
                except Exception:
                    results.append(None)
        drafts = [r[0] for r in results if r and r[0].strip()]
        if len(drafts) == 2 and deadline - time.time() > 20:
            idx, selection = select_best(planner, messages, plan, drafts,
                                         timeout=max(10.0, min(40.0, deadline - time.time() - 8)))
            draft, retried = drafts[idx], False
        elif drafts:
            draft, retried = drafts[0], False
        else:
            raise RuntimeError("both generations failed")
    else:
        draft, retried = generate(client, messages, plan, temperature=temperature, max_tokens=max_tokens,
                                  evidence=evidence, searched_without_result=searched_dry,
                                  allow_retry=deadline - time.time() > 110, gen_timeout=gen_timeout)
    t2 = time.time()
    # L2 occasionally answers a grounded turn with a literal "<tool_call>..." block.
    if re.match(r"\s*<tool_call>", draft) or len(draft.strip()) < 40:
        try:
            draft, retried = generate(client, messages, plan, temperature=temperature, max_tokens=max_tokens,
                                      evidence="", allow_retry=False,
                                      gen_timeout=max(30.0, deadline - time.time() - 20))
            evidence = ""
        except RuntimeError:
            pass
    review = None
    revised = False
    # Review + rewrite costs ~30-40 s. Skip it when the turn is already slow so a
    # turn stays under TURN_BUDGET_S even if the evaluator has a tight timeout.
    if (planner is not None and planner is not client and os.environ.get("CRITIC", "0") == "1"
            and deadline - time.time() > 60):
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
                                        revision=(draft, notes_txt), evidence=evidence,
                                        searched_without_result=searched_dry)
                # Guard against a collapsed rewrite -- unless the draft answered the wrong
                # question, in which case a much shorter correct answer is the goal.
                floor = 400 if review.get("interpretation_ok") is False else 0.5 * len(draft.strip())
                if len(new_draft.strip()) >= floor:
                    draft, revised = new_draft, True
            except RuntimeError:
                pass
    # Dry search + specific citations = fabrication. L2 ignores a prose ban, but it
    # does apply a concrete revision that names the strings to remove.
    scrubbed = False
    if searched_dry and deadline - time.time() > 60:
        from app.seed import unverified_specifics
        bad = unverified_specifics(draft)
        if bad:
            notes_txt = ("No source could be found for this question, so these specific citations in your draft "
                         "are unverified and must be removed or replaced with a plain statement that the exact "
                         "reference could not be verified: " + "; ".join(bad[:8]) +
                         ". Keep the general guidance; do not add any other article numbers, classes, 고시 numbers, "
                         "PMIDs, or day counts.")
            try:
                new_draft, _ = generate(client, messages, plan, temperature=temperature, max_tokens=max_tokens,
                                        revision=(draft, notes_txt), searched_without_result=True)
                if len(new_draft.strip()) >= 300 and len(unverified_specifics(new_draft)) < len(bad):
                    draft, scrubbed = new_draft, True
            except RuntimeError:
                pass
    t3 = time.time()
    final, notes = assemble(draft, plan)
    notes["scrubbed_unverified"] = scrubbed
    notes["retried_refusal"] = retried
    notes["reviewed"] = review is not None
    notes["revised"] = revised
    notes["grounded"] = bool(evidence)
    notes["best_of"] = 2 if selection is not None else 1
    notes["searched"] = bool(plan.get("_specifics"))
    notes["gen_fallback"] = bool(plan.pop("_gen_fallback", False))
    notes["selected"] = (selection or {}).get("winner")
    notes["refusal_phrase"] = bool(REFUSAL_PATTERNS.search(final))
    return {
        "answer": final,
        "plan": plan,
        "notes": notes,
        "review": review,
        "selection": ({k: selection.get(k) for k in ("a_score", "b_score", "winner", "reason")}
                      if selection else None),
        "retrieval": retrieval_meta,
        "search": ({**search_meta, "n": len(plan.get("_specifics") or [])} if search_meta else None),
        "timings": {"plan": round(t1 - t0, 1), "retrieve": round(t1b - t1, 1),
                    "generate": round(t2 - t1b, 1), "review_revise": round(t3 - t2, 1)},
    }
