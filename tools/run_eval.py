#!/usr/bin/env python3
"""Run registered chat strategies or raw APIs on HealthBench and grade them.

Strategy mode is the standard comparison path and defaults to the canonical,
UUID-ordered fixed 100 from HealthBench Hard. Raw and legacy harness modes remain
available to reproduce earlier measurements.

    python tools/run_eval.py --strategy baseline_l2 --grader gpt-5.6-sol --resume
    python tools/run_eval.py --strategy siusiubeom_h4 --grader gpt-5.6-sol --resume
    python tools/run_eval.py --strategy multi_patient_sol --grader gpt-5.6-sol --jobs 4 --resume
    python tools/run_eval.py --mode raw --subset tune --n 20 --model gpt-4.1 --grader gpt-4.1

Scoring is `healthbench_eval.calculate_score` reimplemented exactly:
positives-only denominator, per-example score unclipped, final mean clipped to
[0,1]. Grader prompt is GRADER_TEMPLATE verbatim.

Credentials resolve from the environment/.env; GPT Sol grading can also use the
bundled submission_openai_key. Strategy mode uses the same registry and bundled
credential fallbacks as the server. Results land in results/<run-name>/.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "healthbench"
GRADER_SRC = ROOT / "reference" / "simple-evals" / "healthbench_eval.py"

BASELINE_SYSTEM = "You are a helpful assistant."

# Frozen dev splits of `hard`. TUNE is exactly the sample the n=200 baseline was
# measured on (Random(0).sample(hard, 200)); VAL is disjoint and should only be
# graded to confirm a change that already won on TUNE, never to pick prompts.
TUNE_N, TUNE_SEED = 200, 0
VAL_N, VAL_SEED = 200, 1

# One global gate around every API call. The example pool and the per-rubric pool
# are nested, so without this the real concurrency is jobs**2 and blows the TPM cap.
API_GATE = threading.Semaphore(8)

# per-1M-token USD, for the cost estimate only
PRICES = {
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


def price_of(model: str) -> tuple[float, float]:
    # longest prefix wins, else "gpt-4.1" would swallow "gpt-4.1-mini"
    for k in sorted(PRICES, key=len, reverse=True):
        if model.startswith(k):
            return PRICES[k]
    return (0.0, 0.0)


# ------------------------------------------------------------------- env / data


def load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_split(name: str) -> list[dict]:
    path = DATA / f"healthbench_{name}.jsonl"
    if not path.exists():
        sys.exit(f"missing {path}\nrun: python tools/fetch_data.py fetch healthbench")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def is_non_latin(ex: dict, frac: float = 0.10) -> bool:
    """True if >frac of the alphabetic characters in the prompt are outside Latin scripts."""
    text = " ".join(m["content"] for m in ex["prompt"])
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if ord(c) > 0x24F) / len(letters) > frac


def pick_sample(rows: list[dict], subset: str, n: int, seed: int, slice_: str) -> list[dict]:
    if subset == "conquer_val":
        ids = set(json.loads((ROOT / "tools" / "conquer_val_ids.json").read_text(encoding="utf-8"))["prompt_ids"])
        pool = [x for x in rows if x["prompt_id"] in ids]
    elif subset == "fixed100":
        if len(rows) < 100:
            raise ValueError("fixed100 requires at least 100 rows")
        pool = sorted(rows, key=lambda row: row["prompt_id"])[:100]
    elif subset == "tune":
        pool = random.Random(TUNE_SEED).sample(rows, TUNE_N)
    elif subset == "val":
        tune_ids = {x["prompt_id"] for x in random.Random(TUNE_SEED).sample(rows, TUNE_N)}
        rest = [x for x in rows if x["prompt_id"] not in tune_ids]
        pool = random.Random(VAL_SEED).sample(rest, VAL_N)
    else:
        pool = rows
    if slice_ == "nonlatin":
        pool = [x for x in pool if is_non_latin(x)]
    elif slice_ == "multiturn":
        pool = [x for x in pool if len(x["prompt"]) > 1]
    elif slice_ == "longturn":
        pool = [x for x in pool if len(x["prompt"]) >= 6]
    if subset == "fixed100" or n >= len(pool):
        return pool
    return random.Random(seed).sample(pool, n)


def grader_template() -> str:
    """Pull GRADER_TEMPLATE verbatim out of the official source, so it can't drift."""
    if not GRADER_SRC.exists():
        sys.exit(f"missing {GRADER_SRC}\nrun: python tools/fetch_data.py fetch grader")
    src = GRADER_SRC.read_text(encoding="utf-8")
    m = re.search(r'GRADER_TEMPLATE = """(.*?)"""\.strip\(\)', src, re.S)
    if not m:
        sys.exit("could not extract GRADER_TEMPLATE from healthbench_eval.py")
    return m.group(1).strip()


# ----------------------------------------------------------------------- client


class Usage:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.inp = 0
        self.out = 0
        self.calls = 0

    def add(self, i: int, o: int) -> None:
        with self.lock:
            self.inp += i
            self.out += o
            self.calls += 1


def chat(
    base: str,
    key: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    usage: Usage,
    retries: int = 10,
) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    delay = 2.0
    for attempt in range(retries):
        try:
            with API_GATE:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = json.loads(r.read())
            u = data.get("usage") or {}
            usage.add(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
            return data["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = delay + random.uniform(0, 1.0)
                hdr = e.headers.get("Retry-After") if e.headers else None
                if hdr:
                    try:
                        wait = max(wait, float(hdr))
                    except ValueError:
                        pass
                time.sleep(min(wait, 60.0))
                delay = min(delay * 2, 60.0)
                continue
            raise RuntimeError(f"HTTP {e.code} from {model}: {detail}")
        except OSError as e:
            if attempt < retries - 1:
                time.sleep(min(delay + random.uniform(0, 1.0), 60.0))
                delay = min(delay * 2, 60.0)
                continue
            raise RuntimeError(f"{model}: {e}")
    raise RuntimeError(f"{model}: exhausted {retries} retries")


def parse_grade(text: str) -> dict:
    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {}


def responses_grade(
    base: str,
    key: str,
    model: str,
    prompt: str,
    *,
    reasoning_effort: str,
    usage: Usage,
    retries: int = 5,
) -> dict:
    """Grade with the Responses API and a strict HealthBench verdict schema."""
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "explanation": {"type": "string"},
            "criteria_met": {"type": "boolean"},
        },
        "required": ["explanation", "criteria_met"],
    }
    body = json.dumps({
        "model": model,
        "input": prompt,
        "reasoning": {"effort": reasoning_effort},
        "text": {"format": {
            "type": "json_schema", "name": "healthbench_rubric_grade",
            "strict": True, "schema": schema,
        }},
        "store": False,
    }).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/responses",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    delay = 2.0
    for attempt in range(retries):
        try:
            with API_GATE:
                with urllib.request.urlopen(req, timeout=180) as response:
                    data = json.loads(response.read())
            token_usage = data.get("usage") or {}
            usage.add(token_usage.get("input_tokens", 0), token_usage.get("output_tokens", 0))
            text = data.get("output_text")
            if not text:
                for item in data.get("output", []):
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            text = content.get("text")
                            break
            grade = parse_grade(text or "")
            if isinstance(grade.get("criteria_met"), bool):
                return grade
            raise ValueError("grader returned no boolean criteria_met")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(min(delay, 60.0))
                delay *= 2
                continue
            raise RuntimeError(f"HTTP {exc.code} from {model}: {detail}") from exc
        except (OSError, ValueError) as exc:
            if attempt < retries - 1:
                time.sleep(min(delay, 60.0))
                delay *= 2
                continue
            raise RuntimeError(f"{model}: {exc}") from exc
    raise RuntimeError(f"{model}: exhausted {retries} retries")


# ------------------------------------------------------------------------ score


def calculate_score(rubrics: list[dict], grades: list[dict]) -> float | None:
    """Verbatim reimplementation of healthbench_eval.calculate_score."""
    total_possible = sum(r["points"] for r in rubrics if r["points"] > 0)
    if total_possible == 0:
        return None
    achieved = sum(
        r["points"] for r, g in zip(rubrics, grades) if g.get("criteria_met")
    )
    return achieved / total_possible


def tag_values(tag_scores: dict[str, list[float]]) -> dict[str, float]:
    return {k: max(0.0, min(1.0, statistics.mean(v))) for k, v in tag_scores.items()}


# ------------------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--split", default="hard", choices=["hard", "full", "consensus"])
    p.add_argument("--n", type=int, default=20, help="examples to sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", default="strategy", choices=["raw", "harness", "strategy"],
                   help="strategy = registered chat model (standard); raw = one API call; harness = legacy app.engine")
    p.add_argument("--strategy", default="baseline_l2",
                   help="registered model name used by --mode strategy")
    p.add_argument("--system", default=None,
                   help="path to a system prompt file (default: the simple-evals baseline message)")
    p.add_argument("--subset", default="fixed100", choices=["all", "fixed100", "tune", "val", "conquer_val"],
                   help="fixed100 = canonical UUID-ordered 100; tune/val = frozen disjoint 200s; "
                        "conquer_val = the 301 published leaderboard ids (needs --split full)")
    p.add_argument("--slice", default="all", choices=["all", "nonlatin", "multiturn", "longturn"],
                   help="filter the pool before sampling")
    p.add_argument("--model", default="gpt-4.1", help="candidate model")
    p.add_argument("--grader", default="gpt-4.1", help="grader model")
    p.add_argument("--grader-api", default="auto", choices=["auto", "chat", "responses"],
                   help="auto uses Responses for gpt-5.6 and Chat Completions otherwise")
    p.add_argument("--grader-reasoning-effort", default="medium")
    p.add_argument("--candidate-base", default="https://api.openai.com")
    p.add_argument("--candidate-key-env", default="OPENAI_API_KEY")
    p.add_argument("--grader-base", default="https://api.openai.com")
    p.add_argument("--grader-key-env", default="OPENAI_API_KEY")
    p.add_argument("--planner-model", default=None,
                   help="harness mode: model that writes the plan (default: the candidate itself)")
    p.add_argument("--planner-base", default="https://api.openai.com")
    p.add_argument("--planner-key-env", default="OPENAI_API_KEY")
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--name", default=None, help="results subdirectory name")
    p.add_argument("--resume", action="store_true",
                   help="skip prompt_ids already present in the run's results.jsonl")
    p.add_argument("--dry-run", action="store_true", help="estimate size/cost and stop")
    p.add_argument("--assume-answer-chars", type=int, default=1667,
                   help="answer length assumed by --dry-run (measured mean of the raw gpt-4.1 baseline)")
    args = p.parse_args()

    load_dotenv()
    cand_key = os.environ.get(args.candidate_key_env)
    grade_key = os.environ.get(args.grader_key_env)
    if not grade_key and (ROOT / "submission_openai_key").exists():
        grade_key = (ROOT / "submission_openai_key").read_text(encoding="utf-8").strip()
        if grade_key.startswith("b64:"):
            import base64
            grade_key = base64.b64decode(grade_key[4:]).decode().strip()
    candidate_key_required = args.mode == "raw"
    if not args.dry_run and (not grade_key or (candidate_key_required and not cand_key)):
        sys.exit(
            f"missing required ${args.candidate_key_env} / ${args.grader_key_env} credentials"
        )

    rows = load_split(args.split)
    if args.subset in ("fixed100", "tune", "val") and args.split != "hard":
        sys.exit("--subset fixed100/tune/val are defined on --split hard only")
    if args.subset == "conquer_val" and args.split != "full":
        sys.exit("--subset conquer_val needs --split full")
    sample = pick_sample(rows, args.subset, args.n, args.seed, args.slice)
    n_rubrics = sum(len(x["rubrics"]) for x in sample)
    system_prompt = (
        Path(args.system).read_text(encoding="utf-8").strip() if args.system else BASELINE_SYSTEM
    )

    print(f"split={args.split}  subset={args.subset}  slice={args.slice}  "
          f"n={len(sample)}  rubrics={n_rubrics}")
    print(f"system={'baseline' if not args.system else args.system} ({len(system_prompt)} chars)")
    candidate_label = args.strategy if args.mode == "strategy" else args.model
    print(f"candidate={candidate_label} @ {args.candidate_base}   mode={args.mode}")
    grader_api = args.grader_api
    if grader_api == "auto":
        grader_api = "responses" if args.grader.startswith("gpt-5.6") else "chat"
    print(f"grader={args.grader} @ {args.grader_base} ({grader_api})")

    template = grader_template()

    if args.dry_run:
        # Estimate from actual prompt sizes. ~4 chars/token for English; Korean and
        # other non-Latin scripts run closer to 2, so this is a floor for those.
        tpl_tok = len(template) / 4
        ans_tok = args.assume_answer_chars / 4
        g_in = g_out = c_in = 0.0
        for ex in sample:
            convo_tok = sum(len(m["content"]) for m in ex["prompt"]) / 4
            c_in += convo_tok + 10
            for r in ex["rubrics"]:
                g_in += tpl_tok + convo_tok + ans_tok + len(r["criterion"]) / 4
                g_out += 95  # measured mean of the grader's JSON verdict
        c_out = len(sample) * ans_tok
        ci, co = (0.0, 0.0) if args.mode == "strategy" else price_of(args.model)
        gi, go = price_of(args.grader)
        cost = (c_in * ci + c_out * co + g_in * gi + g_out * go) / 1e6
        print(f"\nwould make {len(sample):,} completion + {n_rubrics:,} grading calls")
        print(f"  candidate  ~{c_in / 1e6:.2f}M in / {c_out / 1e6:.2f}M out   ({candidate_label})")
        print(f"  grader     ~{g_in / 1e6:.2f}M in / {g_out / 1e6:.2f}M out   ({args.grader})")
        print(f"  estimated cost  ~${cost:,.2f}"
              + ("" if any(price_of(m) != (0.0, 0.0) for m in (candidate_label, args.grader))
                 else "  (no price table entry - grader/candidate assumed free)"))
        print(f"  assuming {args.assume_answer_chars} chars per answer; "
              f"at --jobs {args.jobs} expect roughly "
              f"{(len(sample) + n_rubrics) / max(args.jobs, 1) * 1.2 / 60:.0f} min if never rate-limited")
        return 0

    cand_usage, grade_usage = Usage(), Usage()
    done = [0]

    engine_client = None
    strategy_factory = None
    if args.mode == "harness":
        sys.path.insert(0, str(ROOT))
        from app import engine  # noqa: E402

        engine_client = engine.L2Client(
            args.candidate_base, cand_key, args.model, max_inflight=args.jobs
        )
        planner_client = None
        if args.planner_model:
            planner_client = engine.L2Client(
                args.planner_base, os.environ.get(args.planner_key_env, ""),
                args.planner_model, max_inflight=args.jobs,
            )
            print(f"planner={args.planner_model} @ {args.planner_base}")
    elif args.mode == "strategy":
        sys.path.insert(0, str(ROOT))
        from chat_models.base import ChatRequest as FrameworkChatRequest  # noqa: E402
        from chat_models.factory import build_registry  # noqa: E402
        from config import Settings  # noqa: E402

        def strategy_factory():
            return build_registry(Settings.from_env()).create(args.strategy)

        # Fail before starting paid grading if the name is invalid.
        strategy_factory()
    lock = threading.Lock()

    # Global in-flight cap. The example pool and the per-rubric pool are nested, so
    # without this the real concurrency is jobs**2 and blows the provider's TPM cap.
    global API_GATE
    API_GATE = threading.Semaphore(args.jobs)

    run_name = args.name or (
        f"{datetime.now():%Y%m%d-%H%M%S}-{args.mode}-{candidate_label}-{args.split}{len(sample)}"
    )
    outdir = ROOT / "results" / run_name.replace("/", "_")
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"

    # Resume: grading is the expensive half, so a partial run must never be thrown away.
    prior: list[dict] = []
    if args.resume and results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            prior = [json.loads(line) for line in f if line.strip()]
        seen = {r["prompt_id"] for r in prior}
        before = len(sample)
        sample = [x for x in sample if x["prompt_id"] not in seen]
        print(f"resuming: {len(prior)} already graded, {len(sample)}/{before} left")
    sink = results_path.open("a" if args.resume else "w", encoding="utf-8")

    def run_example(ex: dict) -> dict:
        convo = [dict(m) for m in ex["prompt"]]
        if args.subset == "conquer_val":
            convo = [{"role": "system", "content": "You are Chain-of-Evidence"}] + convo
        harness_meta = None
        if strategy_factory is not None:
            strategy = strategy_factory()
            result = asyncio.run(strategy.complete(FrameworkChatRequest(
                messages=convo,
                original_payload={"temperature": args.temperature, "max_tokens": args.max_tokens},
            )))
            answer = result.content
            harness_meta = result.metadata
        elif engine_client is not None:
            out = engine.answer(
                engine_client, convo, temperature=args.temperature, max_tokens=args.max_tokens,
                planner=planner_client,
            )  # None -> engine default (6000)
            answer = out["answer"]
            harness_meta = {k: out.get(k) for k in ("plan", "notes", "timings", "review", "retrieval", "selection", "search")}
        else:
            messages = [{"role": "system", "content": system_prompt}] + convo
            answer = chat(
                args.candidate_base,
                cand_key,
                args.model,
                messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens or 2048,
                usage=cand_usage,
            )
        with_response = convo + [{"role": "assistant", "content": answer}]
        convo_str = "\n\n".join(f"{m['role']}: {m['content']}" for m in with_response)

        def grade_one(r: dict) -> dict:
            prompt = template.replace("<<conversation>>", convo_str).replace(
                "<<rubric_item>>", f"[{r['points']}] {r['criterion']}"
            )
            if grader_api == "responses":
                return responses_grade(
                    args.grader_base, grade_key, args.grader, prompt,
                    reasoning_effort=args.grader_reasoning_effort, usage=grade_usage,
                )
            for _ in range(3):
                output = chat(
                    args.grader_base, grade_key, args.grader,
                    [{"role": "user", "content": prompt}], temperature=args.temperature,
                    max_tokens=1024, usage=grade_usage,
                )
                grade_result = parse_grade(output)
                if isinstance(grade_result.get("criteria_met"), bool):
                    return grade_result
            return {"criteria_met": False, "explanation": "grader failed to return valid JSON"}

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            grades = list(pool.map(grade_one, ex["rubrics"]))

        score = calculate_score(ex["rubrics"], grades)
        record = {
            "prompt_id": ex["prompt_id"],
            "example_tags": ex["example_tags"],
            "score": score,
            "answer": answer,
            "answer_chars": len(answer),
            "harness": harness_meta,
            "rubric_grades": [
                {
                    "points": r["points"],
                    "criterion": r["criterion"],
                    "tags": r["tags"],
                    "criteria_met": g.get("criteria_met"),
                    "explanation": g.get("explanation", ""),
                }
                for r, g in zip(ex["rubrics"], grades)
            ],
        }
        with lock:
            done[0] += 1
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            print(f"  [{done[0]}/{len(sample)}] {ex['prompt_id'][:8]} score={score:+.3f}",
                  flush=True)
        return record

    started = time.time()
    failures = 0
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_example, ex) for ex in sample]
            fresh = []
            for fut in futures:
                try:
                    fresh.append(fut.result())
                except Exception as e:  # one example dying must not lose the rest
                    failures += 1
                    print(f"  ! example failed: {e}", flush=True)
    finally:
        sink.close()
    elapsed = time.time() - started
    if engine_client is not None:
        cand_usage.add(engine_client.usage["in"], engine_client.usage["out"])
        cand_usage.calls = engine_client.usage["calls"]
        if planner_client is not None:
            pu = planner_client.usage
            print(f"  planner {args.planner_model}: {pu['in']:,}in/{pu['out']:,}out over {pu['calls']} calls")
    results = prior + fresh
    if failures:
        print(f"\n  {failures} example(s) failed; scoring the {len(results)} that completed")
    if not results:
        sys.exit("no examples completed")

    scores = [r["score"] for r in results if r["score"] is not None]
    overall = max(0.0, min(1.0, statistics.mean(scores)))

    theme_scores: dict[str, list[float]] = collections.defaultdict(list)
    axis_scores: dict[str, list[float]] = collections.defaultdict(list)
    for r in results:
        for t in r["example_tags"]:
            if t.startswith("theme:"):
                theme_scores[t.split(":", 1)[1]].append(r["score"])
        per_axis: dict[str, list[dict]] = collections.defaultdict(list)
        for g in r["rubric_grades"]:
            for t in g["tags"]:
                if t.startswith("axis:"):
                    per_axis[t.split(":", 1)[1]].append(g)
        for axis, items in per_axis.items():
            s = calculate_score(items, items)
            if s is not None:
                axis_scores[axis].append(s)

    ci, co = (0.0, 0.0) if args.mode == "strategy" else price_of(args.model)
    gi, go = price_of(args.grader)
    cost = (
        cand_usage.inp * ci + cand_usage.out * co + grade_usage.inp * gi + grade_usage.out * go
    ) / 1e6

    # results.jsonl was written incrementally as each example finished; only the
    # summary is emitted here.
    summary = {
        "mode": args.mode,
        "candidate_model": candidate_label,
        "planner_model": args.planner_model if args.mode == "harness" else None,
        "grader_model": args.grader,
        "split": args.split,
        "subset": args.subset,
        "slice": args.slice,
        "system_prompt_file": args.system,
        "n": len(results),
        "seed": args.seed,
        "overall_score": overall,
        "mean_unclipped": statistics.mean(scores),
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
        "negative_examples": sum(1 for s in scores if s < 0),
        "mean_answer_chars": statistics.mean(r["answer_chars"] for r in results),
        "by_theme": tag_values(theme_scores),
        "by_axis": tag_values(axis_scores),
        "elapsed_sec": round(elapsed, 1),
        "candidate_tokens": {"in": cand_usage.inp, "out": cand_usage.out, "calls": cand_usage.calls},
        "grader_tokens": {"in": grade_usage.inp, "out": grade_usage.out, "calls": grade_usage.calls},
        "est_cost_usd": round(cost, 3),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 62)
    print(f"OVERALL  {overall:.3f}   ({args.mode} {candidate_label} on {args.split}, n={len(sample)})")
    print("=" * 62)
    print(f"  mean (unclipped) {statistics.mean(scores):+.3f}   median {statistics.median(scores):+.3f}")
    print(f"  range {min(scores):+.3f} .. {max(scores):+.3f}   negative examples {summary['negative_examples']}/{len(scores)}")
    print(f"  mean answer length {summary['mean_answer_chars']:.0f} chars")
    print("\n  by theme")
    for k, v in sorted(theme_scores.items(), key=lambda kv: statistics.mean(kv[1])):
        print(f"    {k:<22} {statistics.mean(v):+.3f}  (n={len(v)})")
    print("\n  by axis")
    for k, v in sorted(axis_scores.items(), key=lambda kv: statistics.mean(kv[1])):
        print(f"    {k:<22} {statistics.mean(v):+.3f}")
    print(f"\n  {elapsed:.0f}s  |  candidate {cand_usage.inp:,}in/{cand_usage.out:,}out  "
          f"grader {grade_usage.inp:,}in/{grade_usage.out:,}out  |  ~${cost:.2f}")
    print(f"  -> results/{outdir.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
