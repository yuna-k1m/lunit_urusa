#!/usr/bin/env python3
"""Run a HealthBench evaluation locally and score it with the official rubric grader.

Baseline sanity check before any L2 harness exists. `--mode raw` is the honest
floor: one chat completion, no retrieval, no prompt engineering, exactly the
config simple-evals uses for its published numbers (system message "You are a
helpful assistant.", temperature 0.5, max_tokens 2048).

    python tools/run_eval.py --n 20                       # raw gpt-4.1 on 20 hard examples
    python tools/run_eval.py --n 50 --split full
    python tools/run_eval.py --model gpt-4.1-mini --grader gpt-4.1-mini --n 10
    python tools/run_eval.py --mode raw --candidate-base https://model.hackathon.lunit.io \\
        --candidate-key-env LUNIT_FM_API_KEY --model Lunit/L2-preview --n 20

Scoring is `healthbench_eval.calculate_score` reimplemented exactly:
positives-only denominator, per-example score unclipped, final mean clipped to
[0,1]. Grader prompt is GRADER_TEMPLATE verbatim.

Reads OPENAI_API_KEY from the environment or a .env file at the repo root.
Stdlib only. Results land in results/<run-name>/.
"""

from __future__ import annotations

import argparse
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

# One global gate around every API call. The example pool and the per-rubric pool
# are nested, so without this the real concurrency is jobs**2 and blows the TPM cap.
API_GATE = threading.Semaphore(8)

# per-1M-token USD, for the cost estimate only
PRICES = {
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
    p.add_argument("--mode", default="raw", choices=["raw"], help="raw = no harness")
    p.add_argument("--model", default="gpt-4.1", help="candidate model")
    p.add_argument("--grader", default="gpt-4.1", help="grader model")
    p.add_argument("--candidate-base", default="https://api.openai.com")
    p.add_argument("--candidate-key-env", default="OPENAI_API_KEY")
    p.add_argument("--grader-base", default="https://api.openai.com")
    p.add_argument("--grader-key-env", default="OPENAI_API_KEY")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max-tokens", type=int, default=2048)
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
    if not args.dry_run and not (cand_key and grade_key):
        sys.exit(
            f"missing ${args.candidate_key_env} / ${args.grader_key_env} "
            "(put it in .env at the repo root)"
        )

    rows = load_split(args.split)
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    n_rubrics = sum(len(x["rubrics"]) for x in sample)

    print(f"split={args.split}  n={len(sample)}  rubrics={n_rubrics}")
    print(f"candidate={args.model} @ {args.candidate_base}   mode={args.mode}")
    print(f"grader={args.grader} @ {args.grader_base}")

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
        ci, co = price_of(args.model)
        gi, go = price_of(args.grader)
        cost = (c_in * ci + c_out * co + g_in * gi + g_out * go) / 1e6
        print(f"\nwould make {len(sample):,} completion + {n_rubrics:,} grading calls")
        print(f"  candidate  ~{c_in / 1e6:.2f}M in / {c_out / 1e6:.2f}M out   ({args.model})")
        print(f"  grader     ~{g_in / 1e6:.2f}M in / {g_out / 1e6:.2f}M out   ({args.grader})")
        print(f"  estimated cost  ~${cost:,.2f}"
              + ("" if any(price_of(m) != (0.0, 0.0) for m in (args.model, args.grader))
                 else "  (no price table entry - grader/candidate assumed free)"))
        print(f"  assuming {args.assume_answer_chars} chars per answer; "
              f"at --jobs {args.jobs} expect roughly "
              f"{(len(sample) + n_rubrics) / max(args.jobs, 1) * 1.2 / 60:.0f} min if never rate-limited")
        return 0

    cand_usage, grade_usage = Usage(), Usage()
    done = [0]
    lock = threading.Lock()

    # Global in-flight cap. The example pool and the per-rubric pool are nested, so
    # without this the real concurrency is jobs**2 and blows the provider's TPM cap.
    global API_GATE
    API_GATE = threading.Semaphore(args.jobs)

    run_name = args.name or (
        f"{datetime.now():%Y%m%d-%H%M%S}-{args.mode}-{args.model}-{args.split}{len(sample)}"
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
        messages = [{"role": "system", "content": BASELINE_SYSTEM}] + convo
        answer = chat(
            args.candidate_base,
            cand_key,
            args.model,
            messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            usage=cand_usage,
        )
        with_response = convo + [{"role": "assistant", "content": answer}]
        convo_str = "\n\n".join(f"{m['role']}: {m['content']}" for m in with_response)

        def grade_one(r: dict) -> dict:
            prompt = template.replace("<<conversation>>", convo_str).replace(
                "<<rubric_item>>", f"[{r['points']}] {r['criterion']}"
            )
            for _ in range(3):
                out = chat(
                    args.grader_base,
                    grade_key,
                    args.grader,
                    [{"role": "user", "content": prompt}],
                    temperature=args.temperature,
                    max_tokens=1024,
                    usage=grade_usage,
                )
                g = parse_grade(out)
                if isinstance(g.get("criteria_met"), bool):
                    return g
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

    ci, co = price_of(args.model)
    gi, go = price_of(args.grader)
    cost = (
        cand_usage.inp * ci + cand_usage.out * co + grade_usage.inp * gi + grade_usage.out * go
    ) / 1e6

    # results.jsonl was written incrementally as each example finished; only the
    # summary is emitted here.
    summary = {
        "mode": args.mode,
        "candidate_model": args.model,
        "grader_model": args.grader,
        "split": args.split,
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
    print(f"OVERALL  {overall:.3f}   ({args.mode} {args.model} on {args.split}, n={len(sample)})")
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
