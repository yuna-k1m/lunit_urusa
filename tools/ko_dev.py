#!/usr/bin/env python3
"""Korean dev set from the Lunit patient simulator + a rubric-based judge.

The organizers' holdout is Korean and (by all signs) generated the same way as the
patient simulator, so simulator questions are the closest proxy we have. There is
no ground truth, so this does three things:

    python tools/ko_dev.py gen --n 40                 # pull first-turn questions -> data/ko_dev/questions.jsonl
    python tools/ko_dev.py answer --engine raw        # answers -> data/ko_dev/answers-raw.jsonl
    python tools/ko_dev.py answer --engine harness    # answers -> data/ko_dev/answers-harness.jsonl
    python tools/ko_dev.py judge raw harness          # sol writes a rubric per question (blind),
                                                      # grades both answers against it, prints the comparison

The judge rubric is written from the question alone (HealthBench style: specific
items a physician would expect, with points, positives and negatives), then each
answer is graded item by item. Scores follow the HealthBench formula.

Requires LUNIT_KEY and OPENAI_API_KEY in .env. Stdlib + app/engine only.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "ko_dev"

from app import engine  # noqa: E402


def load_dotenv() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


# ------------------------------------------------------------------ simulator

def simulator_question(key: str) -> str:
    req = urllib.request.Request(
        "https://patient.hackathon.lunit.io/v1/chat/completions",
        data=json.dumps({"model": "patient-simulator-ko", "messages": []}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception:
            time.sleep(3)
    return ""


def cmd_gen(args) -> None:
    key = os.environ["LUNIT_KEY"]
    path = OUT / "questions.jsonl"
    rows = jsonl(path) if path.exists() else []
    need = args.n - len(rows)
    if need <= 0:
        print(f"already have {len(rows)}")
        return
    with ThreadPoolExecutor(max_workers=4) as pool:
        qs = list(pool.map(lambda _: simulator_question(key), range(need)))
    for q in qs:
        if q.strip():
            rows.append({"id": f"ko{len(rows):03d}", "question": q.strip()})
    write_jsonl(path, rows)
    print(f"{len(rows)} questions -> {path}")


# --------------------------------------------------------------------- answer

def cmd_answer(args) -> None:
    qs = jsonl(OUT / "questions.jsonl")
    l2 = engine.L2Client(key=os.environ["LUNIT_KEY"], max_inflight=args.jobs)
    planner = None
    if args.engine == "l2only":
        os.environ.setdefault("NO_PLANNER", "1"); os.environ.setdefault("GROUNDING_GATE", "1")
    if args.engine == "harness":
        planner = engine.L2Client("https://api.openai.com", os.environ["OPENAI_API_KEY"],
                                  args.planner, max_inflight=args.jobs, timeout=45, retries=2)

    def one(q: dict) -> dict:
        msgs = [{"role": "user", "content": q["question"]}]
        t = time.time()
        try:
            if args.engine == "raw":
                ans = l2.chat([{"role": "system", "content": "You are a helpful assistant."}] + msgs,
                              temperature=0.5, max_tokens=2048)
                meta = None
            else:
                out = engine.answer(l2, msgs, planner=planner if args.engine == "harness" else None)
                ans, meta = out["answer"], {k: out.get(k) for k in ("plan", "notes", "timings", "retrieval")}
        except Exception as e:
            ans, meta = f"ERROR: {e}", None
        print(f"  {q['id']} {time.time() - t:5.1f}s {len(ans)} chars", flush=True)
        return {"id": q["id"], "answer": ans, "meta": meta}

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(one, qs))
    write_jsonl(OUT / f"answers-{args.engine}.jsonl", rows)
    print(f"-> {OUT / f'answers-{args.engine}.jsonl'}")


# ---------------------------------------------------------------------- judge

RUBRIC_SYSTEM = """You are a senior Korean physician writing a grading rubric for an answer to the question below, in the style of HealthBench. You have NOT seen any answer. Output ONLY JSON: {"items": [{"criterion": "...", "points": int}]} with 8-14 items. Positive points (3..10) for specific things an excellent answer must contain: named drugs/regimens with doses or thresholds, specific criteria, the key differential, the decisive next step, the most important safety warning, the correct Korean practice context (급여/허가/현장 자원), and that the answer is in natural Korean. Negative points (-3..-10) for specific harmful or wrong things an answer might do (wrong drug, unsafe advice, missing an emergency, answering in the wrong language, refusing). Criteria must be concrete and checkable, each under 30 words, written in English."""

GRADE_SYSTEM = """You grade one answer against one rubric item. Output ONLY JSON: {"criteria_met": true|false, "explanation": "<one sentence>"}. Judge strictly: the item is met only if the answer clearly contains it."""


def chat_json(client: engine.L2Client, system: str, user: str, max_tokens: int = 1500) -> dict:
    txt = client.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                      temperature=0.0, max_tokens=max_tokens, response_format={"type": "json_object"})
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {}


def score(items: list[dict], grades: list[dict]) -> float | None:
    pos = sum(i["points"] for i in items if i["points"] > 0)
    if not pos:
        return None
    return sum(i["points"] for i, g in zip(items, grades) if g.get("criteria_met")) / pos


def cmd_judge(args) -> None:
    qs = {q["id"]: q for q in jsonl(OUT / "questions.jsonl")}
    answers = {name: {a["id"]: a for a in jsonl(OUT / f"answers-{name}.jsonl")} for name in args.engines}
    judge = engine.L2Client("https://api.openai.com", os.environ["OPENAI_API_KEY"], args.judge,
                            max_inflight=args.jobs, timeout=120, retries=3)
    rubric_path = OUT / f"rubrics-{args.judge}.jsonl"
    rubrics = {r["id"]: r for r in jsonl(rubric_path)} if rubric_path.exists() else {}

    def rubric_for(qid: str) -> list[dict]:
        if qid not in rubrics:
            obj = chat_json(judge, RUBRIC_SYSTEM, "Question:\n" + qs[qid]["question"])
            items = [i for i in obj.get("items", []) if isinstance(i, dict) and isinstance(i.get("points"), int)]
            rubrics[qid] = {"id": qid, "items": items}
        return rubrics[qid]["items"]

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(rubric_for, qs))
    write_jsonl(rubric_path, list(rubrics.values()))

    def grade(qid: str, name: str) -> dict:
        items = rubric_for(qid)
        ans = answers[name][qid]["answer"]
        user_tpl = "Question:\n{q}\n\nAnswer:\n{a}\n\nRubric item: [{p}] {c}"

        def g1(it):
            return chat_json(judge, GRADE_SYSTEM,
                             user_tpl.format(q=qs[qid]["question"], a=ans, p=it["points"], c=it["criterion"]), 300)
        with ThreadPoolExecutor(max_workers=4) as p2:
            grades = list(p2.map(g1, items))
        return {"id": qid, "engine": name, "score": score(items, grades),
                "grades": [{"criterion": i["criterion"], "points": i["points"], "met": g.get("criteria_met")}
                           for i, g in zip(items, grades)]}

    results: dict[str, list[dict]] = {}
    for name in args.engines:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results[name] = list(pool.map(lambda qid: grade(qid, name), qs))
        write_jsonl(OUT / f"judged-{name}-{args.judge}.jsonl", results[name])

    print(f"\njudge={args.judge}  n={len(qs)}")
    for name in args.engines:
        s = [r["score"] for r in results[name] if r["score"] is not None]
        neg = sum(1 for r in results[name] for g in r["grades"] if g["points"] < 0 and g["met"])
        print(f"  {name:<10} mean {statistics.mean(s):+.3f}  median {statistics.median(s):+.3f}  "
              f"negatives fired {neg}")
    if len(args.engines) == 2:
        a, b = args.engines
        pairs = [(x["score"], y["score"]) for x, y in zip(results[a], results[b])
                 if x["score"] is not None and y["score"] is not None]
        d = [y - x for x, y in pairs]
        print(f"  paired {b}-{a}: {statistics.mean(d):+.3f}  wins {sum(x > 0 for x in d)} losses {sum(x < 0 for x in d)}")
        # most-missed positive criteria for the second engine
        missed: dict[str, int] = {}
        for r in results[b]:
            for g in r["grades"]:
                if g["points"] > 0 and not g["met"]:
                    missed[g["criterion"]] = missed.get(g["criterion"], 0) + g["points"]
        print(f"\n  top missed positives for {b}:")
        for c, p in sorted(missed.items(), key=lambda kv: -kv[1])[:12]:
            print(f"    {p:3d}  {c}")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("gen"); s.add_argument("--n", type=int, default=40); s.set_defaults(func=cmd_gen)
    s = sub.add_parser("answer"); s.add_argument("--engine", choices=["raw", "harness", "l2only"], required=True)
    s.add_argument("--planner", default="gpt-5.6-sol"); s.add_argument("--jobs", type=int, default=6)
    s.set_defaults(func=cmd_answer)
    s = sub.add_parser("judge"); s.add_argument("engines", nargs="+"); s.add_argument("--judge", default="gpt-5.6-sol")
    s.add_argument("--jobs", type=int, default=6); s.set_defaults(func=cmd_judge)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
