"""Tiny local explorer for the HealthBench JSONL files in data/healthbench/.

Usage (from repo root, with the venv python):
    .venv/Scripts/python.exe tools/hb.py stats
    .venv/Scripts/python.exe tools/hb.py show hard 0
    .venv/Scripts/python.exe tools/hb.py show full --id <prompt_id>
    .venv/Scripts/python.exe tools/hb.py filter full --theme global_health --multiturn --limit 5
    .venv/Scripts/python.exe tools/hb.py grep full "postpartum"
    .venv/Scripts/python.exe tools/hb.py export hard --limit 20 --out sample.json

On Windows set PYTHONIOENCODING=utf-8 first, or output with non-ASCII text will crash cp949.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "healthbench"

SPLITS = {
    "full": DATA / "healthbench_full.jsonl",
    "hard": DATA / "healthbench_hard.jsonl",
    "consensus": DATA / "healthbench_consensus.jsonl",
    "meta": DATA / "healthbench_meta_eval.jsonl",
}


def load(split: str) -> list[dict]:
    path = SPLITS[split]
    if not path.exists():
        sys.exit(f"missing {path} — see docs/healthbench-notes.md for download URLs")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tags_of(ex: dict, prefix: str) -> list[str]:
    return [t.split(":", 1)[1] for t in ex.get("example_tags", []) if t.startswith(prefix + ":")]


def render(ex: dict, show_rubrics: bool = True) -> str:
    out = [f"prompt_id: {ex['prompt_id']}", f"tags: {', '.join(ex.get('example_tags', []))}", ""]
    for m in ex["prompt"]:
        out.append(f"[{m['role']}]")
        out.append(m["content"])
        out.append("")
    if show_rubrics:
        rubrics = sorted(ex["rubrics"], key=lambda r: -r["points"])
        out.append(f"--- {len(rubrics)} rubrics "
                   f"(positive total = {sum(r['points'] for r in rubrics if r['points'] > 0)}) ---")
        for r in rubrics:
            axes = ",".join(t.split(":", 1)[1] for t in r["tags"] if t.startswith("axis:"))
            out.append(f"  [{r['points']:+3}] ({axes}) {r['criterion']}")
    ideal = (ex.get("ideal_completions_data") or {}).get("ideal_completion")
    if ideal:
        out += ["", "--- physician ideal completion (truncated) ---", ideal[:1500]]
    return "\n".join(out)


def cmd_stats(_args) -> None:
    for split in ("full", "hard", "consensus"):
        rows = load(split)
        turns = [len(x["prompt"]) for x in rows]
        chars = [sum(len(m["content"]) for m in x["prompt"]) for x in rows]
        rub = [len(x["rubrics"]) for x in rows]
        mt = sum(1 for t in turns if t > 1)
        print(f"\n== {split}: {len(rows)} examples, {sum(rub)} rubrics")
        print(f"   multi-turn {mt} ({mt / len(rows):.0%}), max turns {max(turns)}")
        print(f"   median prompt chars {int(statistics.median(chars))}, median rubrics {int(statistics.median(rub))}")
        themes = collections.Counter(t for x in rows for t in tags_of(x, "theme"))
        print("   themes: " + ", ".join(f"{k}={v}" for k, v in themes.most_common()))
        axes = collections.Counter(
            t.split(":", 1)[1] for x in rows for r in x["rubrics"] for t in r["tags"] if t.startswith("axis:")
        )
        if axes:
            print("   axes:   " + ", ".join(f"{k}={v}" for k, v in axes.most_common()))


def cmd_show(args) -> None:
    rows = load(args.split)
    if args.id:
        ex = next((x for x in rows if x["prompt_id"] == args.id), None)
        if ex is None:
            sys.exit(f"prompt_id {args.id} not found in {args.split}")
    else:
        ex = rows[args.index]
    print(render(ex, show_rubrics=not args.no_rubrics))


def cmd_filter(args) -> None:
    rows = load(args.split)
    for ex in rows:
        if args.theme and args.theme not in tags_of(ex, "theme"):
            continue
        if args.category and args.category not in tags_of(ex, "physician_agreed_category"):
            continue
        if args.multiturn and len(ex["prompt"]) == 1:
            continue
        if args.singleturn and len(ex["prompt"]) > 1:
            continue
        print(render(ex, show_rubrics=not args.no_rubrics))
        print("=" * 100)
        args.limit -= 1
        if args.limit <= 0:
            return


def cmd_grep(args) -> None:
    rows = load(args.split)
    needle = args.needle.lower()
    hits = 0
    for ex in rows:
        text = " ".join(m["content"] for m in ex["prompt"])
        if needle in text.lower():
            hits += 1
            first = ex["prompt"][0]["content"].replace("\n", " ")[:160]
            print(f"{ex['prompt_id']}  turns={len(ex['prompt'])}  {first}")
            if hits >= args.limit:
                break
    print(f"\n{hits} match(es) shown")


def cmd_export(args) -> None:
    rows = load(args.split)
    if args.theme:
        rows = [x for x in rows if args.theme in tags_of(x, "theme")]
    rows = rows[: args.limit]
    out = Path(args.out)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} examples -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats").set_defaults(func=cmd_stats)

    s = sub.add_parser("show")
    s.add_argument("split", choices=SPLITS)
    s.add_argument("index", nargs="?", type=int, default=0)
    s.add_argument("--id")
    s.add_argument("--no-rubrics", action="store_true")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("filter")
    s.add_argument("split", choices=SPLITS)
    s.add_argument("--theme")
    s.add_argument("--category")
    s.add_argument("--multiturn", action="store_true")
    s.add_argument("--singleturn", action="store_true")
    s.add_argument("--no-rubrics", action="store_true")
    s.add_argument("--limit", type=int, default=3)
    s.set_defaults(func=cmd_filter)

    s = sub.add_parser("grep")
    s.add_argument("split", choices=SPLITS)
    s.add_argument("needle")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_grep)

    s = sub.add_parser("export")
    s.add_argument("split", choices=SPLITS)
    s.add_argument("--theme")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--out", default="healthbench_sample.json")
    s.set_defaults(func=cmd_export)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
