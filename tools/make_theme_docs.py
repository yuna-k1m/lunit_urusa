#!/usr/bin/env python3
"""Generate one human-readable Markdown file per HealthBench theme.

    python tools/make_theme_docs.py              # -> docs/healthbench-themes/
    python tools/make_theme_docs.py --examples 6 --out docs/hb

Each theme file contains: what the theme tests, how it is distributed across the
splits, which rubric axes dominate, the most common reward/penalty phrasings
mined from the rubrics, and a handful of full worked examples (conversation +
every rubric, sorted by point value + the physician's ideal answer).

Output is deterministic - examples are picked by sorted prompt_id - so
regenerating produces no spurious diffs.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "healthbench"

# Theme name -> (Korean title, what it probes, why it is hard)
THEMES = {
    "global_health": (
        "글로벌 헬스",
        "미국 외 지역·저자원 환경에 맞는 답변, 사용자 언어로의 응답",
        "가장 큰 테마(22%)이며 rubric이 **사용자가 쓴 언어로 답했는지**와 현지에서 실제로 구할 수 "
        "있는 자원을 전제했는지를 직접 채점한다. 미국 기준 진료 경로를 그대로 답하면 감점된다.",
    ),
    "hedging": (
        "불확실성 표현",
        "확신할 수 없는 것을 과신하지도, 쓸모없이 얼버무리지도 않기",
        "두 방향 모두 감점 대상이다. `no-uncertainty` 예시에서 헤징하면 감점, "
        "`any-reducible-uncertainty` 예시에서 단정하면 감점. 질문 자체가 어느 쪽인지 판단해야 한다.",
    ),
    "communication": (
        "커뮤니케이션",
        "상대가 일반인인지 의료 전문가인지에 따라 어휘·깊이 조정",
        "같은 내용이라도 대상에 따라 다른 답이 정답이다. `health-professional` 예시에서 "
        "지나치게 쉬운 설명을 하거나, 일반인에게 전문용어를 쏟아내면 감점.",
    ),
    "context_seeking": (
        "맥락 탐색",
        "정보가 부족할 때 추측 대신 되묻기",
        "**양방향 채점.** `not-enough-context`는 되묻기를 보상하지만 "
        "`enough-context`·`context-does-not-matter`는 되묻기를 감점한다. 고정 정책이 불가능하다.",
    ),
    "emergency_referrals": (
        "응급 이송 판단",
        "응급 상황을 인지하고 즉시 상급 조치를 안내",
        "예시 수는 적지만(482개) 음수 rubric의 절대값이 가장 크다. 놓치면 단일 문항 점수가 "
        "깊은 마이너스로 내려간다. 반대로 비응급을 응급으로 몰아도 감점.",
    ),
    "health_data_tasks": (
        "임상 데이터 작업",
        "요약·기록 작성·코딩 등 구조화된 과제 수행",
        "instruction_following 비중이 높다. 요청한 형식·항목을 그대로 지켰는지가 "
        "내용 정확도만큼 중요하다.",
    ),
    "complex_responses": (
        "복합 응답",
        "다항목·장문·세부사항이 많은 답변",
        "completeness rubric이 촘촘하다. 빠뜨린 항목마다 점수가 깎이지만, "
        "length-adjusted 채점 변형이 존재하므로 늘려 쓰는 것으로는 해결되지 않는다.",
    ),
}

AXIS_KO = {
    "accuracy": "정확성",
    "completeness": "완결성",
    "context_awareness": "맥락 인식",
    "communication_quality": "전달 품질",
    "instruction_following": "지시 준수",
}


def load(name: str) -> list[dict]:
    path = DATA / f"healthbench_{name}.jsonl"
    if not path.exists():
        sys.exit(f"missing {path}\nrun: python tools/fetch_data.py fetch healthbench")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def theme_of(ex: dict) -> str | None:
    for t in ex.get("example_tags", []):
        if t.startswith("theme:"):
            return t.split(":", 1)[1]
    return None


def categories_of(ex: dict) -> list[str]:
    return [
        t.split(":", 1)[1]
        for t in ex.get("example_tags", [])
        if t.startswith("physician_agreed_category:")
    ]


def axes_of(rubric: dict) -> list[str]:
    return [t.split(":", 1)[1] for t in rubric["tags"] if t.startswith("axis:")]


def is_cluster(rubric: dict) -> bool:
    """Cluster-level rubrics are the long meta-prompts shared by HealthBench Consensus.

    `full` carries both: 49,184 per-example rubrics and 8,053 cluster rubrics
    (all worth +5). They are graded together, but only the per-example ones are
    useful to read as a checklist.
    """
    return "level:cluster" in rubric["tags"]


def example_rubrics(ex: dict) -> list[dict]:
    return [r for r in ex["rubrics"] if not is_cluster(r)]


def cluster_rubrics(ex: dict) -> list[dict]:
    return [r for r in ex["rubrics"] if is_cluster(r)]


def common_openings(rubrics: list[dict], n: int = 10) -> list[tuple[str, int]]:
    """Mine the most frequent opening verb phrases from rubric criteria."""
    counter: collections.Counter = collections.Counter()
    for r in rubrics:
        words = re.findall(r"[A-Za-z']+", r["criterion"])[:3]
        if words:
            counter[" ".join(words)] += 1
    return counter.most_common(n)


def fmt_conversation(ex: dict) -> str:
    out = []
    for m in ex["prompt"]:
        who = "**User**" if m["role"] == "user" else "**Assistant** (기존 답변)"
        body = m["content"].strip()
        out.append(f"{who}\n\n" + "\n".join("> " + line for line in body.splitlines()))
    return "\n\n".join(out)


def fmt_rubrics(ex: dict) -> str:
    rows = ["| 점수 | 축 | 기준 |", "| ---: | --- | --- |"]
    for r in sorted(example_rubrics(ex), key=lambda r: -r["points"]):
        axes = ", ".join(AXIS_KO.get(a, a) for a in axes_of(r))
        crit = r["criterion"].replace("|", "\\|").replace("\n", " ")
        rows.append(f"| **{r['points']:+d}** | {axes} | {crit} |")
    pos = sum(r["points"] for r in ex["rubrics"] if r["points"] > 0)
    neg = sum(r["points"] for r in ex["rubrics"] if r["points"] < 0)
    rows.append("")
    rows.append(f"만점 분모 = {pos}점 (양수만 합산) · 감점 가능 총량 = {neg}점")

    clusters = cluster_rubrics(ex)
    if clusters:
        rows.append("")
        rows.append("<details>")
        rows.append(
            f"<summary>cluster 단위 rubric {len(clusters)}개 (각 +5, consensus 평가와 "
            "공유되는 긴 메타 기준 — 위 분모에 이미 포함되어 있다)</summary>"
        )
        rows.append("")
        for r in clusters:
            rows.append(f"**[{r['points']:+d}]**")
            rows.append("")
            rows.append("\n".join("> " + ln for ln in r["criterion"].strip().splitlines()))
            rows.append("")
        rows.append("</details>")
    return "\n".join(rows)


def pick_examples(pool: list[dict], hard_ids: set[str], k: int) -> list[dict]:
    """Deterministic, varied selection: short single-turn, multi-turn, has-ideal, hard."""
    pool = sorted(pool, key=lambda x: x["prompt_id"])
    picked: list[dict] = []

    def take(pred):
        for ex in pool:
            if ex in picked:
                continue
            if pred(ex):
                picked.append(ex)
                return

    take(lambda x: len(x["prompt"]) == 1 and x.get("ideal_completions_data"))
    take(lambda x: len(x["prompt"]) >= 3)
    take(lambda x: x["prompt_id"] in hard_ids)
    take(lambda x: len(x["prompt"]) >= 5)
    for ex in pool:
        if len(picked) >= k:
            break
        if ex not in picked:
            picked.append(ex)
    return picked[:k]


def render_theme(
    theme: str, full: list[dict], hard_ids: set[str], cons_ids: set[str], n_examples: int
) -> str:
    ko_title, probes, why_hard = THEMES[theme]
    pool = [x for x in full if theme_of(x) == theme]
    rubrics = [r for x in pool for r in x["rubrics"] if not is_cluster(r)]
    n_cluster = sum(len(cluster_rubrics(x)) for x in pool)
    pos = [r for r in rubrics if r["points"] > 0]
    neg = [r for r in rubrics if r["points"] < 0]
    n_hard = sum(1 for x in pool if x["prompt_id"] in hard_ids)
    n_cons = sum(1 for x in pool if x["prompt_id"] in cons_ids)
    n_mt = sum(1 for x in pool if len(x["prompt"]) > 1)
    axis_counts = collections.Counter(a for r in rubrics for a in axes_of(r))
    cat_counts = collections.Counter(c for x in pool for c in categories_of(x))

    L: list[str] = []
    A = L.append
    A(f"# {ko_title} · `{theme}`")
    A("")
    A(f"> {probes}")
    A("")
    A("## 한눈에")
    A("")
    A("| 항목 | 값 |")
    A("| --- | --- |")
    A(f"| full 내 예시 수 | {len(pool):,}개 ({len(pool) / len(full):.0%}) |")
    A(f"| 그중 hard | {n_hard:,}개 |")
    A(f"| 그중 consensus | {n_cons:,}개 |")
    A(f"| 멀티턴 | {n_mt:,}개 ({n_mt / len(pool):.0%}) |")
    A(f"| 예시별 rubric | {len(rubrics):,}개 (양수 {len(pos):,} / 음수 {len(neg):,}) |")
    A(f"| 음수 비중 | {len(neg) / len(rubrics):.0%} |")
    A(f"| 예시당 평균 rubric | {len(rubrics) / len(pool):.1f}개 |")
    A(f"| cluster 단위 rubric | {n_cluster:,}개 (각 +5, consensus와 공유) |")
    A("")
    A("**왜 어려운가.** " + why_hard)
    A("")
    A("> rubric은 두 층위로 존재한다. **예시별 rubric**(`level:example`)은 이 문항 전용의 "
      "구체적 체크리스트이고, **cluster rubric**(`level:cluster`)은 consensus 평가와 공유되는 "
      "긴 서술형 메타 기준으로 전부 +5점이다. 채점 시 둘 다 분모에 들어가지만, "
      "읽고 체크리스트로 쓸 수 있는 것은 전자다.")
    A("")

    A("## 채점 축 분포")
    A("")
    A("| 축 | rubric 수 | 비중 |")
    A("| --- | ---: | ---: |")
    for axis, cnt in axis_counts.most_common():
        A(f"| {AXIS_KO.get(axis, axis)} (`{axis}`) | {cnt:,} | {cnt / len(rubrics):.0%} |")
    A("")

    if cat_counts:
        A("## 하위 조건 (`physician_agreed_category`)")
        A("")
        A("같은 테마 안에서도 조건에 따라 **정답 행동이 반대**가 된다.")
        A("")
        for cat, cnt in cat_counts.most_common():
            A(f"- `{cat}` — {cnt}개")
        A("")

    A("## rubric이 자주 요구하는 것 / 벌하는 것")
    A("")
    A("예시별(`level:example`) rubric 기준문의 첫 어구를 빈도순으로 뽑은 것. "
      "답변 체크리스트로 쓸 수 있다.")
    A("")
    A(f"**보상 (양수 {len(pos):,}개 중 상위 어구)**")
    A("")
    for phrase, cnt in common_openings(pos, 12):
        A(f"- `{phrase}...` × {cnt}")
    A("")
    A(f"**감점 (음수 {len(neg):,}개 중 상위 어구)**")
    A("")
    for phrase, cnt in common_openings(neg, 12):
        A(f"- `{phrase}...` × {cnt}")
    A("")

    A("## 최고 배점 / 최대 감점 기준 실물")
    A("")
    A("**+10점짜리 기준 (요구사항의 핵심)**")
    A("")
    top = sorted(pos, key=lambda r: -r["points"])[:6]
    for r in top:
        A(f"- [{r['points']:+d}] {r['criterion']}")
    A("")
    A("**-10점짜리 기준 (절대 하면 안 되는 것)**")
    A("")
    bottom = sorted(neg, key=lambda r: r["points"])[:6]
    for r in bottom:
        A(f"- [{r['points']:+d}] {r['criterion']}")
    A("")

    A("## 예시")
    A("")
    for i, ex in enumerate(pick_examples(pool, hard_ids, n_examples), 1):
        marks = []
        if ex["prompt_id"] in hard_ids:
            marks.append("hard")
        if ex["prompt_id"] in cons_ids:
            marks.append("consensus")
        if len(ex["prompt"]) > 1:
            marks.append(f"{len(ex['prompt'])}턴")
        cats = categories_of(ex)
        if cats:
            marks.append(" / ".join(f"`{c}`" for c in cats))
        suffix = f" — {', '.join(marks)}" if marks else ""
        A(f"### 예시 {i}{suffix}")
        A("")
        A(f"`{ex['prompt_id']}`")
        A("")
        A(fmt_conversation(ex))
        A("")
        A("#### 채점 기준")
        A("")
        A(fmt_rubrics(ex))
        A("")
        ideal = (ex.get("ideal_completions_data") or {}).get("ideal_completion")
        if ideal:
            snippet = ideal.strip()
            truncated = len(snippet) > 2000
            snippet = snippet[:2000] + ("\n\n…(이하 생략)" if truncated else "")
            A("<details>")
            A("<summary>의사가 작성한 모범 답안 (ideal_completion)</summary>")
            A("")
            A(snippet)
            A("")
            A("</details>")
            A("")
        A("---")
        A("")

    A("전체 예시는 `tools/hb.py`로 직접 열람:")
    A("")
    A("```bash")
    A(f"PYTHONIOENCODING=utf-8 python tools/hb.py filter full --theme {theme} --limit 5")
    A("```")
    A("")
    return "\n".join(L)


def render_index(full: list[dict], hard_ids: set[str], out: Path) -> str:
    L: list[str] = []
    A = L.append
    A("# HealthBench 문제 유형별 정리")
    A("")
    A("HealthBench 5,000개 예시는 7개 테마로 나뉜다. 각 테마는 *다른 능력*을 채점하며,")
    A("일부는 서로 상충한다 (되묻기를 보상하는 테마와 벌하는 테마가 동시에 존재).")
    A("아래 파일은 각 테마가 무엇을 요구하는지, 실제 rubric과 예시가 어떻게 생겼는지를 정리한 것.")
    A("")
    A("채점 공식과 데이터 포맷은 `../healthbench-notes.md`, 플랫폼 전반은")
    A("`../lunit-hackathon-brief.md` 참고.")
    A("")
    A("| 테마 | 예시 수 | hard | 멀티턴 | 무엇을 보는가 |")
    A("| --- | ---: | ---: | ---: | --- |")
    for theme, (ko, probes, _) in THEMES.items():
        pool = [x for x in full if theme_of(x) == theme]
        n_hard = sum(1 for x in pool if x["prompt_id"] in hard_ids)
        n_mt = sum(1 for x in pool if len(x["prompt"]) > 1)
        A(
            f"| [{ko}](./{theme}.md) `{theme}` | {len(pool):,} | {n_hard} "
            f"| {n_mt / len(pool):.0%} | {probes} |"
        )
    A("")
    A("## 읽는 순서 제안")
    A("")
    A("1. **`context_seeking`** — 되묻기 정책을 정하는 문제. 양방향 채점이라 하니스 설계에 "
      "가장 먼저 영향을 준다.")
    A("2. **`global_health`** — 가장 큰 테마. 언어 매칭 규칙이 여기서 나온다.")
    A("3. **`emergency_referrals`** — 감점 폭이 가장 크다. 안전 규칙의 하한선.")
    A("4. **`hedging`** — 확신 표현 톤을 정한다.")
    A("5. 나머지 (`communication`, `health_data_tasks`, `complex_responses`) — 형식과 분량 정책.")
    A("")
    A("## 재생성")
    A("")
    A("```bash")
    A("python tools/fetch_data.py fetch healthbench   # 데이터가 없다면")
    A("python tools/make_theme_docs.py")
    A("```")
    A("")
    A("생성물은 벤치마크 본문을 그대로 담고 있어 `.gitignore` 대상이다 "
      "(canary 문자열이 붙은 데이터의 재배포를 피하기 위함). 각자 로컬에서 생성해 읽을 것.")
    A("")
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="docs/healthbench-themes")
    p.add_argument("--examples", type=int, default=4, help="worked examples per theme")
    args = p.parse_args()

    full = load("full")
    hard_ids = {x["prompt_id"] for x in load("hard")}
    cons_ids = {x["prompt_id"] for x in load("consensus")}

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    (out / "README.md").write_text(render_index(full, hard_ids, out), encoding="utf-8")
    print(f"  wrote {args.out}/README.md")
    for theme in THEMES:
        text = render_theme(theme, full, hard_ids, cons_ids, args.examples)
        (out / f"{theme}.md").write_text(text, encoding="utf-8")
        print(f"  wrote {args.out}/{theme}.md ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
