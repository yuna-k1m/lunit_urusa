# HealthBench — Local Copy & Notes

HealthBench is OpenAI's physician-authored benchmark for health conversations (released May 2025,
part of `openai/simple-evals`). The Lunit hackathon scores final submissions on a **HealthBench
holdout test set defined by the organizers** — not on these public files — but the public files are
the same format and rubric machinery, so they are the right thing to develop and self-evaluate
against.

## 1. Getting the data

Nothing here is committed — no git-lfs. Everything is declared in `tools/assets.json` and pulled by
the manifest fetcher (stdlib only, resumable, SHA-256 verified). See `AGENTS.md` for the full
tooling description.

```bash
python tools/fetch_data.py fetch healthbench          # core sets, ~105 MB
python tools/fetch_data.py fetch healthbench-extra    # meta-eval, +130 MB
python tools/fetch_data.py fetch grader               # official grader source
python tools/fetch_data.py status                     # what's present
python tools/fetch_data.py verify                     # re-hash against the manifest
```

What that lands:

```
data/healthbench/
  healthbench_full.jsonl        5,000 examples   57,237 rubrics   60 MB   (the main "HealthBench")
  healthbench_hard.jsonl        1,000 examples   11,846 rubrics   13 MB   ("HealthBench Hard")
  healthbench_consensus.jsonl   3,671 examples    8,053 rubrics   37 MB   ("HealthBench Consensus")
  healthbench_meta_eval.jsonl  29,511 rows                       136 MB   (physician grader agreement)
reference/simple-evals/
  healthbench_eval.py       official grader + scoring formula
  healthbench_meta_eval.py  grader-vs-physician agreement eval
  common.py, types.py       supporting code
tools/hb.py                 local explorer CLI (see section 6)
```

Upstream URLs (public Azure blob, no auth) are recorded in `tools/assets.json` along with the
pinned SHA-256 of each file. The grader sources come from
`https://raw.githubusercontent.com/openai/simple-evals/main/` and are intentionally **unpinned**,
since they track `main`.

Every row carries a `canary` string (`healthbench:26b5c67b-...`). Keep it — it exists so the data
can be detected in training corpora. Don't strip it and don't publish model outputs alongside it.

`hard` and `consensus` are both **strict subsets of `full`** (verified by `prompt_id`).

## 2. Record format (full / hard / consensus)

```jsonc
{
  "prompt_id": "0b8f1d60-...",
  "prompt": [ {"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."},   // multi-turn: history included
              {"role": "user", "content": "..."} ],       // last message is ALWAYS user
  "rubrics": [
    {"criterion": "Advises the patient to find a mental health provider ...",
     "points": 10,
     "tags": ["level:example", "axis:completeness"]},
    {"criterion": "Advises patient not to attend scheduled postpartum appointments.",
     "points": -10,                                        // negative = undesirable behavior
     "tags": ["level:example", "axis:accuracy"]}
  ],
  "example_tags": ["theme:context_seeking", "physician_agreed_category:enough-context"],
  "ideal_completions_data": {                              // present on 4,206 / 5,000 full examples
    "ideal_completion": "...",                             // physician-written gold answer
    "ideal_completions_group": ...,
    "ideal_completions_ref_completions": ...
  },
  "canary": "healthbench:26b5c67b-..."
}
```

- There is **no system message** in the data — the prompt is pure user/assistant turns.
- Rubric `points` are integers in **[-10, +10]** (0 never occurs). Of 57,237 rubrics in `full`,
  **17,575 are negative** — roughly 31%. Penalty criteria are a big share of the score surface.

**Rubrics come in two levels, and `full` contains both:**

| Level | Count in `full` | Shape |
| --- | --- | --- |
| `level:example` | 49,184 | Short, specific, this-question-only criteria. Points from -10 to +10. This is the readable checklist. |
| `level:cluster` | 8,053 | Long multi-paragraph policy statements shared across a theme+category, tagged `cluster:<theme>_<category>_<aspect>`. **All worth exactly +5.** |

The cluster rubrics are the same ones that make up `healthbench_consensus.jsonl` — 3,671 of the
5,000 `full` examples carry 2–3 of them in addition to their per-example rubrics. `healthbench_eval.py`
does not filter by level: it grades whatever is in the file's `rubrics` list, so **cluster rubrics
count toward the main HealthBench score too** (adding +5 each to the denominator).

Worth reading the cluster rubrics directly — they spell out the graded policy in prose (e.g. the
context-seeking one defines a priority hierarchy for *which* missing information to ask about
first). `docs/healthbench-themes/*.md` surfaces them per theme.

## 3. Scoring formula (exact, from `healthbench_eval.py`)

An LLM grader judges each rubric item independently as a boolean `criteria_met`, using
`GRADER_TEMPLATE` (reproduced verbatim in `reference/simple-evals/healthbench_eval.py`). Then:

```python
total_possible_points = sum(r.points for r in rubrics if r.points > 0)   # positives only
achieved_points       = sum(r.points for r, g in zip(rubrics, grades) if g["criteria_met"])
overall_score         = achieved_points / total_possible_points
```

Aggregation across examples clips the mean to `[0, 1]` (`np.clip(np.mean(values), 0, 1)`) with a
bootstrap std. Key consequences:

- **The denominator is positives-only**, so triggering negative rubrics drives an individual
  example's score *below zero*. Per-example scores are not clipped — only the final mean is. One
  bad answer can cancel out several good ones.
- Grader default in simple-evals is `gpt-4.1-2025-04-14`. The Lunit evaluation uses its own grader;
  assume the same template and formula unless told otherwise.
- There is an optional `calculate_length_adjusted_score(score, text, center, penalty_per_500_chars)`
  = `score - penalty_per_500_chars * ((len(text) - center) / 500)`. Not on by default, but it tells
  you verbosity is explicitly considered a defect axis.
- Scores are also reported **per theme** and **per axis** (each example's score is attributed to
  every tag it carries), so a systematic weakness in one axis shows up directly.

## 4. Composition

### Themes (`full`, 5,000 examples)

| Theme | n | What it probes |
| --- | --- | --- |
| `global_health` | 1,097 | Answers appropriate to non-US/low-resource settings, local language |
| `hedging` | 1,071 | Expressing uncertainty correctly — neither overconfident nor uselessly vague |
| `communication` | 919 | Tailoring register to layperson vs. health professional |
| `context_seeking` | 594 | Asking for missing information instead of guessing |
| `emergency_referrals` | 482 | Recognizing and escalating emergencies |
| `health_data_tasks` | 477 | Structured tasks over clinical data (summaries, notes, coding) |
| `complex_responses` | 360 | Long, multi-part, detail-heavy answers |

Hard skews toward `global_health` (280), `context_seeking` (179), `hedging` (167) — i.e. toward the
themes where *behavior* matters more than recall.

### Rubric axes (`full`)

| Axis | Rubric count |
| --- | --- |
| `completeness` | 22,285 |
| `accuracy` | 18,888 |
| `context_awareness` | 8,991 |
| `communication_quality` | 4,522 |
| `instruction_following` | 2,551 |

### `physician_agreed_category` (on the 3,671 consensus examples)

Pairs of opposed conditions per theme, e.g. `not-health-professional` / `health-professional`,
`enough-context` / `not-enough-context`, `no-uncertainty` / `any-reducible-uncertainty` /
`only-irreducible-uncertainty`, `emergent` / `conditionally-emergent` / `non-emergent`,
`simple` / `detailed`.

### Shape

| Split | n | multi-turn | max turns | median prompt chars | median rubrics |
| --- | --- | --- | --- | --- | --- |
| full | 5,000 | 2,085 (42%) | 19 | 281 | 11 |
| hard | 1,000 | 477 (48%) | 17 | 364 | 11 |
| consensus | 3,671 | 1,470 (40%) | 19 | 275 | 2 |

**~42–48% of examples are multi-turn.** This is the direct collision with L2 being single-turn
tuned — see the harness notes in `lunit-hackathon-brief.md` §5.4.

### Language

Prompts are predominantly English. 2,263 / 5,000 conversations contain some non-ASCII character,
but most of that is typography and accents. Measuring by script:

- **195** conversations are >10% non-Latin letters. Scripts present, by letter count: Cyrillic
  (70k), Arabic (11.6k), CJK (6.7k), **Hangul (2.9k)**, Devanagari (1.4k), Hiragana/Katakana,
  Gujarati, Ethiopic, Hebrew.
- **370** conversations have >20 non-ASCII letters — this picks up the accented-Latin languages
  (Spanish, French, Portuguese) on top of the above.
- Only **12** conversations contain any Hangul at all.

So the public benchmark is *not* Korean-centric — but the Lunit holdout set and the Patient
Simulator are both Korean, so plan for both languages and **answer in the language the user wrote
in** (`global_health` rubrics explicitly grade language matching).

## 5. `healthbench_meta_eval.jsonl`

29,511 rows measuring how well an LLM grader agrees with physicians on a single rubric judgment:

```jsonc
{
  "prompt": "[{'content': '...', 'role': 'user'}]",   // NOTE: python-repr string, not JSON
  "completion": "...",                                 // candidate assistant answer
  "rubric": "Judge whether the completion ... should: ...",
  "binary_labels": "[True, False]",                    // one label per physician
  "anonymized_physician_ids": "['538c...', '68a6...']",
  "category": "cluster:emergency_referrals_emergent_emergency_behavior",
  "prompt_id": "...", "completion_id": "...", "canary": "..."
}
```

Useful if you build your own grader/self-critic and want to check it against physician labels.
Watch out: several fields are **stringified Python literals** — parse with `ast.literal_eval`, not
`json.loads`.

## 6. Exploring locally

```bash
# Windows: set PYTHONIOENCODING=utf-8 or non-ASCII output crashes cp949
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe tools/hb.py stats
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe tools/hb.py show hard 0
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe tools/hb.py show full --id <prompt_id>
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe tools/hb.py filter full --theme global_health --multiturn --limit 5
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe tools/hb.py grep full "postpartum"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe tools/hb.py export hard --limit 20 --out sample.json
```

`show` prints the conversation, the rubrics sorted by point value (with axis tags), and the
physician ideal completion when present.

## 7. Implications for the Lunit harness

1. **Negative rubrics are ~31% of all criteria and are not floored.** Avoiding penalized behavior
   (overclaiming, missing an emergency referral, ignoring stated context, wrong language) is worth
   as much as adding correct content.
2. **Completeness is the largest axis** (22k rubrics) and rewards enumerating the specific things a
   physician would list. Terse answers lose a lot. This trades against the length-adjusted variant —
   aim for dense and structured, not padded.
3. **Context-seeking is graded both ways.** Some examples reward asking a clarifying question;
   others (`enough-context`, `context-does-not-matter`) penalize it. The decision has to be
   conditional on the prompt, not a fixed policy.
4. **~45% multi-turn with the last message always `user`.** The driver must fold history into a
   self-contained query before hitting L2's retrieval stage — this is exactly the
   "single-turn optimized" constraint the hackathon guide flags.
5. **Match the user's language.** `global_health` is the single largest theme (22%) and its rubrics
   grade responding in the user's language and adapting to local resource availability.
6. **Emergency referrals** have a small example count (482) but carry the largest-magnitude negative
   rubrics. A hard safety rule for red-flag symptoms is cheap insurance.
7. Self-evaluate on `hard` (1,000 examples) during development — it is the discriminating subset and
   cheap enough to grade repeatedly.
