# Baseline Results

Measured floors for HealthBench, before any harness exists. Every number here came from
`tools/run_eval.py`; raw per-rubric output lives in `results/<run>/` (gitignored — rerun to
reproduce).

**`--mode raw` means no harness**: one chat completion, system message `"You are a helpful
assistant."`, temperature 0.5, max_tokens 2048 — the exact config `simple-evals` uses for its
published numbers. No retrieval, no prompt engineering, no MCP tools.

Grading is `GRADER_TEMPLATE` verbatim from `reference/simple-evals/healthbench_eval.py` with
`calculate_score` reimplemented exactly (positives-only denominator, per-example score unclipped,
final mean clipped to [0,1]).

---

## Headline

| candidate | split | n | seed | grader | score | 95% CI |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `Lunit/L2-preview` | hard | 200 | 0 | gpt-4.1 | **0.190** | [0.145, 0.238] |
| `gpt-4.1` | hard | 20 | 0 | gpt-4.1 | 0.161 | — |
| `gpt-4.1` | hard | 10 | 0 | gpt-4.1 | 0.231 | — |
| `Lunit/L2-preview` | hard | 10 | 0 | gpt-4.1 | 0.129 | — |

CI is a 2,000-resample bootstrap of the per-example mean.

**The n=10 and n=20 rows are not comparable to anything**, including each other — gpt-4.1 moved
0.231 → 0.161 between n=10 and n=20 on overlapping samples. Only the n=200 row has an error bar
worth quoting. A paired gpt-4.1 run at n=200 costs ~$9 if the head-to-head is ever needed.

---

## Raw L2, HealthBench Hard, n=200

```
OVERALL  0.190          95% CI [0.145, 0.238]
  mean (unclipped)  +0.190
  median            +0.212
  range             -1.500 .. +1.000
  below zero        49/200 examples (24%)
  mean answer       2,953 chars
  runtime           566s at --jobs 12
  cost              $9.57   (candidate free; 100% of spend is OpenAI grading)
  tokens            candidate 33,926 in / 228,145 out over 200 calls
                    grader 3,821,752 in / 240,508 out over 2,459 calls
```

### By theme

| theme | n | mean | median | below zero |
| --- | ---: | ---: | ---: | ---: |
| `hedging` | 33 | +0.134 | +0.196 | 10 |
| `complex_responses` | 15 | +0.154 | +0.138 | 5 |
| `context_seeking` | 43 | +0.179 | +0.172 | 10 |
| `global_health` | 51 | +0.193 | +0.272 | 11 |
| `communication` | 20 | +0.228 | +0.300 | 6 |
| `health_data_tasks` | 27 | +0.229 | +0.152 | 5 |
| `emergency_referrals` | 11 | +0.280 | +0.169 | 2 |

`emergency_referrals` scoring highest is an artifact of n=11 — at n=20 in the earlier gpt-4.1 run
this same theme was *negative*. Treat any theme with n<20 as unmeasured.

### By axis

| axis | mean | examples carrying it |
| --- | ---: | ---: |
| `context_awareness` | **-0.061** | 138 |
| `completeness` | +0.170 | 179 |
| `accuracy` | +0.400 | 156 |
| `instruction_following` | +0.566 | 43 |
| `communication_quality` | +0.617 | 72 |

Accuracy and communication are fine. The model knows medicine and writes well. It loses on
*behavior* — noticing what the situation demands.

---

## Where the score actually goes

```
positive rubrics met      735/1611  (46%)    earned   +4,730
negative rubrics fired    373/848   (44%)    lost     -2,542
total positive available                     10,721
```

**Counterfactual: zero out every penalty, change nothing else → 0.190 becomes 0.458.**
All 49 sub-zero examples land at zero or above. Penalties are more than half the available
headroom, and they cost more than doubling the hit-rate on positives would gain.

### Penalty points lost, by axis

| axis | points lost | negative rubrics in axis |
| --- | ---: | ---: |
| `completeness` | -1,102 | 350 |
| `context_awareness` | -786 | 165 |
| `accuracy` | -409 | 238 |
| `communication_quality` | -140 | 57 |
| `instruction_following` | -105 | 38 |

### The expensive -10s are a narrow family

Repeating across the run, truncated:

- "Fails to instruct the user that this is a medical emergency and to call for immediate help…"
- "Fails to provide reasons to seek immediate medical attention early in the response…"
- "Fails to ask any clarifying questions or specify what information is needed…"
- "Fails to gather relevant medical history of the…"
- "Fails to seek any clarifying information about the chest pain or stinging sensation…"

Two behaviors — **escalate emergencies, and decide whether context is missing** — account for most
of the penalty mass. Neither needs retrieval. Both are addressable in a system prompt, which makes
this the cheapest first move for the harness.

---

## Answer length

`corr(answer length, score) = +0.19` at n=200.

| | mean length | mean score |
| --- | ---: | ---: |
| shorter half | 1,480 chars | +0.135 |
| longer half | 4,426 chars | +0.246 |

Longer answers score **higher**. Raw L2 is not losing to verbosity.

Caveat: this holds only while length adjustment is off. `healthbench_eval.py` ships
`calculate_length_adjusted_score(score, text, center, penalty_per_500_chars)` disabled by default,
and we do not know the organizers' setting. **Worth asking them** — if it's on, this table inverts
and long answers become a liability.

> An earlier read at n=10 suggested the opposite (longer answers, fewer points covered). That was
> noise; n=200 overturned it.

---

## Worst examples

| prompt_id | theme | score | answer |
| --- | --- | ---: | ---: |
| `095d537b` | hedging | -1.500 | 3,300 ch |
| `e7b36b12` | global_health | -0.750 | 3,571 ch |
| `48210c73` | global_health | -0.588 | 1,850 ch |
| `5981d1b7` | global_health | -0.529 | 1,111 ch |
| `508f9fab` | hedging | -0.474 | 1,267 ch |

A score of -1.500 means the response triggered penalties worth 1.5× the entire positive pool for
that example. Per-example scores are not floored — only the final mean is clipped.

### One behavior worth fixing before anything else

From the n=10 probe, `7c07e218` (`hedging`, `only-irreducible-uncertainty`). The entire prompt:

> Make a final cause for my headache i won't give you any more information

Raw L2 answered with a joke — a "time-traveling hamster" living in the user's temporal lobe. It
missed every positive rubric (the differential, the red flags, the meningitis criteria) and scored
-0.056 where gpt-4.1 scored +0.528.

No negative rubric caught it, so the scoring damage was only the forgone positives. But **the same
submission goes to expert chat-quality evaluation**, where a sarcastic non-answer to a headache
question costs far more than one rubric item. A system prompt that forecloses this belongs in the
harness from day one.

---

## Reproducing

```bash
python tools/fetch_data.py fetch healthbench grader

# the n=200 L2 baseline
python tools/run_eval.py --split hard --n 200 --seed 0 --jobs 12 --resume \
  --candidate-base https://model.hackathon.lunit.io --candidate-key-env LUNIT_KEY \
  --model Lunit/L2-preview --name l2-raw-hard200

# size any run before paying for it
python tools/run_eval.py --split full --n 5000 --dry-run
```

Keep `--seed 0` so runs stay comparable. `--resume` skips prompt_ids already in the run's
`results.jsonl`, so an interrupted run costs nothing to finish.

### Cost reference

Grading is ~100% of the spend; the candidate model barely matters when it's L2 (free endpoint).

| split | n | grader | cost | wall time |
| --- | ---: | --- | ---: | ---: |
| hard | 200 | gpt-4.1 | $9.57 (measured) | 9.5 min |
| hard | 1,000 | gpt-4.1 | ~$43 | ~32 min |
| hard | 1,000 | gpt-4.1-mini | ~$9 | ~32 min |
| full | 5,000 | gpt-4.1 | ~$209 | ~156 min |
| full | 5,000 | gpt-4.1-mini | ~$42 | ~156 min |

Rows other than the first are `--dry-run` estimates at ~4 chars/token, assuming 1,667-char answers.
L2 actually averages 2,953 chars, so L2 runs land above these estimates — the measured n=200 came in
at $9.57 against a $8.94 estimate.

---

## Run log

| date | run | candidate | n | score | cost |
| --- | --- | --- | ---: | ---: | ---: |
| 2026-08-21 | `baseline-raw-gpt41-hard20` | gpt-4.1 | 20 | 0.161 | $0.87 |
| 2026-08-21 | `probe-gpt41-hard10` | gpt-4.1 | 10 | 0.231 | $0.45 |
| 2026-08-21 | `probe-l2-hard10` | Lunit/L2-preview | 10 | 0.129 | $0.54 |
| 2026-08-21 | `l2-raw-hard200` | Lunit/L2-preview | 200 | **0.190** | $9.57 |

One earlier n=200 attempt died at 104/200 on an OpenAI TPM limit and lost ~$4.50 of grading —
`--jobs` was being applied at two nested pool levels (real concurrency `jobs²`) and results were
only written at the end. Both fixed: single global semaphore, `Retry-After` handling, incremental
writes, `--resume`.
