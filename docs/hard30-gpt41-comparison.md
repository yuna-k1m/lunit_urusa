# HealthBench Hard 30 comparison (GPT-4.1 grader)

Date: 2026-08-21

## Setup

- Runner: `tools/run_eval.py`, strategy mode
- Dataset: pinned HealthBench Hard split
- Sample: 30 examples selected with `--subset all --n 30 --seed 0`
- Grader: `gpt-4.1` through Chat Completions
- System prompt: runner baseline
- All three result files contain 30 unique prompt IDs from the same sample.

The benchmark result files are gitignored and are not included in this report because they contain
benchmark text and canary-bearing data.

## Results

| Strategy | Mean | Median | Range | Negative examples | Mean answer length |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_l2` | 0.182 | 0.276 | -0.750 to 0.722 | 6/30 | 4,155 chars |
| `siusiubeom_h4` | 0.370 | 0.432 | -0.583 to 1.000 | 3/30 | 3,277 chars |
| `multi_patient_sol` | **0.426** | **0.451** | **-0.074 to 1.000** | **2/30** | **2,187 chars** |

Relative to `baseline_l2`, `siusiubeom_h4` improved the mean by 0.188 and
`multi_patient_sol` improved it by 0.244. The paired example-level comparisons were:

| Comparison | Mean delta | Wins | Ties | Losses |
| --- | ---: | ---: | ---: | ---: |
| `siusiubeom_h4` vs. `baseline_l2` | +0.188 | 19 | 3 | 8 |
| `multi_patient_sol` vs. `baseline_l2` | +0.244 | 19 | 3 | 8 |
| `multi_patient_sol` vs. `siusiubeom_h4` | +0.056 | 13 | 4 | 13 |

## Theme scores

| Theme | `baseline_l2` | `siusiubeom_h4` | `multi_patient_sol` |
| --- | ---: | ---: | ---: |
| global_health | 0.000 | 0.000 | **0.333** |
| hedging | 0.212 | 0.464 | **0.528** |
| communication | 0.107 | 0.114 | **0.277** |
| context_seeking | 0.065 | **0.467** | 0.421 |
| emergency_referrals | 0.258 | 0.265 | **0.356** |
| health_data_tasks | 0.373 | 0.512 | **0.532** |
| complex_responses | 0.485 | **0.603** | 0.162 |

Theme sample sizes are small: 1 to 8 examples per theme. These breakdowns are diagnostic signals,
not stable estimates.

## Interpretation

Both composed strategies substantially outperformed direct L2 on this sample while producing
shorter answers and fewer negative-scoring examples. `multi_patient_sol` had the highest aggregate
score and reduced the worst-case result from -0.750 to -0.074. Its largest visible advantages were
in global health, hedging, communication, and emergency referral behavior.

The difference between the two composed strategies is less conclusive. `multi_patient_sol` led by
0.056 overall, but the paired comparison was evenly split at 13 wins and 13 losses, with 4 ties.
`siusiubeom_h4` remained stronger on context seeking and the single complex-response example.

`multi_patient_sol` uses Sol as its finalizer. A useful next comparison is the canonical fixed-100
run using the team's GPT-5.6 Sol development grader.

## Limitations

- Thirty examples are too few for a leaderboard prediction or a firm ranking between close models.
- GPT-4.1 is not the official holdout grader and differs from the team's canonical development grader.
- Theme slices are especially small.
- The run was resumed across multiple invocations, so elapsed time, grader-call totals, and estimated
  costs in the generated summaries cover only the final invocation for each strategy and are not
  comparable end-to-end measurements.
- Candidate token usage is not fully observable for composed strategies.
