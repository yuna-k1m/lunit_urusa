# Multi-patient count comparison on HealthBench Hard 30

Date: 2026-08-21

## Setup

- Strategy: `multi_patient_sol`
- Patient profile counts: 3 and 5
- Dataset: pinned HealthBench Hard split
- Sample: 30 examples selected with `--subset all --n 30 --seed 0`
- Grader: `gpt-4.1` through Chat Completions
- Concurrency: `--jobs 4`
- System prompt: runner baseline
- All result files contain the same 30 unique prompt IDs.
- The requested 7-patient case was intentionally skipped.

The benchmark result files are gitignored and are not included in this report because they contain
benchmark text and canary-bearing data.

## Results

| Patient profiles | Mean | Median | Range | Negative examples | Mean answer length |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | **0.426** | **0.451** | -0.074 to 1.000 | **2/30** | **2,187 chars** |
| 5 | 0.351 | 0.350 | -0.212 to 1.000 | 3/30 | 2,256 chars |

Increasing the profile count from 3 to 5 reduced the mean by **0.075** (17.6% relative to the
3-profile score). In the paired example-level comparison, 5 profiles won 7 examples, tied 10, and
lost 13. The 3-profile configuration is the better result on this sample.

## Theme scores

| Theme | 3 profiles | 5 profiles | Delta (5 - 3) |
| --- | ---: | ---: | ---: |
| communication | **0.277** | 0.209 | -0.068 |
| complex_responses | **0.162** | 0.132 | -0.029 |
| context_seeking | **0.421** | 0.270 | -0.151 |
| emergency_referrals | 0.356 | **0.445** | +0.089 |
| global_health | **0.333** | 0.198 | -0.135 |
| health_data_tasks | **0.532** | 0.419 | -0.113 |
| hedging | **0.528** | 0.512 | -0.016 |

The largest regressions were context seeking, global health, and health-data tasks. Emergency
referrals was the only theme with a material improvement. Theme sample sizes are only 1 to 8, so
these breakdowns are diagnostic rather than stable estimates.

## Pipeline diagnostics

All 30 three-profile examples completed all 3 requested profile calls. For the five-profile run,
27 examples completed all 5 calls, 2 completed 4, and 1 completed 3. All examples still used the Sol
finalizer, and neither run fell back to direct L2. Partial profile-call failures may have contributed
to the five-profile result, but they affected only 3 of 30 examples and do not explain the broader
13-loss paired pattern by themselves.

## Interpretation and limitations

More candidate perspectives did not improve this configuration. Five profiles produced slightly
longer answers while scoring worse on completeness and context awareness; the extra candidates may
be adding aggregation noise or diluting high-value details. The practical default remains 3
profiles unless a larger evaluation contradicts this result.

- Thirty examples are too few for a firm leaderboard prediction.
- GPT-4.1 is not the official holdout grader and differs from the team's canonical GPT-5.6 Sol
  development grader.
- Candidate generation is stochastic, and the 3-profile result was reused from the existing matched
  run rather than regenerated alongside the 5-profile run.
- Candidate token usage is not observable for these composed strategies. The recorded grader cost
  for the new 5-profile run was approximately $1.21.
