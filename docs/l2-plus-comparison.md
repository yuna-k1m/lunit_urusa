# L2-plus comparison

Matched HealthBench Hard comparison using the same 50-example seed-0 sample,
GPT-4.1 grader, 12 jobs, and 4,096-token runner setting. The regular-L2 side is
the existing run; it was not regenerated.

| Internal backend | Score | Wall time | Mean candidate time | Median candidate time |
| --- | ---: | ---: | ---: | ---: |
| Regular L2 (existing) | 0.3209 | 191.8 s | 30.19 s/example | 26.40 s/example |
| L2-plus | 0.3953 | 182.1 s | 31.73 s/example | 28.85 s/example |
| Difference | +0.0744 | -9.7 s | +1.55 s/example | +2.45 s/example |

L2-plus improved 24 examples, tied 9, and regressed 17. Its absolute score gain
was 7.44 points (23.2% relative). A paired bootstrap 95% interval for the mean
difference was approximately -0.005 to +0.152, so this 50-example run is
promising but not conclusive.

The end-to-end wall time was 5.1% lower, but concurrent candidate generation
and rubric grading make that number noisy. Candidate timing recorded inside the
harness was 5.1% slower on the mean and 9.3% slower on the median. This is the
more useful latency comparison.

Retrieval was requested on 4 of 50 examples. All four ran local search before
MCP; three found local catalog matches (15 hits total), and three turns produced
grounded evidence. The four retrieval stages used 23.4 seconds in total.

Results:

- Regular: `results/siusiubeom_h4-hard50-seed0-gpt41-j12-m4096/`
- L2-plus: `results/siusiubeom_h4-l2plus-hard50-seed0-gpt41-j12-m4096/`

The comparison command for the new side was:

```bash
L2_BACKEND=l2_plus python tools/run_eval.py \
  --mode strategy --strategy siusiubeom_h4 \
  --split hard --subset all --n 50 --seed 0 \
  --grader gpt-4.1 --grader-api chat \
  --jobs 12 --max-tokens 4096 \
  --name siusiubeom_h4-l2plus-hard50-seed0-gpt41-j12-m4096 --resume
```
