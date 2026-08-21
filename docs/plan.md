# Plan

Working plan for the Lunit hackathon driver. Living document — update the status column and the
open questions as they resolve. Numbers cited here come from `docs/baseline-results.md`.

## Thesis

Three levers, in the order they are expected to pay off:

1. **Behavior, not knowledge.** Raw L2 already scores +0.40 on accuracy and +0.62 on communication
   quality. It loses on *behavior*: it almost never asks for missing context (cluster rubric
   `not-enough-context_context_seeking` met 1/26, `any-reducible-uncertainty_seeks_context` 0/9),
   and it triggers penalties worth more than the positive headroom (zeroing penalties alone:
   0.190 → 0.458). Behavior is fixable with prompting and orchestration; it needs no retrieval.
2. **Intermediate models** (allowed: "only the final answer must come from L2") for the judgment
   calls L2 is weak at — urgency triage, context-sufficiency, audience, language, multi-turn
   query rewriting — and as a critic. **Blocked on whether the eval box can reach them** (see
   open questions).
3. **MCP two-stage retrieval** — the moat for the Korean holdout and expert chat track (MFDS
   approvals, HIRA reimbursement, Korean guidelines, KCD), and the thing L2 was actually trained
   for. Expected to move the public-HealthBench number little; expected to matter a lot for
   Korean regulatory/drug questions and for how the expert reviewers perceive grounding.

## What we learn from HealthBench, and what we refuse to learn

The holdout is a different set, so only *policy* transfers. The 37 unique `level:cluster` rubrics
(dumped locally to `docs/healthbench-themes/cluster-policies.md`, gitignored) are literal prose
statements of the graded policy per theme × physician category. They are the curriculum for the
system prompt and for any router/critic. `ideal_completions` teach *shape* (structure, how to
hedge, how to escalate), not content.

Guardrails against overfitting:

- Never retrieve HealthBench examples at runtime. Never quote rubric text in prompts (canary data).
- **Frozen splits** in `tools/run_eval.py`: `--subset tune` is exactly the 200 examples the
  baseline was measured on (`Random(0).sample(hard, 200)`); `--subset val` is a disjoint 200.
  Iterate on tune; grade val only to confirm a change that already won on tune.
- The dashboard validation set is the arbiter. The public number is a proxy.

## Experiment ladder

| # | Step | Status | Result |
| --- | --- | --- | --- |
| 0 | Raw L2 baseline, tune n=200 | done | **0.190** [0.145, 0.238] |
|   | *Score ladder (tune n=200):* raw 0.190 → v2 prompt 0.213 → h1 0.239 → h3 0.271 → **h4 0.357** | | |
| 1 | Probe container → what can the eval box reach; how does the evaluator send turns | built, awaiting trial | — |
| 2 | `run_eval.py`: `--system`, `--subset tune/val`, `--slice nonlatin/multiturn` | done | — |
| 3 | L2-only system prompt distilled from cluster policies | done | v2: **0.213**, paired Δ +0.022 [−0.018, +0.064] — not significant |
| 3b | L2-only planner → brief → assemble (`app/engine.py`) | **h3 = current engine** | h1 0.239 (+0.049); h2 aborted (validation exposed 28% planner role-breaks); **h3 0.271, Δ vs base +0.080 [+0.037, +0.122]**, 99W/63L, 0 fallbacks |
| 4 | Planner = `gpt-5.6-sol`, writer = L2 (`--planner-model` / `PLANNER_MODEL`) | **h4 = submitted (SHA 50acb80)** · **val confirmed: raw 0.130 → 0.269, Δ +0.139 [+0.092, +0.193]** | **h4 0.357**, Δ vs base **+0.166 [+0.121, +0.211]** 135W/46L; Δ vs h3 +0.086 [+0.043, +0.129]. Runtime use still gated on eval-box reachability (#1) |
| 4b | Critic pass (sol reviews, L2 revises once; `CRITIC=1`) | measured, **off by default** | h5 tune 0.377 vs h4 0.359, Δ +0.018 [−0.023, +0.060] — n.s.; complex_responses +0.013; +30 s/turn. Not worth the timeout risk |
| 5 | L2 two-stage harness with MCP (retrieval + generation prompts, `finalize_retrieval`, cite_uid resolution, tool-call cap) | not started | — |
| 6 | Multi-turn: fold history into a self-contained query before retrieval | not started | — |
| 7 | Korean chat loop against the patient simulator, judged by a rubric written from the cluster policies | not started | — |
| 8 | Per-theme error analysis loop on tune → prompt/policy edits → confirm on val | ongoing | — |

Every step from 3 on is an ablation against the previous best, same config (`--subset tune --n 200
--seed 0`, grader gpt-4.1 at temperature 0.5). A run costs ~$10 and ~10 min.

### Lessons so far

- **L2 ignores behavioral prose in a system prompt.** v2 asked for 1–3 clarifying questions when
  context is missing; the context-seeking cluster rubric moved 1/26 → 5/26. Without
  `response_format=json_object` L2 even ignores a planner system prompt and just answers the
  user. Behavior has to be imposed structurally (planner JSON → concrete brief → driver-side
  assembly). h1 took that rubric to 11/26.
- **Planner failure modes seen at n=200 (h1):** 12% fallbacks (truncation + role-breaks such as
  `{"error": "I can't …"}` that still parse as JSON); over-triage of clinicians' reference
  questions as "emergent" (brief then says "keep it short" and completeness dies); missed
  multi-turn emergencies (choking, device failure); fabricated vitals/exam findings in drafted
  notes under the completeness push. h2 addresses each: schema validation + mini-planner retry,
  neutral brief on fallback, HP-vs-layperson emergency briefs, a no-invention rule.
- **The planner must see the conversation as a transcript inside one user message.** Given real
  chat turns, L2 joined the conversation and answered the user in ~25% of plans
  (`{"advice": …}`, `{"answer": …}`), which parse as JSON. As a `<conversation>` transcript:
  0/200 failures (h3). This is the single most important engineering fact about L2 so far.
- h3 residuals: `enough-info-to-complete-task` −0.155 (planner asks on completable data tasks);
  `hedging_any-reducible-uncertainty_seeks_context` 4/9; L2 almost never writes the question block
  itself (assembler appended it 101/105 times) — structural placement is load-bearing.

- Any wording that sounds like "be careful before giving X" makes L2 **refuse**; a refusal loses
  every positive rubric. The prompt must say explicitly that standard dosing/regimen information is
  always appropriate as general information.
- "Skip basics for professionals" cost more than it gained — completeness rubrics reward a brief
  definition and a differential even for an HP audience. Never instruct the model to omit content;
  instruct it to be brief.
- Grader noise at temperature 0.5 flips individual rubrics between runs of near-identical answers.
  n=10 paired comparisons mean nothing; use n=200.

## Submission facts (from the teammate's successful run, see `submission-success-runbook.md`)

- CoEval injects **no environment variables**; credentials ship as files in the image
  (`submission_api_key`, `submission_openai_key` base64-encoded). The repo is private.
- Bare L2 proxy scored **49.4** on the leaderboard → the organizers' holdout is far easier than
  `hard` (raw L2 = 0.19/0.13 there). Expect absolute numbers to differ; relative gains to transfer.
- h4 degrades gracefully: planner unreachable → L2 plans (h3 level), never a failed turn.

## Open questions (need the organizers or a trial)

1. **Can the eval container reach `api.openai.com`?** The h4 trial answers this indirectly (score
   near h4 vs near h3 level). The probe build (`DRIVER_ENGINE=probe`) answers it directly.
2. **Holdout composition** — a sample of the public 5,000, or freshly written Korean items? This
   decides how much the public tune number means and how much Korean-specific work matters.
3. **Length-adjusted scoring** on or off? Raw L2's longer answers score higher (corr +0.19) only
   while it is off.
4. **Evaluator timeouts** per turn. MCP retrieval is multi-call at up to 60 s per tool.
5. **Is the multi-turn driven with the full history each turn** (stateless, like the simulator) or
   with our previous answers substituted by the benchmark's? The probe's request dump answers this.

## Harness shape (target)

```
incoming messages (full history, last = user)
  │
  ├─ [planner]   language · audience · urgency bucket · context-sufficiency · self-contained query
  │              (frontier model if reachable, else L2 with a compact prompt)
  │
  ├─ [L2 generation stage]  system prompt from cluster policies + planner brief
  │       └─ tool: retrieve_relevant_content(query)
  │              └─ [L2 retrieval stage]  MCP tools + finalize_retrieval, capped calls,
  │                                      cite_uids resolved back to content
  │
  └─ [critic]    language match · emergency placement · no unwarranted questions · no refusal
                 → at most one revision pass by L2
```

L2 writes all medical content. Other models plan and check; they never author the final text.

## MCP facts (verified 2026-08-21 from the dev machine)

- Streamable HTTP, **stateless** (no `Mcp-Session-Id`), 21 tools. Schemas dumped locally to
  `docs/healthbench-themes/mcp_tools.json` (gitignored; regenerate with a `tools/list` call).
- Citable results are JSON with an `items` list; each item has `cite_uid`, `source_type`,
  `source_id`, `url`, `content`, plus type-specific fields (`drug_name`/`section` for DailyMed,
  `doc_title`/`node_id`/page ranges for the index). The retrieval stage must cache every item by
  `cite_uid` as it goes so `finalize_retrieval` can be resolved back to content.
- `index_list_documents` relevance ranking is weak (a CKD blood-pressure query returned NCCN
  Kidney *Cancer* first); `index_get_relevant_nodes` across the whole corpus was on target.

## Submission hygiene

- Branch `lunit/hackathon-submission`; Dockerfile at root; `0.0.0.0:8000`; build < 5 min
  (probe image builds in ~25 s).
- `DRIVER_ENGINE` env var selects `probe` vs `harness`; default must be `harness` before the
  final submission.
- The **last** dashboard submission is the final one. Never leave a probe build as the last
  submission.
