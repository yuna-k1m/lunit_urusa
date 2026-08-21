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
|   | *Score ladder (tune n=200):* raw 0.190 → v2 prompt 0.213 → h1 0.239 → h3 0.271 → h4 0.357 → h7 0.362 → **h9 0.393** | | |
| 1 | Probe container → what can the eval box reach; how does the evaluator send turns | built, awaiting trial | — |
| 2 | `run_eval.py`: `--system`, `--subset tune/val`, `--slice nonlatin/multiturn` | done | — |
| 3 | L2-only system prompt distilled from cluster policies | done | v2: **0.213**, paired Δ +0.022 [−0.018, +0.064] — not significant |
| 3b | L2-only planner → brief → assemble (`app/engine.py`) | **h3 = current engine** | h1 0.239 (+0.049); h2 aborted (validation exposed 28% planner role-breaks); **h3 0.271, Δ vs base +0.080 [+0.037, +0.122]**, 99W/63L, 0 fallbacks |
| 4 | Planner = `gpt-5.6-sol`, writer = L2 (`--planner-model` / `PLANNER_MODEL`) | **h4 = submitted (SHA 50acb80)** · **val confirmed: raw 0.130 → 0.269, Δ +0.139 [+0.092, +0.193]** | **h4 0.357**, Δ vs base **+0.166 [+0.121, +0.211]** 135W/46L; Δ vs h3 +0.086 [+0.043, +0.129]. Runtime use still gated on eval-box reachability (#1) |
| 4b | Critic pass (sol reviews, L2 revises once; `CRITIC=1`) | measured, **off by default** | h5 tune 0.377 vs h4 0.359, Δ +0.018 [−0.023, +0.060] — n.s.; complex_responses +0.013; +30 s/turn. Not worth the timeout risk |
| 4c | **max_tokens 2048 → 6000 for L2 generation** (hidden reasoning channel was truncating answers) + seeded retrieval + dry-search scrub | **h9 = submitted** | **h9 tune 0.393**, Δ vs h4 +0.036 [−0.004, +0.076]; complex_responses +0.116, emergency +0.121 |
| 5 | MCP retrieval stage (`app/retrieval.py`), planner-gated | **h7 = submitted** | h6 (broad gating, ~30% of turns) hurt: grounded English examples +0.07 → −0.36 from off-topic Korean evidence. Now gated to Korean regulatory asks / explicit guideline-or-label asks; Korean HIRA query verified grounded on 고시 제2025-169호 with inline citation. h7 tune 0.362 vs h4 0.357, Δ +0.005 [−0.035, +0.047]: no regression, fires on ~5% of HealthBench turns |
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

## 2026-08-22 status: no OpenAI egress from the evaluation box (organizers confirmed)

Every dashboard score (49.4 raw → 49.9–51.3 for all our builds) is explained by this: the sol planner
never ran there. Measured on the 301 leaderboard items (gpt-4.1 judge replica):

| config | score | note |
| --- | ---: | --- |
| raw L2 | 0.501 | matches dashboard 49.4 |
| L2-only harness, lean no-think planner (h19 = `a264e84`) | 0.512 | +0.011 n.s.; 24 s/turn, 0 fallbacks |
| L2-only, thinking planner (h20, n=97) | 0.480 | null, 60 s/turn |
| **no planner at all** (h21 = `2fce8c9`): generation prompt + neutral brief + assembler | **0.529** | +0.025 [+0.001, +0.052] vs raw; best L2-only option; 19–28 s/turn |
| sol-planned harness (h10) | 0.590 | +0.089 [+0.058, +0.120] — unreachable from the eval box |

Also null on this set with sol available: specificity checklist (h12), best-of-2 (h14), web-search
specifics (h17), critic/revision (h16 +0.013 n.s.). Real but small: emergency brevity (−10 verbosity
penalties), long-thread completeness brief (+0.06 on the 28 ≥6-turn items).

Operational facts: CoEval times out a request at 180 s and scores it 0 → per-turn deadline 150 s,
single-attempt calls, no retry on timeout, speculative no-thinking fallback. Telemetry via the OpenAI
Files API (filename-encoded) exists for environments with egress. Startup egress check switches to
L2-only planning automatically.

## Current submission (2026-08-22, `0384ae1`)

No planner + L2-only grounding gate (Korean regulatory / literature / label cues → seeded MCP
retrieval, L2 selects evidence) + GEN_TEMPERATURE=0.0. Korean dev set: raw 0.156 → 0.227 (sol judge).
Benchmark-neutral. Teammate's "quality gate" commit `ef790a8` was reverted: −0.025 at 77/301 vs the
no-planner baseline (h25). Rule for this branch: nothing merges into the submission head without a
paired conquer_val measurement.

## What the leaderboard actually is (from the open-source evaluator, lunit-io/CoEval)

- **`conquer_val` = 301 theme-stratified items from public HealthBench Main**; ids published in
  CoEval (`tools/conquer_val_ids.json` here; `--split full --subset conquer_val`). The final test
  is a disjoint, matched slice of the same pool. Judge: `glm-5.2-fp8` (test judge undisclosed).
  My earlier "holdout = Korean MCP corpus" hypothesis was wrong; the simulator serves the chat track.
- Runner: 16-way concurrency, 180 s request timeout ×2 attempts, **failures/empty answers = 0**,
  `max_tokens: 6144`, system prompt "You are Chain-of-Evidence". Wall-clock cap ~30 min; judging
  is shared across ~8 concurrent evaluations and dominates overhead (a 0-s-answer probe took 16 min,
  ~4 of which were the probe's own DNS stall).
- Local replica (gpt-4.1 judge): raw 0.501 → h10 **0.590** (+0.089 [+0.058, +0.120]); CoEval's own
  runner against our container: 301 items in **11 min 13 s** end to end, 0 inference failures.
- h10 weakness on this set: 70% of positive points captured; penalties cost 0.112; remaining loss is
  mostly **specificity near-misses** (timeframes, named authorities, canonical terms, explicit
  caveats) rather than missing topics or behaviour. Asking is net-positive (ask-related misses 911
  pts vs ask-related penalties 361) — not a lever. `enable_thinking=false` is −32% latency but
  −0.06 score at n=59: rejected.

## L2 facts that cost us points until found

- **L2 has a hidden `reasoning` channel that counts against `max_tokens`** (5–6k chars on hard
  questions). At the simple-evals budget of 2048 the visible answer is cut off
  (`finish_reason=length`); ~5–13% of answers in earlier runs ended mid-sentence and scored
  0.13 vs 0.37. Generation now uses 6000 and ignores the evaluator's `max_tokens`.
- The organizers' holdout is almost certainly generated from the MCP corpus the same way the
  patient simulator is: simulator questions paraphrase specific PubMed abstracts, HIRA notices,
  guideline recommendations and statutes (one "YouTube said…" question resolved to PMID
  34970451 in one call). Retrieval coverage is the score on that set. Korean dev set:
  `tools/ko_dev.py` (raw 0.140 → h7 0.186–0.213 with a sol judge; judge noise ≈ ±0.03 at n=40).
- law.go.kr article endpoints were down upstream on 2026-08-21 (`no 법령 envelope` for every
  MST); the statute seed chain is in place for when they return.

## Retrieval stage facts (verified 2026-08-21)

- L2 does OpenAI tool calling and picks sensible tools; `tool_choice` forced by name works in
  short contexts, `"required"` does not. In long tool contexts L2 rarely calls
  `finalize_retrieval`, so a sol **selector** picks relevant `cite_uid`s from the cache instead.
- L2 loops on `index_list_documents` / `index_get_document_structure` (no `cite_uid`); both are
  hidden. After `index_get_relevant_nodes` the harness auto-reads the top hit's pages (`range`
  field) so citable content exists after one model step. Page items keep text under `pages[]`.
- Budget 4 calls / 35 s; a grounded turn costs ~60–75 s end to end. Off-topic evidence hurts
  more than no evidence: gate narrowly and filter Korean regulatory items out of non-Korean
  answers.

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
