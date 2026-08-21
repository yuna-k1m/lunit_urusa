# AGENTS.md

Working notes for anyone (human or agent) picking up this repo.

Project: a **conversation driver for the Lunit hackathon** — an OpenAI-compatible service that
answers Korean/English medical questions using Lunit's L2 model, graded on a HealthBench-style
holdout set.

---

## Quickstart

```bash
# 1. large data is NOT in git — pull it down (~105 MB for the core sets)
python tools/fetch_data.py fetch healthbench

# 2. check it landed
python tools/fetch_data.py status

# 3. poke at the benchmark
PYTHONIOENCODING=utf-8 python tools/hb.py stats
PYTHONIOENCODING=utf-8 python tools/hb.py show hard 0
```

On Windows use `./.venv/Scripts/python.exe` and keep the `PYTHONIOENCODING=utf-8` prefix for
`hb.py` — the console defaults to cp949 and dies on non-ASCII benchmark text.

---

## Data policy: no git-lfs

**Nothing large is committed.** No LFS, no binaries in history. Everything big is declared in
`tools/assets.json` and fetched on demand by `tools/fetch_data.py`, verified by size + SHA-256.
`data/` is gitignored except for `.gitkeep`.

Why: LFS quota/bandwidth becomes a shared-account problem, the evaluation container must build in
under 5 minutes, and a 235 MB benchmark is reproducible from a URL anyway.

### `tools/fetch_data.py` — the general asset fetcher

Stdlib only, no pip install, works in a bare container. Not HealthBench-specific — use it for any
large asset this project ends up needing.

```bash
python tools/fetch_data.py                  # fetch everything missing
python tools/fetch_data.py list             # show the manifest, grouped
python tools/fetch_data.py status           # present / MISSING per asset
python tools/fetch_data.py fetch healthbench          # a group
python tools/fetch_data.py fetch healthbench-hard     # a single asset
python tools/fetch_data.py fetch --force --jobs 6     # re-download, 6 at a time
python tools/fetch_data.py verify           # re-hash on-disk files
python tools/fetch_data.py clean healthbench-extra --yes    # free 130 MB
```

Behaviour:

- Streams to `<dest>.part` and **resumes** via HTTP `Range` if interrupted — a dropped 136 MB
  transfer doesn't restart.
- A pinned asset (has `sha256`) is hashed before being moved into place. Mismatch → the bad file is
  parked as `.bad` and the fetch reports failure.
- `fetch` on an already-present pinned file re-hashes it and silently re-downloads if it's corrupt.
  `--no-verify` skips that if you're in a hurry.
- Unpinned assets (`"sha256": null`) track a moving source (e.g. a git branch) — never hash-checked,
  only re-fetched with `--force`.
- `verify` exits non-zero on missing/corrupt, so it works as a CI or pre-run gate.

### Adding a new large asset

```bash
python tools/fetch_data.py add <URL> data/whatever/file.bin --group mygroup --description "..."
```

Downloads it, computes size + SHA-256, and appends the entry to `tools/assets.json`. Commit the
manifest change; the file itself stays untracked.

For an asset behind a token, add `"auth_env": "MY_TOKEN"` to its manifest entry (or pass
`--auth-env`) and the fetcher sends `Authorization: Bearer $MY_TOKEN`. **Never put a key in the
manifest.**

Manifest entry shape:

```jsonc
{
  "name": "healthbench-hard",
  "group": "healthbench",
  "url": "https://.../hard_2025-05-08-21-00-10.jsonl",
  "dest": "data/healthbench/healthbench_hard.jsonl",   // relative to repo root
  "size": 12581564,
  "sha256": "b0320430...",                              // null = unpinned
  "description": "HealthBench Hard - 1,000 examples"
}
```

### What's in the manifest today

| Group | Size | Contents |
| --- | --- | --- |
| `healthbench` | 105 MB | full (5,000 ex.), hard (1,000), consensus (3,671) |
| `healthbench-extra` | 130 MB | meta-eval — physician-vs-grader labels, only needed to build a grader |
| `grader` | tiny | `healthbench_eval.py` and friends from `openai/simple-evals` (unpinned, tracks `main`) |

---

## Repo layout

```
AGENTS.md                    <- you are here
tools/
  fetch_data.py              general manifest-driven asset downloader
  assets.json                the asset manifest (committed; the data is not)
  hb.py                      HealthBench explorer CLI
  make_theme_docs.py         generates docs/healthbench-themes/ from the data
docs/
  lunit-hackathon-brief.md   platform reference: MCP tools, L2 two-stage model, submission rules
  healthbench-notes.md       benchmark format, exact scoring formula, composition stats
  healthbench-themes/        gitignored; one readable .md per problem type (generated)
data/                        gitignored; populated by fetch_data.py
reference/simple-evals/      official grader source (fetched, gitignored)
```

---

## HealthBench in one page

Full detail in `docs/healthbench-notes.md`. The parts that drive design decisions:

**Format.** Each record is `{prompt: [messages], rubrics: [{criterion, points, tags}],
example_tags, ideal_completions_data?}`. No system message. The last message is always `user`.
`hard` and `consensus` are strict subsets of `full`.

**Scoring.**

```python
total_possible = sum(r.points for r in rubrics if r.points > 0)   # positives only
achieved       = sum(r.points for r, g in zip(rubrics, grades) if g["criteria_met"])
score          = achieved / total_possible                        # can go negative
```

An LLM grader judges each rubric item as a standalone boolean. Only the *final mean across
examples* is clipped to [0,1] — individual examples are not floored.

**The five things that follow from that:**

1. **31% of rubrics carry negative points** (17,575 of 57,237) and aren't floored. Avoiding penalized
   behavior is worth as much as adding correct content; one bad answer cancels several good ones.
2. **Completeness is the largest axis** (22,285 rubrics) — enumerate the specific items a physician
   would list. But a length-adjusted scoring variant exists in the grader, so: dense and structured,
   not padded.
3. **Context-seeking is graded both ways.** `not-enough-context` examples reward a clarifying
   question; `enough-context` / `context-does-not-matter` examples penalize one. Must be conditional
   on the prompt, never a fixed policy.
4. **~45% of examples are multi-turn** (up to 19 turns). L2 is single-turn tuned, so the driver has
   to fold history into a self-contained query before the retrieval stage.
5. **Answer in the user's language.** `global_health` is the largest theme (22%) and its rubrics
   grade language matching and local-resource realism.

**Themes** (full): global_health 1,097 · hedging 1,071 · communication 919 · context_seeking 594 ·
emergency_referrals 482 · health_data_tasks 477 · complex_responses 360.

**Two rubric levels.** `full` carries 49,184 `level:example` rubrics (short, specific, -10..+10)
*and* 8,053 `level:cluster` rubrics (long prose policy statements, all +5, shared with the consensus
set). The grader doesn't filter by level, so both count. The cluster ones are worth reading — they
state the graded policy explicitly.

### Per-theme reading material

```bash
python tools/make_theme_docs.py       # -> docs/healthbench-themes/*.md
```

One Markdown file per theme: what it tests, split distribution, axis breakdown, the most frequent
reward/penalty phrasings mined from the rubrics, the +10 and -10 criteria verbatim, and four worked
examples (conversation + every rubric + the physician's ideal answer). Deterministic output, so
regenerating produces no spurious diffs.

Output is **gitignored** — it reproduces benchmark text verbatim and that data carries a canary
string. Everyone regenerates locally from the committed script.

Develop against `hard` (1,000 examples) — it's the discriminating subset and cheap to re-grade.
The organizers score the final submission on **their own holdout**, not these files.

---

## Lunit platform in one page

Full detail in `docs/lunit-hackathon-brief.md`.

One `lunit_` API key for all three endpoints: model (`model.hackathon.lunit.io`), patient simulator
(`patient.hackathon.lunit.io`), MCP (`mcp.hackathon.lunit.io/mcp`). All are **Lunit-network only**.

**L2 is a two-stage model, not a chat model:**

- *Retrieval stage* — gets the MCP tools plus a locally-defined `finalize_retrieval`. It gathers
  evidence and ends by reporting `cite_uid`s, **not** an answer.
- *Generation stage* — gets exactly one tool, `retrieve_relevant_content`, which runs the retrieval
  stage and returns the evidence. Separate system prompt per stage; never merge them.

Rules: retrieval queries must be self-contained; cap tool calls; **the final output must come from
L2** even if intermediate orchestration uses other models.

**Submission contract:** `Dockerfile` at repo root, image builds in <5 min, container auto-starts and
serves on `0.0.0.0:8000`, OpenAI-compatible with at least `GET /v1/models` and
`POST /v1/chat/completions`. Branch `lunit/hackathon-submission`, submit the full 40-char SHA.

---

## Conventions

- **Never commit anything from `data/`** or `reference/`. If you need a new large file, register it
  with `fetch_data.py add`.
- `tools/*.py` are stdlib-only on purpose — they must run before any dependency install.
- The evaluation environment has **no external network**. Anything `fetch_data.py` pulls is a
  development-time convenience; the submitted container must not depend on it at runtime.
- HealthBench rows carry a `canary` string. Leave it in place and don't publish model outputs
  alongside it.
