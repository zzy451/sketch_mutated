# sketch_mutated repository instructions

This repo is a split-file GP sketch search system with mechanism-first proposal goals.

## What Codex should read first
Start with:
- `Sketch_Global_Proposal_SplitFiles_v3.md`
- `common.py`
- `cli.py`
- `evaluator.py`
- `helpers.py`
- `llm_engine.py`
- `evolution.py`
- `init_dex_language.py`
- `update_language.py`
- `query_language.py`

## Primary repo goals
Keep the project aligned with these priorities:
1. preserve split-file ownership boundaries
2. treat primitive language files as source-of-truth
3. stabilize Priority 0 wiring before broad mechanism changes
4. prefer small, reversible patches over broad refactors
5. verify after Python edits before claiming success

## Authoritative files
- Proposal: `Sketch_Global_Proposal_SplitFiles_v3.md`
- Split-file code:
  - `common.py`
  - `cli.py`
  - `evaluator.py`
  - `helpers.py`
  - `llm_engine.py`
  - `evolution.py`
  - `mutate_cmsketch_refactored.py`
- Primitive source-of-truth:
  - `init_dex_language.py`
  - `update_language.py`
  - `query_language.py`

## Ownership rules
Respect file boundaries strictly.

- `common.py`
  - primitive registry glue
  - shared constants/config
  - source-of-truth wiring
- `cli.py`
  - arg parsing
  - default primitive reference path injection
  - run-mode/config entry
- `evaluator.py`
  - AST legality / simplify / runtime semantics
  - search-time runtime vs exported runtime consistency
- `helpers.py`
  - numeric-risk policy
  - seed filtering
  - novelty / mechanism metadata helpers
- `llm_engine.py`
  - proposal generation
  - schema materialization
  - primitive validation
- `evolution.py`
  - GP loop
  - island dynamics
  - telemetry roll-up
  - migration / promote / feedback logic
- primitive language files
  - primitive names
  - signatures
  - phase ownership
  - local source-of-truth semantics

Do not smear runtime, registry, novelty policy, and GP loop logic across files without naming the owner first.

## Default workflow
For broad or ambiguous work:
1. use `project-orient`
2. identify owner file(s)
3. choose the matching repo skill
4. make the smallest reversible patch
5. run verification
6. report what changed, what was verified, and what remains unverified

## Repo-local skills
Skills live in `.agents/skills/`.

Use this routing table:
- repo map / ownership / where to patch first
  - `project-orient`
- primitive registry / default primitive paths / import compatibility / Priority 0 wiring
  - `priority0-wiring`
- proposal changes that must stay synced with implementation
  - `proposal-sync`
- primitive signature drift / runtime drift / validator drift
  - `primitive-runtime-alignment`
- small or medium code patch + immediate verification
  - `patch-verify`
- running or comparing smoke/formal experiments
  - `experiment-runner`
- reading logs / telemetry and picking the next smallest patch
  - `telemetry-triage`
- writing a resume note before switching session/agent/window
  - `session-handoff`

## Additional GitHub-installed skills
- `skill-creator`
- `docs`
- `code-review-excellence`

## When to use extra skills
- skill design, skill refactor, or improving existing repo-local skills
  - `skill-creator`
- README, runbook, troubleshooting notes, experiment notes, or documentation cleanup
  - `docs`
- structured review before or after a larger multi-file patch
  - `code-review-excellence`

## Verification policy
After Python edits, prefer this order:

### Fast verification
- `bash .agents/skills/patch-verify/scripts/quick-verify.sh`

### Primitive/runtime alignment verification
- `bash .agents/skills/primitive-runtime-alignment/scripts/verify-alignment.sh`

### Minimum compile check
- `python3 -m py_compile common.py cli.py evaluator.py helpers.py llm_engine.py evolution.py mutate_cmsketch_refactored.py init_dex_language.py update_language.py query_language.py`

If the current Python environment does not have required dependencies such as `deap`, compile checks may still pass while import smoke is skipped. In that case, do not claim full runtime verification.

## Experiment discipline
Do not compare runs as equivalent unless key settings match.
Always record:
- seed
- proxy mode
- dataset mode / dataset seeds
- fixed stream usage
- pop/gen
- whether LLM path was enabled
- whether holdout or stage2 real was enabled

For experiment summaries, prefer:
- `experiment-runner`

## Telemetry discipline
When logs exist, prioritize:
- `risk_reject_by_phase`
- `risk_warning_but_allowed_by_phase`
- `risk_reason_topk`
- `DIAG_CHUNK` values
- `novelty_mechanism_cluster_count`
- `novelty_family_count`
- `dominant_family_cooldown_hits`
- `mechanism_override_passes`
- `novelty_incubated`
- `novelty_promoted`
- `repair_injected` vs `novelty_injected`

Do not recommend a broad refactor from telemetry alone.
Name:
- the likely bottleneck
- the owner file
- the smallest next patch
- the expected telemetry change

## Priority reminders
Unless explicitly redirected, prefer:
1. Priority 0 wiring / source-of-truth alignment
2. seed hygiene
3. novelty from family survival to mechanism-cluster branching
4. repair de-dominance
5. mechanism-best feedback into main search
6. only then broader primitive/runtime expansion

## Codex notes
- `AGENTS.md` is the primary project instruction source for Codex.
- Repo-local skills live under `.agents/skills/`.
- Project config lives in `.codex/config.toml`.

## Definition of done
A task is not done unless the response includes:
- files changed
- what changed
- what was verified
- what remains unverified
- next recommended step if the work is partial

## Real-world exception policy
- Do not add fallback code, defensive branching, or guard rails for impossible or effectively unreachable situations.
- Do not write catch-all compatibility logic unless there is evidence the situation actually happens in this repo.
- Prefer the simplest implementation that matches the real project assumptions.
- Add exception handling only if:
  - the user explicitly asks for it, or
  - logs, telemetry, or current code paths show the failure is real.
- Do not inflate a patch just to cover hypothetical edge cases.

## File preservation policy
- Do not delete files unless the user explicitly asks for deletion.
- Do not silently remove files just because they look unused, old, duplicated, or inconvenient.
- If a file appears obsolete, report it as a candidate for cleanup, but leave it in place.
- Prefer editing, disabling, bypassing, or documenting over deleting.
- If a larger cleanup truly seems necessary, propose it first and wait for explicit approval.
