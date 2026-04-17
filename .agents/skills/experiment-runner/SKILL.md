---
name: experiment-runner
description: run, compare, and summarize stage1 proxy, holdout, and stage2 real experiments for sketch_mutated. use when testing a patch, tuning search parameters, validating a closed-loop change, or preparing a compact baseline-vs-patched experiment report.
---

Use this skill for experiment execution and comparison, not for code patching.

Workflow:
1. Decide whether the run is a smoke run or a formal comparison.
2. Record all run-defining settings before execution: seed, proxy mode, dataset mode, dataset seeds, pkts/files/start/shuffle, pop/gen, llm mode, and holdout settings.
3. Prefer fixed streams for fair comparisons when comparing patches.
4. Run the command.
5. Extract and summarize only the load-bearing outputs.

Minimum summary fields:
- command
- seed
- proxy mode
- stage1 best fitness
- stage1 best error
- holdout real error if enabled
- stage2 real error
- whether llm path was enabled
- noteworthy telemetry (`risk_reason_topk`, novelty counts, repair vs novelty injects) if present

Do not compare two runs as if they were equivalent if any of these differ without saying so:
- dataset seeds
- fixed stream paths
- stage1 proxy mode
- pop/gen
- holdout enablement

Default smoke-run pattern for this repo:
- lower `--pop`
- lower `--gen`
- lower `--pkts`
- keep dataset seeds explicit

End with one recommendation:
- keep
- revert
- rerun with tighter controls
- patch before rerunning

## Local scripts
- smoke run wrapper: `bash .agents/skills/experiment-runner/scripts/smoke-run.sh -- <your command>`
- formal compare helper: `bash .agents/skills/experiment-runner/scripts/formal-compare.sh <before.log> <after.log>`

Use the wrapper scripts to keep logs under `.claude/logs/experiment-runner/`.
