---
name: project-orient
description: map the sketch_mutated repository before editing. use when starting work in this project, when the request is broad or ambiguous, when you need to explain the codebase, or when you need to decide which module or skill should own the next change.
---

Start by reading:
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

Build a short working map:
1. Summarize the current project goal in one paragraph.
2. Identify the direct-construction files.
3. Identify which file owns runtime semantics, GP context, LLM proposal logic, island/evolution dynamics, CLI configuration, and primitive source-of-truth.
4. Route the task to one of the project skills if it is recurring.

Use this routing table:
- wiring / primitive registry / default primitive path / import compatibility -> `priority0-wiring`
- proposal changes that must sync into code -> `proposal-sync`
- primitive signature or runtime semantic mismatch -> `primitive-runtime-alignment`
- running or comparing experiments -> `experiment-runner`
- reading logs and deciding next patches -> `telemetry-triage`

Do not start a multi-file patch before the ownership map is clear.

Additional routing:
- smallest reversible code patch + fast verification -> `patch-verify`
- summarize current state before switching window/agent -> `session-handoff`

Additional constraint:
- Do not expand a task because of hypothetical edge cases.
- Do not recommend extra fallback branches for situations that are not evidenced in the current repo, logs, or request.
- Keep the ownership map focused on real responsibilities and real failure paths.

Additional constraint:
- Do not recommend file deletion during orientation.
- Build the ownership map using the existing file set.
- If some files look obsolete or overlapping, note them as cleanup candidates, but do not delete them.
