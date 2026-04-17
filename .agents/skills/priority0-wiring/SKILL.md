---
name: priority0-wiring
description: handle split-file wiring and primitive source-of-truth alignment in sketch_mutated. use when working on common.py, cli.py, evaluator.py, llm_engine.py, init_dex_language.py, update_language.py, query_language.py, or when the user mentions priority 0, primitive registry, default primitive reference paths, flat/package imports, startup consistency logging, or runtime/reference alignment.
---

This skill is the reusable version of the old `PRIORITY0_TODO.md` instruction.
Read `references/checklist.md` first and follow it as the execution checklist.

Execution order:
1. Audit the three primitive source-of-truth files.
2. Patch `common.py` registry glue without duplicating primitive truth.
3. Patch `cli.py` so default primitive reference paths bind automatically.
4. Patch import fallbacks only where needed for flat/package compatibility.
5. Check `evaluator.py` and `llm_engine.py` for runtime/reference drift.
6. Run a focused verification pass.

Focused verification:
`python3 -m py_compile common.py cli.py evaluator.py helpers.py llm_engine.py evolution.py mutate_cmsketch_refactored.py init_dex_language.py update_language.py query_language.py`

Report with exactly these sections:
- Files changed
- What was wired
- Remaining mismatches
- Why this helps

Additional constraint:
- Do not add defensive fallback logic for impossible wiring states.
- Only patch import fallback, registry fallback, or path fallback when the problem is real and reproducible in this repo.
- Do not widen Priority 0 patches just to guard hypothetical launch styles that are not actually used.

Additional constraint:
- Do not delete files during Priority 0 wiring.
- Priority 0 should connect, align, or minimally patch the existing split-file structure, not shrink it by removing files.
- If a file seems stale, report it as a possible future cleanup item, but keep it intact.
