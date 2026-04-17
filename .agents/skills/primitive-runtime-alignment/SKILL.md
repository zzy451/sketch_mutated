---
name: primitive-runtime-alignment
description: align primitive source-of-truth files with search-time runtime, exported runtime, and llm primitive validation in sketch_mutated. use when primitives are added, renamed, or retyped; when init_dex/list_3 semantics change; when overflow or plane handling changes; or when evaluator and primitive files disagree.
---

Treat `init_dex_language.py`, `update_language.py`, and `query_language.py` as the primitive source-of-truth.

Audit these alignment surfaces together:
- primitive signatures in the three language files
- pset registration in those files
- runtime assumptions in `evaluator.py`
- registry glue in `common.py`
- dual-source validation in `llm_engine.py`
- any exported runtime helpers or generated runtime code paths

Always check these failure modes:
- `list_3` tuple shape drift
- `x/y/z` or plane-routing drift
- `overflow_matrices` vs any stale `overflow_state` names
- forbidden cross-phase primitive use
- root return type mismatch (`list` vs `float`)
- search-time runtime and exported runtime disagreeing on the same primitive

Suggested verification snippets after edits:
- `python3 -m py_compile common.py evaluator.py llm_engine.py init_dex_language.py update_language.py query_language.py`
- run a tiny import smoke test from project root to ensure all three primitive files load

When reporting, list:
- signature mismatches fixed
- runtime mismatches fixed
- remaining ambiguities

## Local scripts
- verification helper: `bash .agents/skills/primitive-runtime-alignment/scripts/verify-alignment.sh`

Use the script after primitive-signature, registry-glue, evaluator-runtime, or LLM validation changes.

Additional constraint:
- Do not add defensive branches for hypothetical primitive/runtime mismatches.
- Fix only the mismatches that are evidenced by current source-of-truth files, runtime behavior, validation failures, or logs.
- Do not add fallback semantics for impossible primitive states.
