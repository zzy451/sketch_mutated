# Priority 0 TODO: Split-File Wiring and Primitive Source-of-Truth Alignment

## Goal

Finish the initial wiring so that the split-file project has a stable foundation before deeper loop work begins.

This stage must focus on:

- primitive registry consistency,
- default primitive reference path injection,
- source-of-truth alignment,
- flat/package compatibility,
- startup consistency logging.

Do not jump into large novelty or mechanism-feedback changes before this layer is stable.

---

## Scope

Primary files for Priority 0:

- `common.py`
- `cli.py`
- `init_dex_language.py`
- `update_language.py`
- `query_language.py`

Secondary files that may need light alignment only:

- `llm_engine.py`
- `evaluator.py`

Do not expand unrelated search-policy logic in this stage.

---

## Task 1: Primitive source-of-truth audit

### Objective
Confirm that the real primitive grammar/reference truth lives in:

- `init_dex_language.py`
- `update_language.py`
- `query_language.py`

### Checklist
- enumerate exported primitives from each file
- check whether each file exposes a clear registry or registry-like structure
- identify missing metadata needed by `common.py`
- identify overlapping or inconsistent primitive naming
- identify signature mismatches
- identify import/path assumptions that break flat/package execution

### Expected output
- a short audit summary
- a list of primitive groups by file
- a list of signature inconsistencies
- a list of missing registry glue requirements

---

## Task 2: Build unified primitive registry in `common.py`

### Objective
Use `common.py` as the central glue layer, not as a duplicate source-of-truth.

### Checklist
- create or refine a unified registry builder
- map init/update/query primitive groups into one shared registry view
- preserve ownership in the three primitive files
- support grammar version / primitive tier switches if already present
- avoid copying semantic truth into multiple places

### Expected output
- unified registry access path
- clear ownership boundaries
- no duplicated primitive truth

---

## Task 3: Default primitive reference path injection in `cli.py`

### Objective
When no explicit primitive reference paths are passed, the system should automatically bind to:

- `init_dex_language.py`
- `update_language.py`
- `query_language.py`

### Checklist
- add default path discovery relative to current project
- support stable behavior in the current directory layout
- keep explicit CLI overrides valid
- print startup consistency logs showing which paths were bound

### Expected output
- startup log showing bound primitive reference files
- no silent fallback
- correct behavior with or without explicit path args

---

## Task 4: Flat/package compatibility cleanup

### Objective
Make the project robust whether it is run as a flat script layout or package-like layout.

### Checklist
- inspect imports in `common.py`, `cli.py`, `llm_engine.py`, `evaluator.py`
- identify relative-import vs direct-import breakpoints
- normalize import strategy
- avoid introducing fragile path hacks unless necessary
- verify project can be launched from the current working directory

### Expected output
- reduced import-path fragility
- no hidden dependency on one launch style only

---

## Task 5: Registry/runtime alignment check

### Objective
Check whether runtime-executed semantics in `evaluator.py` remain aligned with the primitive source-of-truth files.

### Checklist
- compare source-of-truth primitive signatures vs evaluator expectations
- identify primitives that need runtime-side support
- identify mismatches between search-time runtime and any export/runtime path
- do not redesign runtime yet unless needed for consistency

### Expected output
- a mismatch list
- a minimal alignment patch plan
- no large runtime expansion in Priority 0

---

## Task 6: LLM dual-source validation alignment

### Objective
Ensure `llm_engine.py` can validate proposals against the actual primitive source-of-truth.

### Checklist
- inspect how primitive validation currently works
- connect validation to the real primitive reference files
- identify stale motif/primitive catalogue assumptions
- log validation failure reasons clearly

### Expected output
- validation reads from correct sources
- clearer mismatch diagnostics
- fewer silent primitive-reference failures

---

## Acceptance Criteria

Priority 0 is complete only when:

1. the three primitive files are clearly established as source-of-truth
2. `common.py` provides unified registry glue without duplicating truth
3. `cli.py` injects default primitive reference paths correctly
4. startup logs clearly show primitive consistency binding
5. flat/package import issues are reduced or removed
6. `llm_engine.py` validation is aligned with the real primitive references
7. major runtime/reference mismatches are identified and listed

---

## Non-Goals

Do not prioritize these during Priority 0:

- mechanism-cluster novelty expansion
- repair budget redesign
- mechanism champion feedback rollout
- large primitive set expansion
- major runtime feature expansion
- broad LLM prompt redesign

These belong to later priorities.

---

## Suggested Working Order

1. audit primitive files
2. patch `common.py` registry glue
3. patch `cli.py` default reference path injection and startup consistency log
4. fix flat/package import fragility
5. inspect `llm_engine.py` dual-source validation path
6. inspect `evaluator.py` for obvious registry/runtime mismatches
7. produce a short Priority 0 completion summary

---

## Required Summary Format After Priority 0 Work

When reporting back after Priority 0 changes, always include:

### Files changed
- list of touched files

### What was wired
- registry glue
- default paths
- consistency logging
- import compatibility
- validation alignment

### Remaining mismatches
- runtime/reference mismatches
- validation gaps
- import/path edge cases

### Why this helps
- why this stabilizes the split-file foundation
- why this is necessary before seed hygiene / novelty cluster work
