---
name: proposal-sync
description: keep the proposal and the split-file implementation synchronized in sketch_mutated. use when requirements change, direct-construction files change, primitive language files are added or promoted, module responsibilities move, or the user asks to update both proposal and code together.
---

Treat proposal updates as executable design changes, not documentation-only edits.

Workflow:
1. Read the affected proposal section and locate the matching code owner file.
2. For every requested change, answer four questions:
   - which file changes?
   - which function or responsibility changes?
   - which closed loop does it advance? (`numeric-risk`, `novelty/mechanism-cluster`, or `mechanism feedback`)
   - which log or telemetry output should change after the patch?
3. Update `Sketch_Global_Proposal_SplitFiles_v3.md` and the impacted code in the same pass.
4. If the change touches primitive semantics or primitive ownership, update the three primitive source-of-truth files and mention them explicitly in the proposal.
5. Verify imports, path assumptions, and reporting text still match the repository layout.

Always keep these files first-class in proposal scope when relevant:
- `common.py`
- `cli.py`
- `evaluator.py`
- `helpers.py`
- `llm_engine.py`
- `evolution.py`
- `init_dex_language.py`
- `update_language.py`
- `query_language.py`

When reporting, separate:
- proposal changes
- code changes
- remaining drift

Additional constraint:
- Do not delete files during proposal/code sync unless the user explicitly requests it.
- Keep proposal references and implementation references aligned to the existing file set.
- If a file appears obsolete, record the drift or cleanup suggestion, but do not remove the file.
