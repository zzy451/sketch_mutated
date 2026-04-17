---
name: patch-verify
description: make the smallest reversible code patch in sketch_mutated, then run a fast verification pass before declaring success. use when editing one or a few files, fixing a bug, tightening imports, adjusting telemetry, or making a surgical repo-local patch.
---

Use this skill for small or medium code patches, not for broad proposal redesign.

Workflow:
1. Name the owner file and why it owns the change.
2. Make the smallest reversible patch that can test the hypothesis.
3. Avoid drifting into unrelated multi-file cleanup.
4. Run a fast verification pass immediately.
5. Report:
   - files changed
   - what changed
   - what was verified
   - what remains unverified
   - expected telemetry/log delta

Fast verification helper:
- `bash .agents/skills/patch-verify/scripts/quick-verify.sh [optional_file1 optional_file2 ...]`

Default behavior:
- if file arguments are omitted, the script tries changed Python files first
- otherwise it falls back to the repo core Python files

Do not claim the patch is done if verification was skipped.

Additional constraint:
- Do not add defensive code for hypothetical failures.
- Do not turn a small patch into a fallback-heavy patch unless the failure is real and reproducible.
- Keep verification focused on the changed behavior, not imagined edge cases.

Additional constraint:
- Do not delete files as part of a small patch.
- If a patch seems to require removing a file, stop and ask first instead of deleting it.
- Prefer local edits, narrower imports, or explicit deactivation over file removal.
