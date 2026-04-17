---
name: session-handoff
description: write a compact handoff note for sketch_mutated before switching windows, agents, or sessions. use when pausing vibecoding, after a partial patch, before a long run, or when the next step is likely to be resumed later.
---

Use this skill to preserve momentum across sessions.

Workflow:
1. State the current target.
2. List touched files.
3. Record what was verified.
4. Record the blocker or uncertainty.
5. Record the next smallest command or patch.
6. Save the handoff note to a timestamped markdown file.

Helper:
- `bash .agents/skills/session-handoff/scripts/write-handoff.sh [optional_output_file] ["optional note"]`

A good handoff note should let a new agent resume without rereading the entire conversation.
