---
name: telemetry-triage
description: inspect logs and telemetry in sketch_mutated to identify the current bottleneck and recommend the next patch. use when runs stall, novelty collapses, numeric explosions return, repair dominates, or mechanism diversity fails to survive.
---

Map observed telemetry to one of the three main closed loops:
- numeric-risk front-gating
- novelty from family survival to mechanism-cluster multi-peak branching
- mechanism-best feedback into main search

Prioritize these fields when present:
- `risk_reject_by_phase`
- `risk_warning_but_allowed_by_phase`
- `risk_reason_topk`
- `DIAG_CHUNK` values such as `penalty_avg`, `query_avg`, `total_avg`
- `novelty_mechanism_cluster_count`
- `novelty_family_count`
- `dominant_family_cooldown_hits`
- `mechanism_override_passes`
- `novelty_incubated`
- `novelty_promoted`
- `repair_injected` vs `novelty_injected`
- duplicate canonical reject share

Triage procedure:
1. Identify the dominant failure pattern.
2. Classify it as seed hygiene, numeric instability, novelty collapse, repair over-dominance, or mechanism feedback not reaching the main search.
3. Point to the patch owner file.
4. Propose the smallest next patch that would change the decisive telemetry.

Do not recommend broad refactors without naming:
- the target file
- the target function or responsibility
- the expected telemetry delta
