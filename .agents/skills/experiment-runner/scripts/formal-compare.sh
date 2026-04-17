#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: bash .claude/skills/experiment-runner/scripts/formal-compare.sh <before.log> <after.log>" >&2
  exit 1
fi

BEFORE="$1"
AFTER="$2"
PATTERN='best fitness|best error|holdout|stage2|risk_reason_topk|novelty_mechanism_cluster_count|novelty_family_count|novelty_incubated|novelty_promoted|repair_injected|novelty_injected'

echo "===== BEFORE: ${BEFORE} ====="
grep -Ei "$PATTERN" "$BEFORE" || true
echo
echo "===== AFTER: ${AFTER} ====="
grep -Ei "$PATTERN" "$AFTER" || true
