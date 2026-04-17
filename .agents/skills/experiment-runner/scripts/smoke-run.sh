#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
mkdir -p .claude/logs/experiment-runner

if [ $# -eq 0 ]; then
  echo "usage: bash .claude/skills/experiment-runner/scripts/smoke-run.sh -- <command ...>" >&2
  exit 1
fi

if [ "${1:-}" = "--" ]; then
  shift
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG=".claude/logs/experiment-runner/smoke_${TS}.log"

echo "[experiment-runner] running: $*" >&2
"$@" 2>&1 | tee "$LOG"

echo "[experiment-runner] log saved: $LOG" >&2
echo "[experiment-runner] key lines:" >&2
grep -Ei 'best fitness|best error|holdout|stage2|risk_reason_topk|novelty_|repair_injected|novelty_injected' "$LOG" | tail -n 60 || true
