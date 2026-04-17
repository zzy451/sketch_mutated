#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"

OUT="${1:-HANDOFF_$(date +%Y%m%d_%H%M%S).md}"
NOTE="${2:-}"

{
  echo "# Session Handoff"
  echo
  echo "## Time"
  date
  echo
  echo "## Current target"
  echo "- fill this in before handing off"
  echo
  echo "## Files touched"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git status --short || true
  else
    echo "- git not available"
  fi
  echo
  echo "## Recent diff stat"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git diff --stat || true
  else
    echo "- git not available"
  fi
  echo
  echo "## Recent commits"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git log --oneline -5 || true
  else
    echo "- git not available"
  fi
  echo
  echo "## Verified"
  echo "- fill this in"
  echo
  echo "## Blocker / uncertainty"
  echo "- fill this in"
  echo
  echo "## Next smallest step"
  echo "- fill this in"
  echo
  echo "## Optional note"
  echo "${NOTE}"
} > "$OUT"

echo "[session-handoff] wrote ${OUT}" >&2
echo "$OUT"
