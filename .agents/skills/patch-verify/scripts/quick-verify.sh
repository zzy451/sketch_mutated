#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python3" ]; then
  PYBIN="${VIRTUAL_ENV}/bin/python3"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python3" ]; then
  PYBIN="${CONDA_PREFIX}/bin/python3"
else
  PYBIN="python3"
fi

declare -a FILES=()

if [ "$#" -gt 0 ]; then
  FILES=("$@")
else
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    mapfile -t FILES < <( { git diff --name-only -- '*.py'; git diff --cached --name-only -- '*.py'; git ls-files -m '*.py'; } | awk 'NF' | sort -u )
  fi
fi

if [ "${#FILES[@]}" -eq 0 ]; then
  FILES=(common.py cli.py evaluator.py helpers.py llm_engine.py evolution.py mutate_cmsketch_refactored.py init_dex_language.py update_language.py query_language.py)
fi

echo "[patch-verify] using python: ${PYBIN}" >&2
echo "[patch-verify] py_compile files:" >&2
printf '  %s\n' "${FILES[@]}" >&2
"${PYBIN}" -m py_compile "${FILES[@]}"

HAS_DEAP="$("${PYBIN}" - <<'PY'
import importlib.util
print("1" if importlib.util.find_spec("deap") else "0")
PY
)"

if [ "${HAS_DEAP}" != "1" ]; then
  echo "[patch-verify] warning: deap not installed in current python env, skip import smoke." >&2
  echo "[patch-verify] tip: activate the project env first, or run STRICT_IMPORT_SMOKE=1 to force strict failure." >&2
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[patch-verify] git diff --stat" >&2
    git diff --stat || true
  fi
  if [ "${STRICT_IMPORT_SMOKE:-0}" = "1" ]; then
    echo "[patch-verify] STRICT_IMPORT_SMOKE=1 and deap missing -> fail." >&2
    exit 1
  fi
  exit 0
fi

echo "[patch-verify] import smoke for root modules..." >&2
"${PYBIN}" - <<'PY'
import importlib, json, sys
core = [
    "common",
    "cli",
    "evaluator",
    "helpers",
    "llm_engine",
    "evolution",
    "mutate_cmsketch_refactored",
    "init_dex_language",
    "update_language",
    "query_language",
]
ok, bad = [], {}
for m in core:
    try:
        importlib.import_module(m)
        ok.append(m)
    except Exception as e:
        bad[m] = f"{type(e).__name__}: {e}"
print(json.dumps({"ok": not bad, "import_ok": ok, "import_failed": bad}, ensure_ascii=False, indent=2))
if bad:
    sys.exit(1)
PY

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[patch-verify] git diff --stat" >&2
  git diff --stat || true
fi
