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

FILES=(common.py evaluator.py llm_engine.py init_dex_language.py update_language.py query_language.py)

echo "[primitive-runtime-alignment] using python: ${PYBIN}" >&2
echo "[primitive-runtime-alignment] py_compile check..." >&2
"${PYBIN}" -m py_compile "${FILES[@]}"

HAS_DEAP="$("${PYBIN}" - <<'PY'
import importlib.util
print("1" if importlib.util.find_spec("deap") else "0")
PY
)"

if [ "${HAS_DEAP}" != "1" ]; then
  echo "[primitive-runtime-alignment] warning: deap not installed, skip import smoke." >&2
  echo "[primitive-runtime-alignment] activate the project env first if you want full runtime/import alignment validation." >&2
  if [ "${STRICT_IMPORT_SMOKE:-0}" = "1" ]; then
    echo "[primitive-runtime-alignment] STRICT_IMPORT_SMOKE=1 and deap missing -> fail." >&2
    exit 1
  fi
  exit 0
fi

echo "[primitive-runtime-alignment] import smoke check..." >&2
"${PYBIN}" - <<'PY'
import importlib, json, sys
mods = [
    "common",
    "evaluator",
    "llm_engine",
    "init_dex_language",
    "update_language",
    "query_language",
]
ok = []
bad = {}
for m in mods:
    try:
        importlib.import_module(m)
        ok.append(m)
    except Exception as e:
        bad[m] = f"{type(e).__name__}: {e}"

print(json.dumps({
    "ok": not bad,
    "import_ok": ok,
    "import_failed": bad
}, ensure_ascii=False, indent=2))

if bad:
    sys.exit(1)
PY
