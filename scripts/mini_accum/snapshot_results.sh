#!/usr/bin/env bash
# --- SAFE HEADER (no cierres la shell) ---
set +e
set -o pipefail 2>/dev/null || true

safe_exit() { # usa esto en vez de 'exit'
  local code=${1:-0}
  if [ -n "$ZSH_EVAL_CONTEXT" ] && [[ $ZSH_EVAL_CONTEXT == *:file ]]; then return "$code"; fi
  if [ -n "${BASH_VERSION:-}" ] && [[ ${BASH_SOURCE[0]} != "$0" ]]; then return "$code"; fi
  exit "$code"
}

nonfatal(){ "$@" || printf '⚠️  ignorado: %s\n' "$*"; }
# --- /SAFE HEADER ---
set -Eeuo pipefail

# Uso:
#   snapshot_results.sh [REGEX_SUFFIX] [WINDOW ...]
# Ejemplos:
#   snapshot_results.sh                      # usa patrón por defecto y ventanas 2024H1, 2023Q4
#   BASE=1.05 snapshot_results.sh            # cambia baseline para Δbps
#   snapshot_results.sh 'v3p3N2g-F2-H1_FZ-'  2024H1
#   snapshot_results.sh 'v3p3N2g-F2-(H1_FZ|Q4_E3)-'  2024H1 2023Q4

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
PR="$ROOT/scripts/mini_accum/pretty_results.sh"
DOC_DIR="$ROOT/docs/Mini_accum"
mkdir -p "$DOC_DIR"

PATTERN="${1:-v3p3N2g-F2-(H1_FZ|Q4_E3)-}"
shift || true
if (( "$#" > 0 )); then WINDOWS=("$@"); else WINDOWS=(2024H1 2023Q4); fi

if [[ ! -x "$PR" ]]; then
  echo "❌ No encuentro ejecutable: $PR" >&2
  safe_exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$DOC_DIR/snapshot_${STAMP}.txt"

{
  echo "# snapshot_results"
  echo "# pattern: $PATTERN"
  echo "# windows: ${WINDOWS[*]}"
  echo "# BASE: ${BASE:-auto}"
  echo
  for W in "${WINDOWS[@]}"; do
    echo "== $W =="
    # Respeta BASE si viene del entorno; si no, pretty_results usa su default
    BASE="${BASE-}" bash "$PR" "$PATTERN" "$W"
    echo
  done
} | tee "$OUT"

# Mantener/actualizar symlink del último snapshot (relativo dentro de DOC_DIR)
ln -sfn "$(basename "$OUT")" "$DOC_DIR/latest_snapshot.txt"

echo "→ Escribí $OUT"
echo "→ Actualicé latest_snapshot.txt -> $(basename "$OUT")"
