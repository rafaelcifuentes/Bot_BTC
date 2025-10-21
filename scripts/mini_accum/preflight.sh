#!/usr/bin/env bash
: "${ZSH_EVAL_CONTEXT:=}"

# --- SAFE HEADER (no cierres la shell) ---
set +e
set -o pipefail 2>/dev/null || true

safe_exit() { # usa esto en vez de 'exit'
  local code=${1:-0}
  if [ -n "${ZSH_EVAL_CONTEXT-}" ] && [[ ${ZSH_EVAL_CONTEXT-} == *:file ]]; then return "$code"; fi
  if [ -n "${BASH_VERSION:-}" ] && [[ ${BASH_SOURCE[0]} != "$0" ]]; then return "$code"; fi
  exit "$code"
}

nonfatal(){ "$@" || printf '⚠️  ignorado: %s\n' "$*"; }
# --- /SAFE HEADER ---
set -Eeuo pipefail
need(){ command -v "$1" >/dev/null || { echo "Falta $1"; safe_exit 1; }; }
# [skip: opcional] need yq; need awk; need mini-accum-backtest
CFG="${1:-configs/mini_accum/presets/CORE_2025.yaml}"
yq -e '.' "$CFG" >/dev/null
H4=$(yq -r '.data.ohlc_4h_csv' "$CFG"); D1=$(yq -r '.data.ohlc_d1_csv' "$CFG")
[[ -f "$H4" && -f "$D1" ]] || { echo "No encuentro OHLC: $H4 o $D1"; safe_exit 1; }
echo "[OK] Preflight: $CFG"
