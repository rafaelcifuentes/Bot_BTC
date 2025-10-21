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
set -euo pipefail
# Ejecuta el CLI empaquetado leyendo configs/mini_accum/config.yaml
mini-accum-backtest --config "configs/mini_accum/config.yaml" "$@"
