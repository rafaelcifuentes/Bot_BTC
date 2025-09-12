#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
VENV_BIN="${VENV_BIN:-$REPO_ROOT/.venv/bin}"
BT="$VENV_BIN/mini-accum-backtest"

XB="${XB:-36}"
DW="${DW:-72}"
MB_SET=(${MB_SET:-18 20 22})
EMA="${EMA:-13}"
CONFIRM="${CONFIRM:-0}"
TTL="${TTL:-1}"
TTL_CONFIRM="${TTL_CONFIRM:-0}"
WEEKLY_HARD="${WEEKLY_HARD:-false}"

run_one () {
  local mb="$1" win="$2" s="$3" e="$4"
  local cfg="$REPO_ROOT/configs/exp/F1b_xb${XB}_dw${DW}_mb${mb}.yaml"
  YAML_OUT="$cfg" XB="$XB" DW="$DW" MB="$mb" EMA_FAST="$EMA" CONFIRM="$CONFIRM" TTL="$TTL" TTL_CONFIRM="$TTL_CONFIRM" WEEKLY_HARD="$WEEKLY_HARD" \
    bash "$REPO_ROOT/scripts/mini_accum/render.sh" >/dev/null

  local suf="bb1-core-v3p3N2g-F1b-xb${XB}-dw${DW}-mb${mb}-${win}"
  echo "[F1b] $suf"
  REPORT_SUFFIX="$suf" "$BT" --config "$cfg" --start "$s" --end "$e"
}

for mb in "${MB_SET[@]}"; do
  run_one "$mb" "oos23Q4" "2023-10-01" "2023-12-31"
  run_one "$mb" "oos24H1" "2024-01-01" "2024-06-30"
done
