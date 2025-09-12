#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
VENV_BIN="${VENV_BIN:-$REPO_ROOT/.venv/bin}"
BT="$VENV_BIN/mini-accum-backtest"

XB="${XB:-36}"
DW="${DW:-72}"
MB="${MB:-20}"
EMA_SET=(${EMA_SET:-12 14})
CONFIRM="${CONFIRM:-0}"
TTL="${TTL:-1}"
TTL_CONFIRM="${TTL_CONFIRM:-0}"

run_one () {
  local ema="$1" win="$2" s="$3" e="$4"
  local cfg="$REPO_ROOT/configs/exp/F4_xb${XB}_dw${DW}_ema${ema}.yaml"
  YAML_OUT="$cfg" XB="$XB" DW="$DW" MB="$MB" EMA_FAST="$ema" CONFIRM="$CONFIRM" TTL="$TTL" TTL_CONFIRM="$TTL_CONFIRM" \
    bash "$REPO_ROOT/scripts/mini_accum/render.sh" >/dev/null

  local suf="bb1-core-v3p3N2g-F4-xb${XB}-dw${DW}-ema${ema}-${win}"
  echo "[F4] $suf"
  REPORT_SUFFIX="$suf" "$BT" --config "$cfg" --start "$s" --end "$e"
}

for ema in "${EMA_SET[@]}"; do
  run_one "$ema" "oos23Q4" "2023-10-01" "2023-12-31"
  run_one "$ema" "oos24H1" "2024-01-01" "2024-06-30"
done
