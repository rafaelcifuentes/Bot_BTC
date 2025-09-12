#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
VENV_BIN="${VENV_BIN:-$REPO_ROOT/.venv/bin}"
BT="$VENV_BIN/mini-accum-backtest"

XB="${XB:-36}"
DW="${DW:-72}"
MB="${MB:-20}"
EMA="${EMA:-13}"
TTL="${TTL:-1}"
TTL_CONFIRM="${TTL_CONFIRM:-0}"

# delta en bps y confirm bars a testear
DELTAS=(${DELTAS:-5 10})
CONFIRMS=(${CONFIRMS:-0 1})

run_one () {
  local d="$1" c="$2" win="$3" s="$4" e="$5"
  local cfg="$REPO_ROOT/configs/exp/F3_xb${XB}_dw${DW}_d${d}_c${c}.yaml"
  YAML_OUT="$cfg" XB="$XB" DW="$DW" MB="$MB" EMA_FAST="$EMA" EXIT_DELTA_BPS="$d" CONFIRM="$c" TTL="$TTL" TTL_CONFIRM="$TTL_CONFIRM" \
    bash "$REPO_ROOT/scripts/mini_accum/render.sh" >/dev/null

  local suf="bb1-core-v3p3N2g-F3-xb${XB}-dw${DW}-d${d}-c${c}-${win}"
  echo "[F3] $suf"
  REPORT_SUFFIX="$suf" "$BT" --config "$cfg" --start "$s" --end "$e"
}

for d in "${DELTAS[@]}"; do
  for c in "${CONFIRMS[@]}"; do
    run_one "$d" "$c" "oos23Q4" "2023-10-01" "2023-12-31"
    run_one "$d" "$c" "oos24H1" "2024-01-01" "2024-06-30"
  done
done
