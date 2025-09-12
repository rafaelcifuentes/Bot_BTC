#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
VENV_BIN="${VENV_BIN:-$REPO_ROOT/.venv/bin}"
BT="$VENV_BIN/mini-accum-backtest"

XB="${XB:-36}"
DW="${DW:-72}"
MB="${MB:-20}"
EMA="${EMA:-13}"
CONFIRM="${CONFIRM:-0}"
TTL="${TTL:-1}"
TTL_CONFIRM="${TTL_CONFIRM:-0}"

# ATR variants
#   A: pause_affects_sell=false
#   B: pause_affects_sell=true; percentile_p=35
#   C: pause_affects_sell=false; percentile_p=35
variants=("A:false:40" "B:true:35" "C:false:35")

run_one () {
  local tag="$1" pas="$2" p="$3" win="$4" s="$5" e="$6"
  local cfg="$REPO_ROOT/configs/exp/F2_${tag}_xb${XB}_dw${DW}.yaml"
  YAML_OUT="$cfg" XB="$XB" DW="$DW" MB="$MB" EMA_FAST="$EMA" ATR_PAS="$pas" ATR_P="$p" CONFIRM="$CONFIRM" TTL="$TTL" TTL_CONFIRM="$TTL_CONFIRM" \
    bash "$REPO_ROOT/scripts/mini_accum/render.sh" >/dev/null

  local suf="bb1-core-v3p3N2g-F2-${tag}-xb${XB}-dw${DW}-${win}"
  echo "[F2] $suf"
  REPORT_SUFFIX="$suf" "$BT" --config "$cfg" --start "$s" --end "$e"
}

for v in "${variants[@]}"; do
  IFS=: read -r tag pas p <<<"$v"
  run_one "$tag" "$pas" "$p" "oos23Q4" "2023-10-01" "2023-12-31"
  run_one "$tag" "$pas" "$p" "oos24H1" "2024-01-01" "2024-06-30"
done
