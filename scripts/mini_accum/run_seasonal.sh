#!/usr/bin/env bash
set -euo pipefail

START="${1:?start yyyy-mm-dd}"
END="${2:?end yyyy-mm-dd}"
TAG="${3:-run}"

CFG_H1="configs/baselines/F2_H1_FZ.yaml"
CFG_Q4="configs/baselines/F2_Q4_E3.yaml"

MONTH=$(date -j -f "%Y-%m-%d" "$START" "+%m" 2>/dev/null || date -d "$START" "+%m")
if [[ "$MONTH" == "10" || "$MONTH" == "11" || "$MONTH" == "12" ]]; then
  CFG="$CFG_Q4"; SUF="Q4_E3"
else
  CFG="$CFG_H1"; SUF="H1_FZ"
fi

REPORT_SUFFIX="v3p3N2g-F2-${SUF}-${TAG}" \
mini-accum-backtest --config "$CFG" --start "$START" --end "$END"
