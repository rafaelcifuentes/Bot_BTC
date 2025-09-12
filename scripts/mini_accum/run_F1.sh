#!/usr/bin/env bash
set -euo pipefail

XB_LIST=${XB_LIST:-"32 34 36 38"}
DW_LIST=${DW_LIST:-"72 76 80"}
MB=${MB:-20}
EMA_FAST=${EMA_FAST:-13}
CONFIRM=${CONFIRM:-0}
TTL=${TTL:-1}
TTL_CONFIRM=${TTL_CONFIRM:-0}

tpl="configs/exp/base_c0.yaml"
for xb in $XB_LIST; do
  for dw in $DW_LIST; do
    export XB=$xb DW=$dw MB=$MB EMA_FAST=$EMA_FAST CONFIRM=$CONFIRM TTL=$TTL TTL_CONFIRM=$TTL_CONFIRM
    out=$(bash scripts/render.sh "$tpl" ".tmp_configs/F1_xb${xb}_dw${dw}.yaml")
    # Q4
    REPORT_SUFFIX="v3p3N2g-F1-xb${xb}-dw${dw}-oos23Q4" \
      mini-accum-backtest --config "$out" --start 2023-10-01 --end 2023-12-31
    # H1
    REPORT_SUFFIX="v3p3N2g-F1-xb${xb}-dw${dw}-oos24H1" \
      mini-accum-backtest --config "$out" --start 2024-01-01 --end 2024-06-30
  done
done
echo "[F1] listo"
