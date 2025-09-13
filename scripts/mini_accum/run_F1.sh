#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XB_LIST=(32 34 36 38)
DW_LIST=(72 76 80)

for xb in "${XB_LIST[@]}"; do
  for dw in "${DW_LIST[@]}"; do
    out="configs/exp/F1_xb${xb}_dw${dw}.yaml"
    bash "$DIR/render.sh" "$out" "$xb" "$dw"   # MB se puede override: MB=18 bash ...
    echo "[run] xb=$xb dw=$dw :: Q4"
    REPORT_SUFFIX="v3p3N2g-F1-xb${xb}-dw${dw}-oos23Q4" \
    mini-accum-backtest --config "$out" --start 2023-10-01 --end 2023-12-31
    echo "[run] xb=$xb dw=$dw :: H1"
    REPORT_SUFFIX="v3p3N2g-F1-xb${xb}-dw${dw}-oos24H1" \
    mini-accum-backtest --config "$out" --start 2024-01-01 --end 2024-06-30
  done
done

# Scoreboard compacto (H1)
mini-accum-dictamen --reports-dir reports/mini_accum --out /tmp/d.tsv --format tsv --quiet
(
  echo -e "suffix\twindow\tflips\tfpy\tnetBTC\tmdd_vs_HODL\tPASS\tFAIL"
  awk -F'\t' 'NR>1 && $1 ~ /v3p3N2g-F1-xb(32|34|36|38)-dw(72|76|80)/ && $2=="2024H1" {printf "%s\t%s\t%d\t%.1f\t%.6f\t%.6f\t%s\t%s\n",$1,$2,$7,$8,$3,$6,$9,$10}' /tmp/d.tsv \
  | sort -t $'\t' -k5,5nr
) | column -t -s $'\t'
