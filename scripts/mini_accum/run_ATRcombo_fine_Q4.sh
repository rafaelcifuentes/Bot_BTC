#!/usr/bin/env bash
set -euo pipefail

for CFG in configs/exp/F2_ATRxb36_dw76_p*_y*.yaml; do
  [ -e "$CFG" ] || continue
  # Extrae P e Y del nombre del archivo
  # ejemplo: F2_ATRxb36_dw76_p35_y005.yaml -> P=35, YTAG=005
  fname=$(basename "$CFG")
  P=$(echo "$fname" | sed -E 's/.*_p([0-9]+)_y([0-9]+)\.yaml/\1/')
  YTAG=$(echo "$fname" | sed -E 's/.*_p([0-9]+)_y([0-9]+)\.yaml/\2/')
  SUF="v3p3N2g-F2-xb36-dw76-ATRp${P}-y${YTAG}-oos23Q4"

  echo "[Q4] $SUF  <-  $CFG"
  REPORT_SUFFIX="$SUF" mini-accum-backtest --config "$CFG" \
    --start 2023-10-01 --end 2023-12-31
done
