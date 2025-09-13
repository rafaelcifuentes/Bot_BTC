#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_YAML="configs/exp/F2_ATRcombo_xb36_dw76.yaml"

P_LIST=(33 34 35 36 37)
Y_LIST=(0.04 0.05 0.06)

for P in "${P_LIST[@]}"; do
  for Y in "${Y_LIST[@]}"; do
    tagY="${Y/./}"                                  # 0.05 -> 005
    OUT="configs/exp/F2_ATRxb36_dw76_p${P}_y${tagY}.yaml"
    "$DIR/render_atr.sh" "$BASE_YAML" "$OUT" 36 76 "$P" "$Y"
    SUF="v3p3N2g-F2-xb36-dw76-ATRp${P}-y${tagY}-oos24H1"
    REPORT_SUFFIX="$SUF" mini-accum-backtest --config "$OUT" \
      --start 2024-01-01 --end 2024-06-30
  done
done
