#!/usr/bin/env bash
set -euo pipefail

TAG="${1:?Pasa 2022|2023|2024|2025H1}"

case "$TAG" in
  2022)   Y=2022; START=2022-01-01; END=2022-12-31 ;;
  2023)   Y=2023; START=2023-01-01; END=2023-12-31 ;;
  2024)   Y=2024; START=2024-01-01; END=2024-12-31 ;;
  2025H1) Y=2025; START=2025-01-01; END=2025-06-30 ;;
  *) echo "TAG inválido: $TAG"; exit 1;;
esac

HALVINGS=(2012 2016 2020 2024)
last=2012; for h in "${HALVINGS[@]}"; do (( h<=Y )) && last=$h; done
years_since=$((Y-last))

KISS_V1_CFG="configs/mini_accum/presets/CORE_2025.yaml"
E1_CFG="configs/mini_accum/presets/E1_Y2.yaml"

CFG="$KISS_V1_CFG"; (( years_since == 2 )) && CFG="$E1_CFG"
CFG="${MA_FORCE_CFG:-$CFG}"

echo "[REGIME] TAG=$TAG Y=$Y (+$years_since post-halving) -> $CFG"
python -m mini_accum.cli --config "$CFG" --start "$START" --end "$END" --suffix "OOS_${TAG}_REGIME"
