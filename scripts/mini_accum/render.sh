#!/usr/bin/env bash
set -euo pipefail

# Rellena plantillas YAML con tokens. Escoge template según EXIT_DELTA_BPS.
# Uso:
#   YAML_OUT=... XB=36 DW=72 MB=20 EMA_FAST=13 ATR_PAS=false ATR_P=40 \
#   EXIT_DELTA_BPS=0 CONFIRM=0 WEEKLY_HARD=false TTL=1 TTL_CONFIRM=0 \
#   bash scripts/mini_accum/render.sh
#
# Variables (con defaults razonables)
YAML_OUT="${YAML_OUT:-configs/exp/_render.yaml}"
XB="${XB:-36}"
DW="${DW:-72}"
MB="${MB:-20}"
EMA_FAST="${EMA_FAST:-13}"
ATR_PAS="${ATR_PAS:-true}"
ATR_P="${ATR_P:-40}"
YELLOW_BAND="${YELLOW_BAND:-5}"
EXIT_DELTA_BPS="${EXIT_DELTA_BPS:-0}"     # 0 => sin delta; >0 => usa template_exitdelta
CONFIRM="${CONFIRM:-0}"
WEEKLY_HARD="${WEEKLY_HARD:-false}"
TTL="${TTL:-1}"
TTL_CONFIRM="${TTL_CONFIRM:-0}"

# Paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
TPL_DIR="$REPO_ROOT/configs/templates"
TPL_A="$TPL_DIR/mini_accum__template.yaml"
TPL_B="$TPL_DIR/mini_accum__template_exitdelta.yaml"

mkdir -p "$(dirname "$YAML_OUT")"

# Calcula factor de salida si hay delta en bps
EXIT_FACTOR="1.0"
if [[ "$EXIT_DELTA_BPS" != "0" ]]; then
  # factor = 1 - delta_bps/10000
  EXIT_FACTOR=$(python - <<PY
d = float("${EXIT_DELTA_BPS}")/10000.0
print(f"{1.0 - d:.6f}")
PY
)
fi

TPL="$TPL_A"
if [[ "$EXIT_DELTA_BPS" != "0" ]]; then
  TPL="$TPL_B"
fi

# Render simple con sed
sed -e "s/__EMA_FAST__/${EMA_FAST}/g" \
    -e "s/__XB__/${XB}/g" \
    -e "s/__MB__/${MB}/g" \
    -e "s/__DW__/${DW}/g" \
    -e "s/__ATR_PAS__/${ATR_PAS}/g" \
    -e "s/__ATR_P__/${ATR_P}/g" \
    -e "s/__YELLOW__/${YELLOW_BAND}/g" \
    -e "s/__CONFIRM__/${CONFIRM}/g" \
    -e "s/__TTL__/${TTL}/g" \
    -e "s/__TTLCONF__/${TTL_CONFIRM}/g" \
    -e "s/__WEEKLY_HARD__/${WEEKLY_HARD}/g" \
    -e "s/__EXIT_FACTOR__/${EXIT_FACTOR}/g" \
    "$TPL" > "$YAML_OUT"

echo "[render] $YAML_OUT"
