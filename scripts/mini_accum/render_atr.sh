#!/usr/bin/env bash
set -euo pipefail
BASE=${1:-configs/exp/F2_ATRcombo_xb36_dw76.yaml}
OUT=${2:-configs/exp/tmp.yaml}
XB=${3:-36}
DW=${4:-76}
P=${5:-35}
Y=${6:-0.05}

mkdir -p "$(dirname "$OUT")"
# Sustituciones robustas por clave, respetando indentación
sed -E \
  -e "s/^([[:space:]]*cross_buffer_bps:).*/\1 ${XB}/" \
  -e "s/^([[:space:]]*dwell_bars_min_between_flips:).*/\1 ${DW}/" \
  -e "s/^([[:space:]]*percentile_p:).*/\1 ${P}/" \
  -e "s/^([[:space:]]*yellow_band_pct:).*/\1 ${Y}/" \
  "$BASE" > "$OUT"
