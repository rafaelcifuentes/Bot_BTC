#!/usr/bin/env bash
set -Eeuo pipefail
need(){ command -v "$1" >/dev/null || { echo "Falta $1"; exit 1; }; }
need yq; need awk; need mini-accum-backtest
CFG="${1:-configs/mini_accum/presets/CORE_2025.yaml}"
yq -e '.' "$CFG" >/dev/null
H4=$(yq -r '.data.ohlc_4h_csv' "$CFG"); D1=$(yq -r '.data.ohlc_d1_csv' "$CFG")
[[ -f "$H4" && -f "$D1" ]] || { echo "No encuentro OHLC: $H4 o $D1"; exit 1; }
echo "[OK] Preflight: $CFG"
