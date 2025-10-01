#!/usr/bin/env bash
set -euo pipefail
BASE="$1"  # CSV por-barra baseline (e.g., reports/diamante_btc_costes_freeze_2025-09-15_bars.csv)
OUT="reports/heart/diamante_overlay_$(basename "$BASE")"
WEI="reports/heart/w_diamante.csv"

mkdir -p reports/heart
python scripts/apply_heart_overlay.py \
  --weights_csv "$WEI" \
  --out_csv "$OUT" \
  "$BASE"

echo "[OK] Overlay -> $OUT"
