#!/usr/bin/env bash
set -euo pipefail

mkdir -p reports/w34_neighbors/raw
# Freezes a validar (ajusta si quieres)
FREEZES=("2025-07-01 00:00" "2025-08-01 00:00")

# Backtest base
PERIOD=1460d
HORIZONS=30,60,90
MAXBARS=975
BASE_SLIP=0.0002;  BASE_COST=0.0004
STRESS_SLIP=0.0003; STRESS_COST=0.0005

run_one() { 
  T="$1"; SL="$2"; TP1="$3"; P="$4"; F="$5"; KIND="$6"; SLIP="$7"; COST="$8"
  TAG="$(echo "$F" | tr ':- ' '_')"
  LABEL="T${T}_SL${SL}_TP1${TP1}_P${P}"
  OUT="reports/w34_neighbors/raw/${TAG}_${LABEL}_${KIND}.csv"
  echo ">> $KIND | freeze=$F | $LABEL -> $OUT"
  python swing_4h_forward_diamond.py --skip_yf \
    --symbol BTC-USD --period "$PERIOD" --horizons "$HORIZONS" \
    --freeze_end "$F" --max_bars "$MAXBARS" \
    --threshold "$T" --sl_atr_mul "$SL" --tp1_atr_mul "$TP1" --partial_pct "$P" \
    --slip "$SLIP" --cost "$COST" \
    --out_csv "$OUT"
}

# Vecinos a probar (uno por línea):  T  SL    TP1   P
COMBOS=(
  "0.60 1.25 0.65 0.80"
  "0.58 1.25 0.70 0.80"
)

for F in "${FREEZES[@]}"; do
  for C in "${COMBOS[@]}"; do
    read -r T SL TP1 P <<<"$C"
    run_one "$T" "$SL" "$TP1" "$P" "$F" base   "$BASE_SLIP"   "$BASE_COST"
    run_one "$T" "$SL" "$TP1" "$P" "$F" stress "$STRESS_SLIP" "$STRESS_COST"
  done
done

echo "OK -> reports/w34_neighbors/raw"
