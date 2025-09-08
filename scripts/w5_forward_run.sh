#!/usr/bin/env bash
set -euo pipefail

# ==== 1) CONFIG EDITABLE ====
FREEZES=("2025-07-01 00:00" "2025-08-01 00:00")

# Vecino A (default robusto)
TA=0.60; SLA=1.25; TP1A=0.65; PA=0.80
# Vecino B (alfa, más sensible)
TB=0.58; SLB=1.25; TP1B=0.70; PB=0.80

# Costes base y stress
SLIP_BASE=0.0002; COST_BASE=0.0004
SLIP_ST=0.0003;  COST_ST=0.0005

# ==== 2) RUTAS ====
BASE_A="reports/w5_oos_btc/a"; BASE_B="reports/w5_oos_btc/b"
ST_A="reports/w5_stress_btc/a"; ST_B="reports/w5_stress_btc/b"
mkdir -p "$BASE_A" "$BASE_B" "$ST_A" "$ST_B" reports/w5_summary

run_one () {
  local FREEZE="$1" T="$2" SL="$3" TP1="$4" P="$5" OUT_DIR="$6" SLIP="$7" COST="$8"
  local TAG; TAG=$(echo "$FREEZE" | tr ':- ' '_')
  PYTHONPATH="$(pwd)" EXCHANGE=binanceus \
  python swing_4h_forward_diamond.py --skip_yf \
    --symbol BTC-USD --period 1460d --horizons 30,60,90 \
    --freeze_end "$FREEZE" --max_bars 975 \
    --threshold "$T" --sl_atr_mul "$SL" --tp1_atr_mul "$TP1" \
    --partial_pct "$P" --slip "$SLIP" --cost "$COST" \
    --out_csv "${OUT_DIR}/${TAG}.csv"
}

echo "[W5] Base (A/B) + Stress para: ${FREEZES[*]}"
for F in "${FREEZES[@]}"; do
  # Base
  run_one "$F" "$TA" "$SLA" "$TP1A" "$PA" "$BASE_A" "$SLIP_BASE" "$COST_BASE"
  run_one "$F" "$TB" "$SLB" "$TP1B" "$PB" "$BASE_B" "$SLIP_BASE" "$COST_BASE"
  # Stress
  run_one "$F" "$TA" "$SLA" "$TP1A" "$PA" "$ST_A"   "$SLIP_ST"  "$COST_ST"
  run_one "$F" "$TB" "$SLB" "$TP1B" "$PB" "$ST_B"   "$SLIP_ST"  "$COST_ST"
done

# ==== 3) Resumen compacto (base vs stress, por vecino) ====
python scripts/w5_compact_summary.py || true

echo "OK -> w5 base+stress + compacto"
