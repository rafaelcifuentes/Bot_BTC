#!/usr/bin/env bash
set -euo pipefail

# === Parámetros editables ===
SYMBOL="BTC-USD"
PERIOD="1460d"
HORIZONS="30,60,90"

# Costes base y stress
BASE_SLIP="0.0002"; BASE_COST="0.0004"
STRESS_SLIP="0.0003"; STRESS_COST="0.0005"

# Vecinos (A y B)
A_T="0.60"; A_SL="1.25"; A_TP1="0.65"; A_P="0.80"
B_T="0.58"; B_SL="1.25"; B_TP1="0.70"; B_P="0.80"

# Etiqueta de semana (puedes fijarla a w6, w7, etc.)
WTAG="${WTAG:-w$(date +"%V")}"

# === Salidas ===
OUT_BASE_A="reports/${WTAG}_oos_btc/a"
OUT_BASE_B="reports/${WTAG}_oos_btc/b"
OUT_STRS_A="reports/${WTAG}_stress_btc/a"
OUT_STRS_B="reports/${WTAG}_stress_btc/b"
OUT_SUMM="reports/${WTAG}_summary"
mkdir -p "$OUT_BASE_A" "$OUT_BASE_B" "$OUT_STRS_A" "$OUT_STRS_B" "$OUT_SUMM"

# === Últimos dos martes (00:00) ===
FREEZES=$(python - <<'PY'
from datetime import datetime,timedelta
now = datetime.now()
# martes = 1 (lunes=0)
offset = (now.weekday() - 1) % 7
last_tue = (now - timedelta(days=offset)).replace(hour=0,minute=0,second=0,microsecond=0)
prev_tue = last_tue - timedelta(days=7)
for d in [prev_tue, last_tue]:
    print(d.strftime("%Y-%m-%d 00:00"))
PY
)
echo "[freezes]"; echo "$FREEZES" | sed 's/^/ - /'

run_one () {
  local FREEZE="$1" T="$2" SL="$3" TP1="$4" P="$5" SLIP="$6" COST="$7" OUTDIR="$8"
  local TAG="$(echo "$FREEZE" | tr ' :-' '_' )"
  local OUT_CSV="${OUTDIR}/${TAG}.csv"
  local OUT_PLUS="${OUTDIR}/${TAG}_plus.csv"

  echo ">> T=$T SL=$SL TP1=$TP1 P=$P  freeze=$FREEZE -> $OUT_CSV"
  python swing_4h_forward_diamond.py --skip_yf \
    --symbol "$SYMBOL" --period "$PERIOD" --horizons "$HORIZONS" \
    --freeze_end "$FREEZE" --max_bars 975 \
    --threshold "$T" --sl_atr_mul "$SL" --tp1_atr_mul "$TP1" --partial_pct "$P" \
    --slip "$SLIP" --cost "$COST" --out_csv "$OUT_CSV"

  # En muchos flujos ya se emite el _plus; si no, simplemente lo ignoramos.
  # (No forzamos postproceso aquí para mantener el runner ligero)
}

# === BASE (A y B) ===
echo "[base] -> $OUT_BASE_A  /  $OUT_BASE_B"
while read -r FZ; do
  run_one "$FZ" "$A_T" "$A_SL" "$A_TP1" "$A_P" "$BASE_SLIP" "$BASE_COST" "$OUT_BASE_A"
  run_one "$FZ" "$B_T" "$B_SL" "$B_TP1" "$B_P" "$BASE_SLIP" "$BASE_COST" "$OUT_BASE_B"
done <<< "$FREEZES"

# === STRESS (A y B) ===
echo "[stress] -> $OUT_STRS_A  /  $OUT_STRS_B"
while read -r FZ; do
  run_one "$FZ" "$A_T" "$A_SL" "$A_TP1" "$A_P" "$STRESS_SLIP" "$STRESS_COST" "$OUT_STRS_A"
  run_one "$FZ" "$B_T" "$B_SL" "$B_TP1" "$B_P" "$STRESS_SLIP" "$STRESS_COST" "$OUT_STRS_B"
done <<< "$FREEZES"

# === Compactador (ver script Python más abajo) ===
python scripts/wk_compactor.py \
  --baseA "$OUT_BASE_A" --baseB "$OUT_BASE_B" \
  --stressA "$OUT_STRS_A" --stressB "$OUT_STRS_B" \
  --outdir "$OUT_SUMM"

echo "OK -> ${OUT_SUMM}/(compact_base_vs_stress.csv|decision.md)"