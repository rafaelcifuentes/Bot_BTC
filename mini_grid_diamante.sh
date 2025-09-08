#!/usr/bin/env bash
set -euo pipefail

# === Paths y constantes ===
ASSET="BTC-USD"
WIN_Q4="2023Q4:2023-10-01:2023-12-31"
SIG_ROOT="reports/windows_fixed"
OHLC_ROOT="data/ohlc/1m"
OUT_CSV="reports/kpis_grid.csv"
OUT_TOP="reports/top_configs.csv"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

ts="$(date +%F_%H%M%S)"
LOG="$LOG_DIR/diamante_suite_${ts}.log"

# Helper: corre una tanda con env vars y una lista de thresholds
run_block () {
  local mode="$1" freq="$2" span="$3" band="$4" rearm="$5" hys="$6" ths="$7" hors="$8" tag="$9"
  echo "===== RUN $tag | MODE=$mode FREQ=$freq SPAN=$span BAND=$band REARM=$rearm HYS=$hys THS=[$ths] HORS=[$hors] =====" | tee -a "$LOG"

  TREND_FILTER=1 TREND_MODE="$mode" TREND_FREQ="$freq" TREND_SPAN="$span" TREND_BAND_PCT="$band" \
  REARM_MIN="$rearm" HYSTERESIS_PP="$hys" \
  python scripts/run_grid_tzsafe.py \
    --windows "$WIN_Q4" \
    --assets "$ASSET" \
    --horizons $hors \
    --thresholds $ths \
    --signals_root "$SIG_ROOT" \
    --ohlc_root "$OHLC_ROOT" \
    --fee_bps 6 --slip_bps 6 \
    --partial 50_50 --breakeven_after_tp1 \
    --risk_total_pct 0.75 \
    --weights "$ASSET=1.0" \
    --gate_pf 1.6 --gate_wr 0.60 --gate_trades 30 \
    --out_csv "$OUT_CSV" --out_top "$OUT_TOP" \
    2>&1 | tee -a "$LOG"

  echo "" | tee -a "$LOG"
}

# === Bloques “Diamante” enfocados en subir WR/PF sin matar Trades ===
# A) Filtro más lento (soft) para limpiar ruido; thresholds medios-altos
run_block soft 4h 48 0.02 8 0.03 "0.72 0.74 0.76" "60 90" "A_soft_lento"

# B) price_only con banda más exigente, umbrales altos (reduce falsos positivos)
run_block price_only 4h 72 0.03 8 0.03 "0.74 0.76 0.78" "60 90" "B_price_only_estricto"

# C) slope_only con mayor rearm para evitar re-entradas en chop
run_block slope_only 4h 48 0.00 12 0.02 "0.72 0.74 0.76" "60 90" "C_slope_antichop"

# === Diagnóstico resumido y cruces tzsafe ===
python scripts/quick_kpi_diagnosis.py | tee -a "$LOG"
echo -e "\n=== Cruces [tzsafe] detectados ===" | tee -a "$LOG"
grep "\[tzsafe\]" "$LOG" | tail -n 120 || true

echo -e "\nListo. Log guardado en: $LOG"