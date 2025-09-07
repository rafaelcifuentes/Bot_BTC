# scripts/corazon_cmds.zsh
# ---------------------------------------------
# Presets Corazón (Sentiment-EXP) + helpers
# ---------------------------------------------
# -*- mode: sh; -*-

# ============== Variables base ==============
export EXCHANGE=${EXCHANGE:-binanceus}
export FREEZE=${FREEZE:-"2025-08-05 00:00"}
export FG=${FG:-"./data/sentiment/fear_greed.csv"}
export FU=${FU:-"./data/sentiment/funding_rates.csv"}
export OUT=${OUT:-"reports"}
export SYMBOL=${SYMBOL:-"BTC/USD"}

mkdir -p "$OUT"

# ============== Helpers UI ==============
corazon_help() {
  cat <<'HLP'
Comandos disponibles:
  runC_status
  runC_all_freeze
  runC_all_live
  runC_long_freeze
  runC_long_freeze_v2
  runC_long_live_v2
  runC_sweep_freeze
  runC_sweep_long_v2
  runC_sweep_long_v2_22
  runC_sweep_long_v2_planA
  runC_sweep_long_v2_wide
  runC_auto_daily
  runC_auto_status
HLP
}

runC_status() {
  echo "EXCHANGE=$EXCHANGE"
  echo "FREEZE=$FREEZE"
  echo "FG=$FG"
  echo "FU=$FU"
  echo "OUT=$OUT"
  echo "SYMBOL=$SYMBOL"
  [[ -f "$FG" ]] && { echo "--- head FG ---"; head -n 3 "$FG"; }
  [[ -f "$FU" ]] && { echo "--- head FU ---"; head -n 3 "$FU"; }
}

# ============== 1) ALL (sin gates) ==============
runC_all_freeze() {
  EXCHANGE=$EXCHANGE python runner_corazon.py \
    --freeze_end "$FREEZE" --max_bars 975 \
    --fg_csv "$FG" --funding_csv "$FU" \
    --no_gates \
    --threshold 0.60 \
    --out_csv "$OUT/corazon_metrics.csv"
  echo "CSV -> $OUT/corazon_metrics.csv"
}

runC_all_live() {
  EXCHANGE=$EXCHANGE python runner_corazon.py \
    --max_bars 975 \
    --fg_csv "$FG" --funding_csv "$FU" \
    --no_gates \
    --threshold 0.60 \
    --out_csv "$OUT/corazon_metrics_live.csv"
  echo "CSV -> $OUT/corazon_metrics_live.csv"
}

# ============== 2) LONG bias (gates) ==============
# Versión base (suave)
runC_long_freeze() {
  EXCHANGE=$EXCHANGE python runner_corazon.py \
    --freeze_end "$FREEZE" --max_bars 975 \
    --fg_csv "$FG" --funding_csv "$FU" \
    --fg_long_min -0.15 --fg_short_max 0.15 \
    --funding_bias 0.005 \
    --adx1d_len 14 --adx1d_min 28 --adx4_min 12 \
    --threshold 0.60 \
    --out_csv "$OUT/corazon_long_metrics_tuned.csv"
  echo "CSV -> $OUT/corazon_long_metrics_tuned.csv"
}

# Variante V2 (adx más bajo 22)
runC_long_freeze_v2() {
  EXCHANGE=$EXCHANGE python runner_corazon.py \
    --freeze_end "$FREEZE" --max_bars 975 \
    --fg_csv "$FG" --funding_csv "$FU" \
    --fg_long_min -0.15 --fg_short_max 0.15 \
    --funding_bias 0.005 \
    --adx1d_len 14 --adx1d_min 22 --adx4_min 12 \
    --threshold 0.60 \
    --out_csv "$OUT/corazon_long_metrics_v2.csv"
  echo "CSV -> $OUT/corazon_long_metrics_v2.csv"
}

runC_long_live_v2() {
  EXCHANGE=$EXCHANGE python runner_corazon.py \
    --max_bars 975 \
    --fg_csv "$FG" --funding_csv "$FU" \
    --fg_long_min -0.15 --fg_short_max 0.15 \
    --funding_bias 0.005 \
    --adx1d_len 14 --adx1d_min 22 --adx4_min 12 \
    --threshold 0.60 \
    --out_csv "$OUT/corazon_long_metrics_v2_live.csv"
  echo "CSV -> $OUT/corazon_long_metrics_v2_live.csv"
}

# ============== 3) Sweeps ==============
runC_sweep_freeze() {
  EXCHANGE=$EXCHANGE python runner_corazon.py \
    --freeze_end "$FREEZE" --max_bars 975 \
    --fg_csv "$FG" --funding_csv "$FU" \
    --no_gates \
    --sweep_threshold 0.56:0.64:0.01 \
    --out_csv "$OUT/corazon_metrics_sweep.csv"
  echo "CSV sweep -> $OUT/corazon_metrics_sweep.csv"
}

runC_sweep_long_v2() {
  EXCHANGE=$EXCHANGE python runner_corazon.py \
    --freeze_end "$FREEZE" --max_bars 975 \
    --fg_csv "$FG" --funding_csv "$FU" \
    --fg_long_min -0.15 --fg_short_max 0.15 \
    --funding_bias 0.005 \
    --adx1d_len 14 --adx1d_min 28 --adx4_min 12 \
    --sweep_threshold 0.57:0.62:0.01 \
    --out_csv "$OUT/corazon_long_v2_sweep.csv"
  echo "CSV sweep -> $OUT/corazon_long_v2_sweep.csv"
}

runC_sweep_long_v2_22() {
  EXCHANGE=$EXCHANGE python runner_corazon.py \
    --freeze_end "$FREEZE" --max_bars 975 \
    --fg_csv "$FG" --funding_csv "$FU" \
    --fg_long_min -0.15 --fg_short_max 0.15 \
    --funding_bias 0.005 \
    --adx1d_len 14 --adx1d_min 22 --adx4_min 12 \
    --sweep_threshold 0.57:0.62:0.01 \
    --out_csv "$OUT/corazon_long_v2_sweep.csv"
  echo "CSV sweep -> $OUT/corazon_long_v2_sweep.csv"
}

runC_sweep_long_v2_planA() {
  EXCHANGE=$EXCHANGE python runner_corazon.py \
    --freeze_end "$FREEZE" --max_bars 975 \
    --fg_csv "$FG" --funding_csv "$FU" \
    --fg_long_min -0.15 --fg_short_max 0.15 \
    --funding_bias 0.005 \
    --adx1d_len 14 --adx1d_min 22 --adx4_min 10 \
    --sweep_threshold 0.57:0.62:0.01 \
    --out_csv "$OUT/corazon_long_v2_planA_sweep.csv"
  echo "CSV sweep -> $OUT/corazon_long_v2_planA_sweep.csv"
}

runC_sweep_long_v2_wide() {
  EXCHANGE=$EXCHANGE python runner_corazon.py \
    --freeze_end "$FREEZE" --max_bars 975 \
    --fg_csv "$FG" --funding_csv "$FU" \
    --fg_long_min -0.15 --fg_short_max 0.15 \
    --funding_bias 0.005 \
    --adx1d_len 14 --adx1d_min 22 --adx4_min 12 \
    --sweep_threshold 0.55:0.62:0.01 \
    --out_csv "$OUT/corazon_long_v2_wide_sweep.csv"
  echo "CSV sweep -> $OUT/corazon_long_v2_wide_sweep.csv"
}

# ============== 4) Auto diario (compara y hace append) ==============
runC_auto_daily() {
  python corazon_auto.py \
    --exchange "$EXCHANGE" \
    --symbol "$SYMBOL" \
    --fg_csv "$FG" \
    --funding_csv "$FU" \
    --max_bars 975 \
    --freeze_end "$FREEZE" \
    --threshold 0.60 \
    --compare_both \
    --adx_min 22 \
    --report_csv "$OUT/corazon_auto_daily.csv"
}

runC_auto_status() {
  echo "----------------------------------------------"
  echo "Auto daily last rows:"
  tail -n 10 "${OUT}/corazon_auto_daily.csv" 2>/dev/null || echo "No hay reporte aún."
  echo "----------------------------------------------"
}

echo "✔️  corazon_cmds.zsh cargado. Usa:"
echo "   runC_status | runC_all_freeze | runC_all_live | runC_long_freeze | runC_long_freeze_v2 | runC_long_live_v2 | runC_sweep_freeze | runC_sweep_long_v2"