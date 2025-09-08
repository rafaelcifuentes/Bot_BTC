cat > mini_grid_diamante_v3.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

WIN_Q4="2023Q4:2023-10-01:2023-12-31"
COMMON_ARGS="--windows $WIN_Q4 --assets BTC-USD \
  --signals_root reports/windows_fixed --ohlc_root data/ohlc/1m \
  --fee_bps 6 --slip_bps 6 --partial 50_50 --breakeven_after_tp1 \
  --risk_total_pct 0.75 --weights BTC-USD=1.0 \
  --gate_pf 1.6 --gate_wr 0.60 --gate_trades 30 \
  --out_csv reports/kpis_grid.csv --out_top reports/top_configs.csv"

log_crosses () {
  local tag="$1"
  echo "=== Cruces [tzsafe] detectados ($tag) ==="
  grep "\[tzsafe\]" "$2" || true
}

run_block () {
  local TAG="$1"
  shift
  echo
  echo "==================== $TAG ===================="
  set -x
  "$@" 2>&1 | tee "logs/${TAG}_$(date +%F_%H%M).log"
  set +x
  python scripts/quick_kpi_diagnosis.py || true
  log_crosses "$TAG" "logs/${TAG}_$(date +%F_%H%M).log" || true
}

mkdir -p logs

# T1: soft, más reactivo (band 2%, rearm 6, hys 0.05)
export TREND_FILTER=1 TREND_MODE=soft TREND_FREQ=2h TREND_SPAN=24 TREND_BAND_PCT=0.02
export REARM_MIN=6 HYSTERESIS_PP=0.05
run_block "diamante_T1_soft_refine" \
  python scripts/run_grid_tzsafe.py $COMMON_ARGS --horizons 60 90 --thresholds 0.76 0.78 0.80

# T2: soft, un poco más conservador (band 2.5%, rearm 8, hys 0.06)
export TREND_FILTER=1 TREND_MODE=soft TREND_FREQ=2h TREND_SPAN=24 TREND_BAND_PCT=0.025
export REARM_MIN=8 HYSTERESIS_PP=0.06
run_block "diamante_T2_soft_refine_plus" \
  python scripts/run_grid_tzsafe.py $COMMON_ARGS --horizons 60 90 --thresholds 0.76 0.78 0.80

# T3: soft, más lento (3h/36), para intentar suprimir ruido de Q4
export TREND_FILTER=1 TREND_MODE=soft TREND_FREQ=3h TREND_SPAN=36 TREND_BAND_PCT=0.03
export REARM_MIN=8 HYSTERESIS_PP=0.06
run_block "diamante_T3_soft_slow" \
  python scripts/run_grid_tzsafe.py $COMMON_ARGS --horizons 90 120 --thresholds 0.74 0.76 0.78

echo "================================================"
echo "FIN Mini Grid Diamante v3 (Q4). Revisa reports/kpis_grid.csv"
echo "Siguiente paso sugerido:"
echo " - Si un bloque da Trades>=30 y WR=0.58–0.59 (near-miss), relanza ese MISMO bloque"
echo "   sumando +0.02 al threshold y/o +2 al REARM_MIN (poda suave)."
echo " - Luego valida el ganador en 2022H2 y 2024H1 (script validate_winners.sh)."
echo "================================================"
BASH

chmod +x mini_grid_diamante_v3.sh
./mini_grid_diamante_v3.sh