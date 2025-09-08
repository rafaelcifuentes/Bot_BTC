#!/usr/bin/env bash
set -euo pipefail

# Mini Grid Diamante v2 — foco 2023Q4 (BTC-USD)
# Objetivo: conseguir Trades≥30 en Q4 y empujar WR ≥ 0.60
# Salidas: logs/… y reports/kpis_grid.csv + reports/top_configs.csv

LOGDIR="logs"
mkdir -p "$LOGDIR"

ts() { date +%F_%H%M%S; }

COMMON_ARGS=(
  --windows "2023Q4:2023-10-01:2023-12-31"
  --assets BTC-USD
  --signals_root reports/windows_fixed
  --ohlc_root data/ohlc/1m
  --fee_bps 6 --slip_bps 6
  --partial 50_50 --breakeven_after_tp1
  --risk_total_pct 0.75
  --weights BTC-USD=1.0
  --gate_pf 1.6 --gate_wr 0.60 --gate_trades 30
  --out_csv reports/kpis_grid.csv
  --out_top reports/top_configs.csv
)

run_block () {
  local label="$1"; shift
  local tsnow; tsnow="$(ts)"
  echo
  echo "==================== $label ===================="
  echo "[env] TREND_FILTER=$TREND_FILTER TREND_MODE=$TREND_MODE TREND_FREQ=$TREND_FREQ TREND_SPAN=$TREND_SPAN TREND_BAND_PCT=$TREND_BAND_PCT REARM_MIN=$REARM_MIN HYSTERESIS_PP=$HYSTERESIS_PP"
  echo "[cmd] python scripts/run_grid_tzsafe.py $*"
  (
    set -x
    python scripts/run_grid_tzsafe.py "$@" 2>&1
    set +x
    python scripts/quick_kpi_diagnosis.py
  ) | tee "$LOGDIR/${label}_${tsnow}.log"
  echo "=== Cruces [tzsafe] detectados ($label) ==="
  grep "\[tzsafe\]" "$LOGDIR/${label}_${tsnow}.log" | tail -n 80 || true
  echo "Log guardado en: $LOGDIR/${label}_${tsnow}.log"
}

############################################
# Bloques (ajustados a lo que ya observaste)
############################################

# S1) soft — banda estrecha, rearm moderado (mantiene muchos cruces)
export TREND_FILTER=1 TREND_MODE=soft TREND_FREQ=2h TREND_SPAN=24 TREND_BAND_PCT=0.02
export REARM_MIN=4 HYSTERESIS_PP=0.04
run_block "diamante_S1_soft_fast" \
  "${COMMON_ARGS[@]}" \
  --horizons 60 90 \
  --thresholds 0.72 0.74

# S2) soft — tendencia más lenta + banda más amplia (depura entradas marginales; sube WR)
export TREND_FILTER=1 TREND_MODE=soft TREND_FREQ=4h TREND_SPAN=48 TREND_BAND_PCT=0.04
export REARM_MIN=8 HYSTERESIS_PP=0.05
run_block "diamante_S2_soft_slow" \
  "${COMMON_ARGS[@]}" \
  --horizons 90 120 \
  --thresholds 0.72 0.74 0.76

# P1) price_only — exige precio>EMA(1-BAND), suele mejorar WR sacrificando pocas Trades
export TREND_FILTER=1 TREND_MODE=price_only TREND_FREQ=2h TREND_SPAN=24 TREND_BAND_PCT=0.04
export REARM_MIN=8 HYSTERESIS_PP=0.05
run_block "diamante_P1_price" \
  "${COMMON_ARGS[@]}" \
  --horizons 90 120 \
  --thresholds 0.70 0.72 0.74

# SL1) slope_only — filtro por pendiente + horizonte largo (apuesta a calidad de rachas)
export TREND_FILTER=1 TREND_MODE=slope_only TREND_FREQ=2h TREND_SPAN=24 TREND_BAND_PCT=0.00
export REARM_MIN=12 HYSTERESIS_PP=0.04
run_block "diamante_SL1_slope" \
  "${COMMON_ARGS[@]}" \
  --horizons 120 \
  --thresholds 0.72 0.74

echo
echo "================================================"
echo "FIN Mini Grid Diamante v2 (Q4). Revisa reports/kpis_grid.csv"
echo "Siguiente paso sugerido:"
echo " - Si algún bloque da Trades>=30 con WR=0.58–0.59 (near-miss), relanza ese MISMO bloque"
echo "   sumando +0.02 al threshold y/o +2 al REARM_MIN para una 'poda suave'."
echo " - Luego valida el ganador en las 3 ventanas (2022H2, 2023Q4, 2024H1)."
echo "================================================"