#!/usr/bin/env bash
set -euo pipefail

# ==========
# ENV de Día 3 (micro-grid)
# ==========
# Señal de tendencia (igual para todos los folds, walk-forward)
export TREND_FILTER="${TREND_FILTER:-1}"
export TREND_MODE="${TREND_MODE:-soft}"
export TREND_FREQ="${TREND_FREQ:-2h}"
export TREND_SPAN="${TREND_SPAN:-12}"
export TREND_BAND_PCT="${TREND_BAND_PCT:-0.01}"
export REARM_MIN="${REARM_MIN:-4}"
export HYSTERESIS_PP="${HYSTERESIS_PP:-0.04}"

# Rutas
export SIGNALS_ROOT="${SIGNALS_ROOT:-reports/windows_fixed}"
export OHLC_ROOT="${OHLC_ROOT:-data/ohlc/1m}"

# Backtest por defecto (puedes cambiarlos luego al invocar)
export FEE_BPS="${FEE_BPS:-6}"
export SLIP_BPS="${SLIP_BPS:-6}"
export PARTIAL="${PARTIAL:-50_50}"          # opciones: 50_50 | none
export BREAKEVEN_AFTER_TP1="${BREAKEVEN_AFTER_TP1:-1}"
export RISK_TOTAL_PCT="${RISK_TOTAL_PCT:-0.75}"
export WEIGHTS="${WEIGHTS:-BTC-USD=1.0}"
export GATE_PF="${GATE_PF:-1.6}"
export GATE_WR="${GATE_WR:-0.60}"
export GATE_TRADES="${GATE_TRADES:-30}"

# Asegura que Python vea / y /scripts
export PYTHONPATH="$(pwd):$(pwd)/scripts:${PYTHONPATH:-}"

# Resumen bonito
echo "[env_day3] TF=$TREND_FILTER MODE=$TREND_MODE FREQ=$TREND_FREQ SPAN=$TREND_SPAN BAND=$TREND_BAND_PCT RM=$REARM_MIN HYS=$HYSTERESIS_PP"
echo "[env_day3] SIGNALS_ROOT=$SIGNALS_ROOT  OHLC_ROOT=$OHLC_ROOT"
echo "[env_day3] Fees=$FEE_BPS  Slip=$SLIP_BPS  Partial=$PARTIAL  Breakeven=$BREAKEVEN_AFTER_TP1"
echo "[env_day3] Risk=$RISK_TOTAL_PCT  Weights=$WEIGHTS  Gates: PF>=$GATE_PF WR>=$GATE_WR Trades>=$GATE_TRADES"
