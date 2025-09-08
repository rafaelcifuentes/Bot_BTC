#!/usr/bin/env bash
# env_day3.sh
# ENV reproducible para Día 3 (mismo ENV en todos los folds)

export TREND_FILTER=1
export TREND_MODE=soft
export TREND_FREQ=2h
export TREND_SPAN=12
export TREND_BAND_PCT=0.01

# Valores base Día 3
export REARM_MIN="${REARM_MIN:-4}"
export HYSTERESIS_PP=0.04

# Rutas (ajústalas si usas otras carpetas)
export SIGNALS_ROOT="reports/windows_fixed"
export OHLC_ROOT="data/ohlc/1m"

# Backtest parámetros comunes
export FEE_BPS=6
export SLIP_BPS=6
export PARTIAL="50_50"
export RISK_TOTAL_PCT=0.75
export WEIGHTS="BTC-USD=1.0"
export GATE_PF=1.6
export GATE_WR=0.60
export GATE_TRADES=30

# Para imports relativos (evita el ModuleNotFoundError de tzsafe_window)
export PYTHONPATH="$(pwd):$(pwd)/scripts"