#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
mkdir -p "$ROOT/logs"

# Defaults KISS
EXCHANGE="${EXCHANGE:-binanceus}"
DRYRUN="${DRYRUN:-1}"
MAX_TRADE_USD="${MAX_TRADE_USD:-10}"
POS_CAP_PCT="${POS_CAP_PCT:-0.10}"
SYMBOL="${SYMBOL:-BTC/USDC}"
FRESHNESS_MAX_HOURS="${FRESHNESS_MAX_HOURS:-6}"
WATCHDOG_HOURS="${WATCHDOG_HOURS:-8}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUNLOG="$ROOT/logs/canary_live.$TS.log"
PIDFILE="$ROOT/logs/canary_live.$TS.pid"

echo "$TS [INFO] canary_live: start EXCHANGE=$EXCHANGE DRYRUN=$DRYRUN USD<=$MAX_TRADE_USD cap=$POS_CAP_PCT" | tee -a "$RUNLOG"
echo "$TS [INFO] canary_live: python=$ROOT/.venv/bin/python" | tee -a "$RUNLOG"

# Lanza wrapper con entorno limpio, pero inyectando TODO lo que necesitamos
nohup /usr/bin/env caffeinate -dims -t 1800 \
  /usr/bin/env -i \
    HOME="$HOME" PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin" SHELL="/bin/bash" \
    ROOT="$ROOT" LOG_LEVEL=INFO \
    EXCHANGE="$EXCHANGE" DRYRUN="$DRYRUN" \
    MAX_TRADE_USD="$MAX_TRADE_USD" POS_CAP_PCT="$POS_CAP_PCT" SYMBOL="$SYMBOL" \
    FRESHNESS_MAX_HOURS="$FRESHNESS_MAX_HOURS" WATCHDOG_HOURS="$WATCHDOG_HOURS" \
    "$ROOT/.venv/bin/python" "$ROOT/scripts/mini_accum/live_wrapper.py" \
    >> "$RUNLOG" 2>&1 &

echo $! > "$PIDFILE"
LEVEL=INFO CHAN=mini_accum /usr/bin/env python3 "$ROOT/scripts/mini_accum/notify.py" \
  "Canary LIVE lanzado (EXCHANGE=${EXCHANGE}, DRYRUN=${DRYRUN}, USD<=${MAX_TRADE_USD}, cap ${POS_CAP_PCT}); log=$(basename "$RUNLOG")"
