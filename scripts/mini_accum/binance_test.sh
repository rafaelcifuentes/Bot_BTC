#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
mkdir -p "$ROOT/logs"

# Guard de VPN (omite y notifica si no hay túnel)
"$ROOT/scripts/mini_accum/vpn_guard.sh" || exit 0

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUNLOG="$ROOT/logs/binance_test.$TS.log"
PIDFILE="$ROOT/logs/binance_test.$TS.pid"
echo "Log: $RUNLOG"

# Lanza en background con caffeinate (1h)
nohup /usr/bin/env caffeinate -dims -t 3600 \
  env -i HOME="$HOME" PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin" SHELL="/bin/bash" \
  ROOT="$ROOT" LOG_LEVEL=INFO EXCHANGE="binance" MODE="paper" \
  /bin/bash "$ROOT/weekly_runner.sh" \
  > "$RUNLOG" 2>&1 &

echo $! > "$PIDFILE"
disown

# Notificación
LEVEL=INFO CHAN=mini_accum /usr/bin/env python3 "$ROOT/scripts/mini_accum/notify.py" "Binance test lanzado: $(basename "$RUNLOG")"
echo "PID: $(cat "$PIDFILE")"
