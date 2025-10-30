#!/bin/zsh
set -euo pipefail

# Rutas
ROOT=${ROOT:-$HOME/PycharmProjects/Bot_BTC}
PY="$ROOT/.venv/bin/python"; [[ -x "$PY" ]] || PY="$(command -v python3 || command -v python)"

# Parámetros Pilot Live (ultra KISS)
: ${EXCHANGE:=binance}
: ${DRYRUN:=0}
: ${DO_TRADE:=0}
: ${USD:=10}
: ${USD_MAX:=${USD}}
: ${CAP:=0.10}
: ${LOG_LEVEL:=INFO}

# Guardias de seguridad
mkdir -p "$ROOT/logs"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ROOT/logs/canary_live.${TS}.log"

if [[ "$EXCHANGE" != "binance" ]]; then
  echo "[ABORT] EXCHANGE must be 'binance' (global). Got: $EXCHANGE" | tee -a "$LOG"
  exit 2
fi
if [[ "$DRYRUN" != "0" ]]; then
  echo "[ABORT] DRYRUN=0 requerido para Pilot Live" | tee -a "$LOG"
  exit 2
fi
if [[ "$DO_TRADE" != "1" ]]; then
  echo "[ABORT] DO_TRADE=1 requerido para Pilot Live" | tee -a "$LOG"
  exit 2
fi

EP="/Users/rafaelcifuentes/PycharmProjects/Bot_BTC/scripts/mini_accum/live_wrapper.py"  # (inyectado al crear el archivo)

{
  echo "${TS} [INFO] canary_live: start EXCHANGE=$EXCHANGE DRYRUN=$DRYRUN USD_MAX<=${USD_MAX} cap=${CAP}"
  echo "${TS} [INFO] canary_live: python=$PY"
  "$PY" - <<'PY'
import sys, platform
print(platform.python_version())
print("sys.executable:", sys.executable)
PY
  echo "${TS} INFO mini_accum: LOG_LEVEL=${LOG_LEVEL} aplicado"
} | tee -a "$LOG"

# Export envs para el entrypoint Python (sin tocar lógica)
export EXCHANGE DRYRUN DO_TRADE USD USD_MAX CAP LOG_LEVEL SIDE

# Ejecuta el entrypoint
"$PY" "$EP" 2>&1 | tee -a "$LOG" || { rc=$?; echo "${TS} [WARN] canary_live: exit=$rc" | tee -a "$LOG"; exit $rc; }

echo "${TS} [INFO] canary_live: done" | tee -a "$LOG"
