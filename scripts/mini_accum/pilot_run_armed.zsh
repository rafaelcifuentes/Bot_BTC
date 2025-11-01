#!/usr/bin/env zsh
set -euo pipefail
ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
cd "$ROOT"

# Modo ARMED: DO_TRADE=1 + DRYRUN=0 pero SIN enviar (solo imprime el “crear orden real aquí”)
export DO_TRADE=1
export DRYRUN=0
export ORDER_MODE=ARMED         # <- clave: no envía
export USD_MAX=10
export CAP=0.10
export EXCHANGE=binance
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

ts=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ROOT/logs/pilot_armed.$ts.log"

{
  echo "$(date -u +%FT%TZ) [INFO] pilot_armed: start EXCHANGE=$EXCHANGE DO_TRADE=$DO_TRADE DRYRUN=$DRYRUN ORDER_MODE=$ORDER_MODE USD<=$USD_MAX cap=$CAP"
  # Usa TU MISMO entrypoint del canario horario, sin cambiarlo (solo respetando ORDER_MODE=ARMED).
  # Ejemplo: si hoy llamas a tu ejecutor así en el canario:
  # python -m mini_accum.live_exec   (o tu wrapper actual)
  # entonces reutilízalo aquí:
  python -m mini_accum.live_exec || true

  echo "$(date -u +%FT%TZ) [INFO] pilot_armed: done"
} | tee "$LOG"

# Evidencia mínima
mkdir -p "$ROOT/evidence/pilot_armed"
ln -sf "$LOG" "$ROOT/evidence/pilot_armed/last.log"