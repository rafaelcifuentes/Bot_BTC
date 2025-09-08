#!/usr/bin/env bash
set -euo pipefail

# === Ajusta si tu ruta cambia ===
PROJ_DIR="$HOME/PycharmProjects/Bot_BTC"
VENV_PY="$PROJ_DIR/.venv/bin/python"
SCRIPT="runner_conflict_guard_weekly_v2.py"
LOG_DIR="$PROJ_DIR/logs"

# Semillas por defecto (puedes cambiarlas aquí)
SEED_WEIGHTS="0.55,0.45;0.60,0.40;0.65,0.35;0.50,0.50;0.45,0.55"
CAP_MODE="min_leg*1.05"
NET_TOL="0.20"

mkdir -p "$LOG_DIR"
cd "$PROJ_DIR"

# Hora actual en UTC
UTC_DOW="$(date -u '+%u')"     # 2 = martes
UTC_HH="$(date -u '+%H')"      # 08
UTC_MM="$(date -u '+%M')"      # 05

# Permite forzar ejecución manual con RUN_ANYWAY=1
if [[ "${RUN_ANYWAY:-0}" != "1" ]]; then
  if [[ "$UTC_DOW" != "2" || "$UTC_HH" != "08" || "$UTC_MM" != "05" ]]; then
    exit 0
  fi
fi

STAMP="$(date -u '+%Y%m%d_%H%M%S')"
FREEZE_END="$(date -u '+%Y-%m-%d %H:%M')"   # congela exactamente a la hora UTC actual
LOG_FILE="$LOG_DIR/conflict_guard_${STAMP}.log"

# Ejecuta el runner
"$VENV_PY" "$PROJ_DIR/$SCRIPT" \
  --freeze-end "$FREEZE_END" \
  --cap-mode "$CAP_MODE" \
  --net-tol "$NET_TOL" \
  --seed-weights "$SEED_WEIGHTS" \
  > "$LOG_FILE" 2>&1