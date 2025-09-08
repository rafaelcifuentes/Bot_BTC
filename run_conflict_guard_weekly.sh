#!/bin/zsh
set -euo pipefail

REPO="$HOME/PycharmProjects/Bot_BTC"
PY="$REPO/.venv/bin/python"
cd "$REPO"

mkdir -p logs

# Congelamos la corrida al martes actual a las 08:05 UTC
FREEZE_END=$(date -u +"%Y-%m-%d 08:05")

SEEDS="0.40,0.55;0.45,0.50;0.45,0.55;0.50,0.55;0.60,0.40"

$PY "$REPO/runner_conflict_guard_weekly_v2.py" \
  --freeze-end "$FREEZE_END" \
  --cap-mode "min_leg*1.05" \
  --net-tol 0.20 \
  --seed-weights "$SEEDS" \
  >> "$REPO/logs/conflict_guard_$(date -u +%Y%m%d_%H%M%S).log" 2>&1