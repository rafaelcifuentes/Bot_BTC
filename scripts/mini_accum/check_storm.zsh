#!/usr/bin/env zsh
set -euo pipefail

ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
LOGDIR="$ROOT/evidence/dayN_$(date -u +%F)"
pattern='canary_live.*\.log$'

# Fallback: si no hay dayN de hoy, mirar logs globales
if [[ ! -d "$LOGDIR" ]]; then
  LOGDIR="$ROOT/logs"
fi

# Recolecta timestamps de los últimos 24h
now=$(date -u +%s)
dayago=$(( now - 24*3600 ))

typeset -A per_hour
per_hour=()

find "$LOGDIR" -type f -name 'canary_live.*.log' -print0 2>/dev/null \
| xargs -0 stat -f '%m %N' 2>/dev/null \
| awk -v dayago="$dayago" '$1 >= dayago {print}' \
| while read -r m path; do
    # bucket por hora UTC
    hr=$(date -ur "$m" +'%Y-%m-%dT%H')
    per_hour[$hr]=$(( ${per_hour[$hr]:-0} + 1 ))
  done

fail=0
for hr in ${(ko)per_hour}; do
  if (( per_hour[$hr] > 1 )); then
    print -r -- "[STORM] $hrZ: ${per_hour[$hr]} ejecuciones (>1)"
    fail=1
  fi
done

if (( fail == 0 )); then
  print -r -- "[OK] Storm guard (24h): ≤1 ejecución/hora"
else
  print -r -- "[FAIL] Storm guard: se detectaron ráfagas (>1/h)."
  exit 1
fi
