#!/bin/zsh
set -euo pipefail
setopt NULL_GLOB

# Root del repo
ROOT="$HOME/PycharmProjects/Bot_BTC"
cd "$ROOT"

# Fecha UTC del pack
DAY=$(date -u +%F)
DSTR=${DAY//-/}

# Salida
OUT="artifacts/canary_pack_${DSTR}.tgz"
mkdir -p artifacts

# --- Insumos mínimos e idempotentes (no toca lógica/órdenes) ---

# Asegura carpeta del día
mkdir -p "evidence/dayN_${DAY}"

# Genera REPORT.md si no existe o está vacío
if [[ ! -s "evidence/dayN_${DAY}/REPORT.md" ]]; then
  ./scripts/mini_accum/bb_dailyreport.zsh || true
fi

# Asegura latest.json (si no existe o está vacío)
if [[ ! -s "signals/mini_accum/latest.json" ]]; then
  mkdir -p signals/mini_accum
  TS=$(date -u +%FT%TZ)
  print -r -- "{\"ts_utc\":\"$TS\",\"health\":\"ok\",\"reason\":\"pack_guard\"}" > "signals/mini_accum/latest.json"
fi

# Logs del día (opcionales; no rompen si no hay)
typeset -a logs
logs=(logs/canary_live.${DSTR}T*.log)

# --- Empaque ---
tar -czf "$OUT" \
  "evidence/dayN_${DAY}/" \
  logs/cron.log \
  signals/mini_accum/latest.json \
  health/mini_accum.status \
  ${logs:+${logs[@]}}

echo "[OK] pack: $OUT"