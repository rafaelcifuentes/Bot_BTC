#!/usr/bin/env bash
set -euo pipefail

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%MUTC)}"
OUT="${OUT:-docs/mini_accum/checkpoints/$STAMP}"
mkdir -p "$OUT" logs

# 1) Intenta el preflight "real" con defaults (idempotente)
if [ -x scripts/mini_accum/preflight.sh ]; then
  scripts/mini_accum/preflight.sh || true
fi

# 2) Copia todos los CSVs de interés (sin globs frágiles)
copy_all_from_here() {
  find . -type f \( -name '*kpis*.csv' -o -name '*flips*.csv' \) -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do
      cp -f "$f" "$OUT/" || true
    done
}
copy_all_from_here

# 3) Si aún no hay CSVs, rescata los más recientes desde reports/**
copy_latest() {
  local pattern="$1" dest="$2"
  # macOS: stat -f "%m %N"; Linux fallback: stat -c '%Y %n'
  if stat -f %m / >/dev/null 2>&1; then
    find reports -type f -name "$pattern" -print0 2>/dev/null \
    | xargs -0 stat -f "%m %N" 2>/dev/null \
    | sort -nr | head -n1 | awk '{$1=""; sub(/^ /,""); print}' \
    | xargs -I{} cp -f "{}" "$dest" 2>/dev/null || true
  else
    find reports -type f -name "$pattern" -print0 2>/dev/null \
    | xargs -0 stat -c '%Y %n' 2>/dev/null \
    | sort -nr | head -n1 | awk '{$1=""; sub(/^ /,""); print}' \
    | xargs -I{} cp -f "{}" "$dest" 2>/dev/null || true
  fi
}

find "$OUT" -maxdepth 1 -type f -name '*kpis*.csv' >/dev/null 2>&1 || \
  copy_latest '*kpis*.csv'  "$OUT/kpis_latest.csv"
find "$OUT" -maxdepth 1 -type f -name '*flips*.csv' >/dev/null 2>&1 || \
  copy_latest '*flips*.csv' "$OUT/flips_latest.csv"

# 4) Contadores y commit idempotente
K=$(find "$OUT" -maxdepth 1 -type f -name '*kpis*.csv'  | wc -l | tr -d ' ')
F=$(find "$OUT" -maxdepth 1 -type f -name '*flips*.csv' | wc -l | tr -d ' ')
echo "[INFO] Guardados: KPIs=$K FLIPS=$F en $OUT"

if [ "$K" != "0" ] || [ "$F" != "0" ]; then
  git add "$OUT" || true
  git commit -m "preflight: KPIs baseline vs v2 @ $STAMP" || true
else
  echo "[WARN] No se encontraron CSVs de KPIs/FLIPS; revisa rutas de generación."
fi
