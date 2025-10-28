#!/bin/zsh
# bb_dailyreport.zsh — Genera evidence/dayN_YYYY-MM-DD/REPORT.md
# - Cuenta canarios DRYRUN=1 (total/GREEN/PAUSE/YELLOW) filtrando por fecha UTC
# - Detecta ATTEST OK del bloque correcto en logs/cron.log
# - Lista los últimos 12 canarios válidos (start/ready/done + veredicto)
# Seguro e idempotente. No modifica lógica ni envía órdenes.

set -euo pipefail
emulate -L zsh
setopt NULL_GLOB

# --- Raíz del repo (2 niveles arriba de este script) ---
SCRIPT_DIR="${0:A:h}"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Día objetivo (YYYY-MM-DD) ---
DAY="${1:-$(date -u +%F)}"
DSTR="${DAY//-/}"

OUTDIR="$ROOT/evidence/dayN_${DAY}"
OUT="$OUTDIR/REPORT.md"
mkdir -p "$OUTDIR"

# --- ATTEST OK del día (0/1) ---
compute_attest_ok() {
  local LOG="$ROOT/logs/cron.log"
  local STATUS="$ROOT/health/mini_accum.status"
  local DAY_TGT="$DAY"
  local ok=0 last line iso day

  # 1) Prioriza STATUS (mtime = DAY_TGT y health ok en JSON/YAML/key=value)
  if [[ -f "$STATUS" ]] && [[ "$(date -u -r "$STATUS" +%F 2>/dev/null)" == "$DAY_TGT" ]]; then
    if grep -qiE '"health"[[:space:]]*:[[:space:]]*"ok"' "$STATUS" \
       || grep -qiE '(^|[[:space:]])health=ok([[:space:]]|$)' "$STATUS" \
       || grep -qiE '^health:[[:space:]]*"?ok"?' "$STATUS"; then
      ok=1
    fi
  fi

  # 2) Si aún no, revisa el último write_status del cron.log (con ISO o "puro")
  if (( ok == 0 )) && [[ -f "$LOG" ]]; then
    last="$(grep -nE '(^\[OK\] write_status:)|(^[0-9]{4}-[0-9]{2}-[0-9]{2}T.*\[OK\] write_status:)' "$LOG" | tail -1)"
    if [[ -n "$last" ]]; then
      line="${last#*:}"
      if [[ "$line" == \[* ]]; then
        # Sin fecha → usa mtime del LOG como último recurso del día objetivo
        [[ "$(date -u -r "$LOG" +%F 2>/dev/null)" == "$DAY_TGT" ]] && ok=1
      else
        iso="${line%%[[:space:]]*}"   # 2025-10-28T06:07:00Z
        day="${iso%%T*}"
        [[ "$day" == "$DAY_TGT" ]] && ok=1
      fi
    fi
  fi

  print -r -- "$ok"
}
ATTEST_OK="$(compute_attest_ok)"

# --- Recolecta logs de canario del día (logs/ y evidence/day*/ del mismo día) ---
typeset -a FILES
FILES=(
  "$ROOT/logs/canary_live.${DSTR}T"*.log(N)
  "$ROOT"/evidence/day*_"$DAY"/canary_live."${DSTR}T"*.log(N)
)

# --- Orden cronológico (por nombre) ---
if (( ${#FILES} )); then
  IFS=$'\n' FILES=($(printf "%s\n" "${FILES[@]}" | sort))
  unset IFS
fi

# --- Contadores ---
typeset -i TOTAL=0 GREEN=0 PAUSE=0 YELLOW=0
typeset -a DETAILS  # cada elemento = bloque markdown por archivo válido

# --- Clasificación por archivo ---
for f in "${FILES[@]}"; do
  # válido = DRYRUN=1 (sombra binance) + línea start
  if ! grep -qE 'start EXCHANGE=.*DRYRUN=1' "$f"; then
    continue
  fi

  # Extrae primeras apariciones
  s="$(grep -m1 'start EXCHANGE' "$f" | sed 's/.*start /start /')"
  r="$(grep -m1 'ready (signal' "$f" || true)"
  d="$(grep -m1 '\[INFO\] canary_live: done' "$f" || true)"

  verdict="YELLOW"
  if grep -q '\[PAUSE\]' "$f"; then
    verdict="PAUSE"
    ((PAUSE++))
  elif [[ -n "$r" && -n "$d" ]]; then
    verdict="GREEN"
    ((GREEN++))
  else
    verdict="YELLOW"
    ((YELLOW++))
  fi

  ((TOTAL++))

  DETAILS+=(
"# \`$(basename "$f")\`
- ${s:-(sin start)}
${r:+- $r}
${d:+- $d}
- → **${verdict}**"
  )
done

# --- Render del reporte ---
{
  echo "# Canary DRYRUN — Resumen diario (UTC: ${DAY})"
  echo
  echo "- ATTEST OK: ${ATTEST_OK}"
  echo "- Canarios válidos: ${TOTAL}  | GREEN=${GREEN}  | PAUSE=${PAUSE}${TOTAL:+  | YELLOW=${YELLOW}}"
  echo
  echo "## Últimos 12 válidos"
  if (( TOTAL == 0 )); then
    echo
    echo "- (sin válidos hoy)"
  else
    echo
    # imprime los últimos 12 DETAILS
    typeset -i n=${#DETAILS}
    typeset -i start=1
    if (( n > 12 )); then
      start=$(( n - 12 + 1 ))
    fi
    for (( i=start; i<=n; i++ )); do
      echo "${DETAILS[i]}"
      echo
    done
  fi
} > "$OUT"

echo "[OK] bb_dailyreport: $OUT"