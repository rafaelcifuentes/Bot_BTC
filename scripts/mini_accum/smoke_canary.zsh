#!/usr/bin/env zsh
set -euo pipefail
setopt null_glob extended_glob

ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
cd "$ROOT" || exit 1

STREAK_WINDOW_DAYS=${STREAK_WINDOW_DAYS:-7}

echo "== KPI_GUARD =="
tail -n1 health/mini_accum.status 2>/dev/null || print -r -- "[WARN] sin status"

echo "== CANARY: último log =="

collect() {
  local dir="$1"
  [[ -n "$dir" && -d "$dir" ]] || return 0
  find "$dir" -type f -name 'canary_live.*.log' -print0 2>/dev/null \
  | while IFS= read -r -d '' f; do
      local m; m=$(stat -f %m "$f" 2>/dev/null || echo 0)
      printf "%s\t%s\n" "$m" "$f"
    done
}

# Recolecta TODO para 2 vistas: (A) lista de 12; (B) streak últimos N días
set +o pipefail
all=$(
  { collect "$ROOT/logs"
    collect "$ROOT/evidence"
    collect "$ROOT/evidence"/dayN_* 2>/dev/null || true
  } | sort -nr
)
set -o pipefail

# (A) Últimos 12 por mtime
LAST=(${(f)"$(print -r -- "$all" | awk -F'\t' '{print $2}' | head -n12)"})

if (( ${#LAST} == 0 )); then
  print -r -- "[WARN] no se encontraron logs de canario."
  exit 0
fi

latest="${LAST[1]}"
print -r -- "$latest"

grep -q 'ready (signal fresh)' "$latest"; ok_ready=$?
grep -Eq '\[PAPER] flip|flip: simulated' "$latest"; ok_flip=$?
grep -q 'done' "$latest"; ok_done=$?

if (( ok_ready==0 && ok_flip==0 && ok_done==0 )); then
  print -r -- "→ CANARY: GREEN"
else
  (( ok_ready != 0 )) && print -r -- "[MISS] ready (signal fresh)"
  (( ok_flip  != 0 )) && print -r -- "[MISS] \\[PAPER] flip"
  (( ok_done  != 0 )) && print -r -- "[MISS] done"
  print -r -- "→ CANARY: RED"
fi

echo
echo "== Últimos 12 (GREEN/RED) =="
for f in $LAST; do
  mark="RED"
  if grep -q 'ready (signal fresh)' "$f" \
     && grep -Eq '\[PAPER] flip|flip: simulated' "$f" \
     && grep -q 'done' "$f"; then
    mark="GREEN"
  fi
  printf -- "- \`%s\` → **%s**\n" "$(basename "$f")" "$mark"
done

# (B) Streak sobre TODOS los logs de los últimos N días (no limitado a 12)
echo
echo "== Streak (últimos ${STREAK_WINDOW_DAYS} días) =="

now_epoch=$(date -u +%s)
since_epoch=$(( now_epoch - STREAK_WINDOW_DAYS*24*3600 ))

# Filtra por mtime dentro de la ventana
within_window=(${(f)"$(print -r -- "$all" | awk -F'\t' -v s="$since_epoch" '$1 >= s {print $2}')"})
typeset -A day_green
for f in $within_window; do
  # Considera GREEN si pasan las 3 condiciones
  if grep -q 'ready (signal fresh)' "$f" \
     && grep -Eq '\[PAPER] flip|flip: simulated' "$f" \
     && grep -q 'done' "$f"; then
    # Fecha UTC por mtime (YYYY-MM-DD)
    m=$(stat -f %m "$f" 2>/dev/null || echo 0)
    d=$(date -u -r "$m" "+%F" 2>/dev/null || echo "")
    [[ -n "$d" ]] && day_green[$d]=1
  fi
done

green_days=${#day_green}
printf "→ %d/%d días GREEN\n" "$green_days" "$STREAK_WINDOW_DAYS"

exit 0

echo
echo "== STORM (24h) =="
scripts/mini_accum/check_storm.zsh || true
