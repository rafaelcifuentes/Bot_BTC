#!/usr/bin/env zsh
set -euo pipefail

ROOT="${SANDBOX:-$HOME/PycharmProjects/Bot_BTC}"
STATUS="$ROOT/health/mini_accum.status"

echo "== KPI_GUARD =="
if [[ -f "$STATUS" ]]; then tail -n1 "$STATUS"; else echo "[ERR] no existe: $STATUS"; fi

echo "== CANARY: último log =="
# Hallar el último canary log (no interactivo; sin bloqueos)
LAST=$(find "$ROOT" -type f -name 'canary_live.*.log' -print0 2>/dev/null \
       | xargs -0 -I{} echo {} \
       | LC_ALL=C sort -r \
       | head -n1)

if [[ -z "${LAST:-}" ]]; then
  echo "[ERR] no encontré canary_live.*.log"
  exit 1
fi
echo "$LAST"

# Reglas mínimas GREEN
reqs=('start EXCHANGE=binance' 'DRYRUN=1' 'ready (signal fresh)' '\[PAPER] flip' 'done')
fail=0
for r in "${reqs[@]}"; do
  if ! grep -qE "$r" "$LAST"; then
    echo "[MISS] $r"
    fail=1
  fi
done
[[ "$fail" -eq 0 ]] && echo "→ CANARY: GREEN" || echo "→ CANARY: RED"

echo
echo "== Últimos 12 (GREEN/RED) =="
find "$ROOT" -type f -name 'canary_live.*.log' -print0 2>/dev/null \
| xargs -0 -I{} echo {} \
| LC_ALL=C sort -r \
| head -n12 \
| while read -r f; do
    if grep -q 'ready (signal fresh)' "$f" \
       && grep -q '\[PAPER] flip' "$f" \
       && grep -q 'done' "$f"; then mark="GREEN"; else mark="RED"; fi
    printf -- "- \`%s\` → **%s**\n" "$(basename "$f")" "$mark"
  done
