#!/bin/zsh
set -euo pipefail
set -o pipefail 2>/dev/null || true

: ${ROOT:="$HOME/PycharmProjects/Bot_BTC"}

cd "$ROOT"

DAY="$(date -u +%F)"
DSTR="${DAY//-/}"
OUTDIR="evidence/dayN_${DAY}"
mkdir -p "$OUTDIR"
print -r -- "[RUN] selector_shadow: ROOT=$ROOT DAY=$DAY OUTDIR=$OUTDIR"

bb_streak_canary_kiss 3    # hoy+ayer+anteayer → verás 3/5 si todo fue GREENbb_streak_canary_kiss 3    # hoy+ayer+anteayer → verás 3/5 si todo fue GREEN# --- ATTEST del día (OK/FAIL dentro del bloque del día) ---
ATTEST_OK=$(
  awk -v d="$DAY" '
    BEGIN{inblk=0; ok=0}
    $0 ~ "^== ATTEST START " d "T" { if(inblk){print ok; exit} inblk=1; next }
    /^== ATTEST START / && inblk==1 { print ok; exit }
    inblk && /ATTESTATION: OK/ { ok=1 }
    END{ if(inblk) print ok }
  ' "logs/cron.log" 2>/dev/null || echo 0
)
[[ -z "$ATTEST_OK" ]] && ATTEST_OK=0

# --- Canarios del día (UTC) ---
setopt NULL_GLOB
files=($ROOT/logs/canary_live.${DSTR}T*.log(N) $ROOT/evidence/day*_${DAY}/canary_live.${DSTR}T*.log(N))

TOTAL=0 GREEN=0 PAUSE=0 YELLOW=0
valid_list=()

for f in $files; do
  grep -q 'start EXCHANGE=binance DRYRUN=1' "$f" || continue
  (( TOTAL++ ))
  if grep -q 'PAUSE' "$f"; then
    (( PAUSE++ ))
  elif grep -q 'ready (signal' "$f" && grep -q '\[INFO\] canary_live: done' "$f"; then
    (( GREEN++ ))
    valid_list+=("$f")
  else
    (( YELLOW++ ))
  fi
done

# --- Decisión en sombra (no operacional) ---
# Mantener "insuficiente" por diseño (no toca lógica; sólo registro).
DECISION="insuficiente"
RULE="fric_live_shadow_v1"  # sólo etiqueta para trazabilidad

# --- Persistencia (JSON + MD) ---
TS=$(date -u +%FT%TZ)
JSON="${OUTDIR}/selector_shadow.json"
MD="${OUTDIR}/selector_shadow.md"
: > "$JSON"; : > "$MD"

cat > "$JSON" <<EOF
{
  "ts_utc": "$TS",
  "rule": "$RULE",
  "decision": "$DECISION",
  "day_utc": "$DAY",
  "attest_ok": $ATTEST_OK,
  "counts": { "total": $TOTAL, "green": $GREEN, "pause": $PAUSE, "yellow": $YELLOW },
  "notes": "Solo registro en sombra. No cambia lógica ni envía señales."
}
EOF

{
  echo "# Selector por fricción (sombra) — ${DAY} (UTC)"
  echo
  echo "- **ATTEST OK**: ${ATTEST_OK}"
  echo "- **Canarios (binance, DRYRUN=1)** — total: ${TOTAL} | GREEN: ${GREEN} | PAUSE: ${PAUSE} | YELLOW: ${YELLOW}"
  echo "- **Regla (shadow)**: ${RULE}"
  echo "- **Decisión (shadow)**: \`${DECISION}\`"
  echo
  echo "## Evidencias GREEN (últimos 12)"
  if (( ${#valid_list} > 0 )); then
    printf '%s\n' "${valid_list[@]}" | tail -n 12 | sed 's#^#- #'
  else
    echo "- (sin GREEN hoy)"
  fi
} > "$MD"

echo "[SHADOW] selector: total=$TOTAL green=$GREEN pause=$PAUSE yellow=$YELLOW attest_ok=$ATTEST_OK"
echo "[OK] selector_shadow: $JSON  |  $MD"
if [[ ! -f "$JSON" || ! -f "$MD" ]]; then
  echo "[ERROR] selector_shadow: expected outputs missing in $OUTDIR" >&2
  exit 1
fi
