#!/usr/bin/env bash
set -euo pipefail

# Uso:
#   scripts/ops/insert_emit_signals.sh            # autodetecta TARGET
#   scripts/ops/insert_emit_signals.sh --dry-run  # muestra diff sin escribir
#   scripts/ops/insert_emit_signals.sh path/al/loop.sh

dry=0
target=""

if [[ $# -gt 0 ]]; then
  if [[ "${1:-}" == "--dry-run" ]]; then dry=1; shift; fi
fi
if [[ $# -gt 0 ]]; then
  target="$1"
else
  # Autodetección simple: agrega aquí más candidatos si quieres
  for cand in \
    scripts/run_bot.sh \
    scripts/mini_accum/run_bull_hold_levered.sh \
    scripts/mini_accum/runner_cron.sh
  do
    [[ -f "$cand" ]] && { target="$cand"; break; }
  done
fi

[[ -n "$target" && -f "$target" ]] || { echo "[ERR] No encontré TARGET. Pásalo como argumento."; exit 2; }

# Idempotencia: si ya está insertado, salimos
if grep -qE 'emit_signal\.sh|emitir señal \(opt-in\)' "$target"; then
  echo "[SKIP] ya contiene bloque de emisión: $target"
  exit 0
fi

# Bloque a insertar (no expandir variables aquí)
read -r -d '' BLOCK <<'EOF'
# --- emitir señal (opt-in) ---
TAG="$(cat deploy/ACTIVE.tag 2>/dev/null || echo UNKNOWN)"
ACTION="${DECISION:-${ACTION:-HOLD}}"    # BUY / SELL / HOLD
LAST_PRICE="${PX:-nan}"                  # precio observado/evaluado
REASON="${MOTIVO:-n/a}"                  # breve: "breakout", "stop ATR", etc.

ENABLE_SIGNALS=${ENABLE_SIGNALS:-0}
SIGNALS_FILE=${SIGNALS_FILE:-signals/stream.csv}
[ "$ENABLE_SIGNALS" = "1" ] && scripts/ops/emit_signal.sh "$TAG" "$ACTION" "$LAST_PRICE" "$REASON"
# --- fin emisión ---
EOF

export BLOCK

tmp="${target}.tmp.$$"

# Inserta tras la primera asignación a DECISION= o ACTION=; si no hay, lo añade al final.
awk '
BEGIN{done=0}
{
  print $0
  if (!done && $0 ~ /(^|[^A-Za-z0-9_])(DECISION|ACTION)[[:space:]]*=/) {
    print ENVIRON["BLOCK"]
    done=1
  }
}
END{
  if (!done) print ENVIRON["BLOCK"]
}
' "$target" > "$tmp"

if [[ $dry -eq 1 ]]; then
  # Diff amigable (no color por si no está colordiff)
  echo "----- DIFF (dry-run) -----"
  diff -u "$target" "$tmp" || true
  rm -f "$tmp"
  exit 0
fi

mv "$tmp" "$target"
chmod +x "$target"

# Verificación
if ! grep -nE 'emit_signal\.sh|emitir señal \(opt-in\)' "$target" >/dev/null; then
  echo "[ERR] no se insertó el bloque en $target"
  exit 1
fi

echo "[OK] bloque insertado en: $target"
exit 0
