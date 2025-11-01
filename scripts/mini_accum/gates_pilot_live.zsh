#!/usr/bin/env zsh
set -euo pipefail

ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
STATUS="${STATUS:-$ROOT/health/mini_accum.status}"

# 1) KPI Guard / STATUS robusto (no corta en ARMED por defecto)
: "${PILOT_STATUS_STRICT:=0}"   # 0=solo WARN; 1=FAIL corta
if [[ ! -s "$STATUS" ]]; then
  echo "[WARN] STATUS vacío o inexistente: $STATUS"
  [[ "$PILOT_STATUS_STRICT" == "1" ]] && exit 1
else
  if grep -iq '^health=ok\b' "$STATUS" || grep -qE '^\[OK\]|^OK ' "$STATUS"; then
    :
  else
    echo "[WARN] KPI Guard no está OK (STATUS sin OK/health=ok)"
    [[ "$PILOT_STATUS_STRICT" == "1" ]] && exit 1
  fi
fi

# 2) Sin tormenta (≤1/h) — contamos runs ARMED en la última hora
now_epoch=$(date -u +%s)
count_last_hour=$(ls -1t "$ROOT"/logs/pilot_armed.*.log 2>/dev/null | while read -r f; do
  bn=$(basename "$f"); ts=${bn#pilot_armed.}; ts=${ts%.log}
  t=$(date -u -j -f "%Y%m%dT%H%M%SZ" "$ts" "+%s" 2>/dev/null || true)
  [[ -n "$t" ]] && [[ $(( now_epoch - t )) -le 3600 ]] && echo 1
done | wc -l | tr -d ' ')
[[ "${count_last_hour:-0}" -le 1 ]] || { echo "[FAIL] storm guard: $count_last_hour ejecuciones en <1h"; exit 1; }

# 3) ARMED no debe mostrar placed/filled reales
if grep -Eiq 'placed|filled|orderId' "$ROOT"/logs/pilot_armed.*.log 2>/dev/null; then
  echo "[FAIL] Señales de envío real detectadas en logs ARMED"; exit 1;
fi

# 4) === KISS PASS/FAIL mini-ítem ===
: "${KISS_SATS_MIN:=1.00}"
: "${KISS_MDD_RATIO_MAX:=1.00}"
: "${KISS_FPY_MAX:=26}"
# Evita SMOKE/SANITY por defecto; céntrate en OOS_*REGIME
: "${KPI_GLOB:=reports/mini_accum/*kpis*OOS_*REGIME.csv}"
: "${KISS_STRICT:=0}"   # 0=solo WARN; 1=FAIL corta

TMP_OUT="/tmp/kiss_guard.$$.out"
if scripts/mini_accum/kpi_kiss_guard.py >"$TMP_OUT" 2>&1; then
  tail -n1 "$TMP_OUT"
else
  cat "$TMP_OUT" >&2
  echo "[WARN] KPI Guard (KISS) en FAIL (no corta en ARMED)" >&2
  if [[ "$KISS_STRICT" == "1" ]]; then
    rm -f "$TMP_OUT"; exit 1
  fi
fi
rm -f "$TMP_OUT"

echo "[OK] gates_pilot_live"
