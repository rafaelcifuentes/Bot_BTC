#!/usr/bin/env bash
set -euo pipefail

export RUN_MODE=paper
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "== SelfTest mini_accum (RUN_MODE=$RUN_MODE, LOG_LEVEL=$LOG_LEVEL) =="

# --- 1) Latencia (objetivo < 30s) ---
echo "-- Latencia --"
/usr/bin/time -p python3 scripts/mini_accum/live_wrapper.py >/tmp/latency.out 2>/tmp/latency.time || true
cat /tmp/latency.time
real_s=$(awk '/^real/{print $2}' /tmp/latency.time)
if awk "BEGIN{exit !($real_s < 30.0)}"; then
  echo "[PASS] Latencia real=${real_s}s"
else
  echo "[FAIL] Latencia real=${real_s}s (>=30s)"
fi

# --- 2) Idempotencia (no duplica flip si no cambia posición) ---
echo "-- Idempotencia (no-op doble) --"
before_sz=$(stat -f %z reports/mini_accum/flips_log.csv 2>/dev/null || echo 0)
python3 scripts/mini_accum/live_wrapper.py >/dev/null
python3 scripts/mini_accum/live_wrapper.py >/dev/null
after_sz=$(stat -f %z reports/mini_accum/flips_log.csv 2>/dev/null || echo 0)
if [[ "$after_sz" == "$before_sz" ]]; then
  echo "[PASS] Sin duplicados del flip en doble ejecución"
else
  echo "[WARN] Tamaño cambió (${before_sz} -> ${after_sz}). Revisa tail:"
  tail -n 3 reports/mini_accum/flips_log.csv || true
fi

# --- 3) Flip forzado: debe registrarse 1 vez y luego no duplicar ---
echo "-- Flip forzado (1 vez) --"
last_pos=$(jq -r '.position_pct_btc // 0' signals/mini_accum/latest.json 2>/dev/null || echo 0)
# si last_pos>=0.5 => forzamos 0; si no, forzamos 1
if awk "BEGIN{exit !($last_pos >= 0.5)}"; then new_pos=0; else new_pos=1; fi

# IMPORTANTÍSIMO: refrescar ts_utc para que NO sea stale
now_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
jq --argjson np "$new_pos" --arg ts "$now_utc" \
   '.position_pct_btc=$np | .reason="idempotency-test" | .ts_utc=$ts' \
   signals/mini_accum/latest.json > /tmp/latest.json && mv /tmp/latest.json signals/mini_accum/latest.json

sz1=$(stat -f %z reports/mini_accum/flips_log.csv 2>/dev/null || echo 0)
python3 scripts/mini_accum/live_wrapper.py >/dev/null
sz2=$(stat -f %z reports/mini_accum/flips_log.csv 2>/dev/null || echo 0)
python3 scripts/mini_accum/live_wrapper.py >/dev/null
sz3=$(stat -f %z reports/mini_accum/flips_log.csv 2>/dev/null || echo 0)

if [[ "$sz2" != "$sz1" && "$sz3" == "$sz2" ]]; then
  echo "[PASS] Flip forzado se registró 1 vez y no se duplicó"
else
  echo "[FAIL] Comportamiento inesperado en flip forzado (sz: $sz1 -> $sz2 -> $sz3)"
  tail -n 5 reports/mini_accum/flips_log.csv || true
fi

# --- 4) Kill-switch ---
echo "-- Kill-switch --"
OVERRIDE_MODE=PAUSE python3 scripts/mini_accum/live_wrapper.py 2>&1 | grep -q '\[PAUSE\]' \
  && echo "[PASS] PAUSE aplicado" || echo "[FAIL] PAUSE no detectado"

# Aceptamos cualquiera de estos mensajes en NORMAL: [NO-OP], "same position", o un flip
OVERRIDE_MODE=NORMAL python3 scripts/mini_accum/live_wrapper.py 2>&1 \
  | grep -Eq '\[NO-OP\]|same position|flip:' \
  && echo "[PASS] NORMAL aplicado" || echo "[FAIL] NORMAL no detectado"

echo "== Fin SelfTest =="
