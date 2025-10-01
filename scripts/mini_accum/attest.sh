#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
TAG="KISSv1_BASE_20250915_1642_final"
ok=1

echo "== ATTEST START $(date -u +%FT%TZ) =="
# 0) policy asserts (no mezcla/meta, versión/tag fijo, sellos)
if ! /usr/bin/env python3 "$ROOT/scripts/mini_accum/policy_asserts.py"; then
  echo "[FAIL] policy_asserts.py"
  ok=0
fi

# 1) report JSON debe ser OK y sin problemas
rep="$ROOT/reports/mini_accum/policy_asserts_report.json"
status="$(jq -r '.status' "$rep" 2>/dev/null || echo FAIL)"
no_meta="$(jq -r '.no_mezcla_meta' "$rep" 2>/dev/null || echo false)"
base_ok="$(jq -r '.baseline_intacto' "$rep" 2>/dev/null || echo false)"
echo "[INFO] report: status=$status no_mezcla_meta=$no_meta baseline_intacto=$base_ok"
[ "$status" = "OK" ] && [ "$no_meta" = "true" ] && [ "$base_ok" = "true" ] || ok=0

# 2) contrato de señal + versión fija
sig="$ROOT/signals/mini_accum/latest.json"
jq -e --arg v "$TAG" '.version==$v and (.ts_utc|type=="string") and .health' "$sig" >/dev/null || {
  echo "[FAIL] latest.json contrato/version"; ok=0; }

# 3) sellos de resultados/robustez
jq -e '.results_reserved."2023_pct"==185.26 and .results_reserved."2024_pct"==196.86 and .results_reserved."2025H1_pct"==32.96' \
  "$ROOT/reports/mini_accum/perf_seal.json" >/dev/null || { echo "[FAIL] perf_seal.json"; ok=0; }
jq -e '.robustness.PBO_approx==0.107 and .robustness.DSR=="OK" and (.robustness.cost_stress_bps.fee==6) and (.robustness.cost_stress_bps.slip==6)' \
  "$ROOT/reports/mini_accum/robustness_seal.json" >/dev/null || { echo "[FAIL] robustness_seal.json"; ok=0; }

# 4) cron reforzado (guards anti-mezcla presentes) y PATH seguro
cron_block="$(mktemp)"
crontab -l 2>/dev/null | sed -n '/# MINI_ACCUM BEGIN/,/# MINI_ACCUM END/p' > "$cron_block" || true
grep -q 'MIX_DISABLE=1' "$cron_block" || { echo "[FAIL] cron sin guards anti-mezcla"; ok=0; }
grep -q '^PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin' "$cron_block" || echo "[WARN] PATH cron-safe no visible (ok si se define arriba)"

# 5) anti-sleep disponible
command -v caffeinate >/dev/null || echo "[WARN] caffeinate no encontrado"
pmset -g custom | grep -E 'AC Power| sleep| powernap' || true

# 6) runner última ejecución: línea inicial obligatoria
LOG="$ROOT/logs/cron.log"
tail -n 200 "$LOG" 2>/dev/null | grep -q 'LOG_LEVEL=INFO aplicado' || echo "[WARN] cron.log sin línea inicial visible (ok si log recién rotado)"

# 7) resumen
if [ "$ok" -eq 1 ]; then
  echo "== ATTESTATION: OK (ni un satoshi cedido; resultados reservados; robustez sellada) =="
  exit 0
else
  echo "== ATTESTATION: FAIL =="
  exit 1
fi
