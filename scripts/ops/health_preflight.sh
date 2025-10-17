#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

pass(){ printf "[OK] %s\n" "$*"; }
fail(){ printf "[FAIL] %s\n" "$*" >&2; exit 1; }
warn(){ printf "[WARN] %s\n" "$*" >&2; }

# 1) Red/latencia mínima a un endpoint público (ajusta si usas otro exchange)
if command -v curl >/dev/null 2>&1; then
  T0=$(date +%s%3N 2>/dev/null || date +%s)
  if curl -m 5 -sS https://api.kraken.com/0/public/Time >/dev/null; then
    T1=$(date +%s%3N 2>/dev/null || date +%s)
    DT=$((T1-T0))
    pass "Red OK (latencia ~${DT}ms)"
  else
    fail "Red: no pude contactar endpoint público"
  fi
else
  warn "curl no está; salto test de latencia"
fi

# 2) Credenciales mínimas presentes (ajusta nombres si usas otros)
MISSING=0
for v in EXCHANGE_API_KEY EXCHANGE_API_SECRET; do
  if [ -z "${!v:-}" ]; then warn "Falta \$${v}"; MISSING=1; fi
done
[ "$MISSING" -eq 0 ] && pass "Credenciales presentes" || fail "Credenciales incompletas"

# 3) Espacio en disco (mínimo 2GB libres en el proyecto)
NEEDED=$((2*1024*1024*1024))
FREE=$(df -k . | awk 'NR==2{print $4*1024}')
[ "$FREE" -ge "$NEEDED" ] && pass "Disco OK ($(df -h . | awk 'NR==2{print $4" libres"}'))" || fail "Poco espacio en disco"

# 4) Heartbeat escribiendo
mkdir -p corazon
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "$TS,alive" >> corazon/heartbeat.csv
tail -n1 corazon/heartbeat.csv | grep -q "$TS" && pass "Heartbeat escribió $TS" || fail "Heartbeat no escribió"

# 5) Guardarraíles cargados: script existe y cron activo
[ -x scripts/ops/guardrail_rollback.sh ] && pass "guardrail_rollback.sh presente" || fail "Falta scripts/ops/guardrail_rollback.sh"
if crontab -l 2>/dev/null | grep -q 'guardrail_rollback.sh'; then
  pass "Cron de guardarraíl activo"
else
  warn "Cron de guardarraíl NO encontrado (lo activo)"
  ( crontab -l 2>/dev/null; \
    echo "*/2 * * * * cd $HOME/PycharmProjects/Bot_BTC && scripts/ops/guardrail_rollback.sh >> logs/guardrail.log 2>&1" \
  ) | crontab -
  pass "Cron de guardarraíl instalado"
fi

pass "healthPreflight COMPLETO"
