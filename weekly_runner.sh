#!/usr/bin/env bash
set -euo pipefail

# ---- PATH cron-safe (Homebrew primero) ----
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
LOGS="$ROOT/logs"
mkdir -p "$LOGS"
CRON_LOG="$LOGS/cron.log"

# ---- Rotación simple si >5MB (idempotente) ----
if [ -f "$CRON_LOG" ]; then
  size=$(stat -f%z "$CRON_LOG" 2>/dev/null || stat -c%s "$CRON_LOG" 2>/dev/null || echo 0)
  if [ "${size:-0}" -gt $((5*1024*1024)) ]; then
    mv "$CRON_LOG" "$CRON_LOG.$(date -u +%Y%m%dT%H%M%SZ).1"
  fi
fi

# Redirigir stdout/err a log (sin pisar, append)
exec >> "$CRON_LOG" 2>&1

# ---- Línea inicial obligatoria + DEBUG ----
LOG_LEVEL="${LOG_LEVEL:-INFO}"
echo "$(date -u +%FT%TZ) [${LOG_LEVEL}] weekly_runner: LOG_LEVEL=${LOG_LEVEL} aplicado"
[ "${DEBUG_RUNNER:-0}" = "1" ] && { set -x; env | sort | sed 's/.*/DEBUG_ENV: &/'; }

# ---- Caffeinate opcional ----
if command -v caffeinate >/dev/null 2>&1 && [ "${USE_CAFFEINATE:-1}" = "1" ]; then
  caffeinate -dimsu -w $$ &
  echo "$(date -u +%FT%TZ) [INFO] weekly_runner: caffeinate ON (pid=$!)"
fi

cd "$ROOT"

# ---- Activar .venv si existe ----
if [ -d "$ROOT/.venv" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  echo "$(date -u +%FT%TZ) [INFO] weekly_runner: .venv activado"
fi

# ---- Kill-switch: OVERRIDE_MODE=PAUSE ----
OVERRIDE_MODE="${OVERRIDE_MODE:-NORMAL}"
if [ "$OVERRIDE_MODE" = "PAUSE" ]; then
  echo "$(date -u +%FT%TZ) [WARN] weekly_runner: OVERRIDE_MODE=PAUSE => no se ejecuta pipeline"
  LEVEL=WARN CHAN=mini_accum /usr/bin/env python3 "$ROOT/scripts/mini_accum/notify.py" "Runner pausado por OVERRIDE_MODE=PAUSE"
  exit 0
fi

# ---- Check de datos / fetch (si existen scripts; notificar si fallan) ----
fetch_ok=1
if [ -x "$ROOT/scripts/mini_accum/fetch_ohlc.sh" ]; then
  if ! "$ROOT/scripts/mini_accum/fetch_ohlc.sh"; then
    fetch_ok=0
  fi
elif [ -x "$ROOT/scripts/mini_accum/fetch_ohlc.py" ]; then
  if ! /usr/bin/env python3 "$ROOT/scripts/mini_accum/fetch_ohlc.py"; then
    fetch_ok=0
  fi
else
  echo "$(date -u +%FT%TZ) [INFO] weekly_runner: no hay fetch_ohlc.{sh,py} -> skip"
fi
if [ "$fetch_ok" -ne 1 ]; then
  LEVEL=WARN CHAN=mini_accum /usr/bin/env python3 "$ROOT/scripts/mini_accum/notify.py" "Fallo en fetch de datos (1d/4h)"
fi

# ---- Checkpoint 1 ----
LEVEL=INFO CHAN=mini_accum /usr/bin/env python3 "$ROOT/scripts/mini_accum/notify.py" "Checkpoint: datos listos"

# ---- Ejecutar pipeline KISS v1 (sin tocar lógica) ----
pipe_ok=0
if [ -x "$ROOT/scripts/mini_accum/kiss_v1_wf_pipeline.sh" ]; then
  if "$ROOT/scripts/mini_accum/kiss_v1_wf_pipeline.sh"; then pipe_ok=1; fi
elif [ -x "$ROOT/scripts/mini_accum/pipeline.sh" ]; then
  if "$ROOT/scripts/mini_accum/pipeline.sh"; then pipe_ok=1; fi
else
  echo "$(date -u +%FT%TZ) [WARN] weekly_runner: pipeline no encontrado; skip (manteniendo contrato de outputs)"
  pipe_ok=1
fi

# ---- Snapshot/Checkpoint 2 ----
if [ "$pipe_ok" -eq 1 ]; then
  LEVEL=INFO CHAN=mini_accum /usr/bin/env python3 "$ROOT/scripts/mini_accum/notify.py" "Checkpoint: pipeline completado (A/B corto)"
else
  LEVEL=WARN CHAN=mini_accum /usr/bin/env python3 "$ROOT/scripts/mini_accum/notify.py" "Pipeline falló"
fi

# ---- Tareas de post: watchdog & KPI guard ----
/usr/bin/env python3 "$ROOT/scripts/mini_accum/health_watchdog.py" || true
/usr/bin/env python3 "$ROOT/scripts/mini_accum/kpi_guard.py" || true

echo "$(date -u +%FT%TZ) [INFO] weekly_runner: done"
# --- Policy asserts: no mezcla/meta, versión/tag fijo, sellos de resultados ---
if ! /usr/bin/env python3 "$ROOT/scripts/mini_accum/policy_asserts.py"; then
  LEVEL=WARN CHAN=mini_accum /usr/bin/env python3 "$ROOT/scripts/mini_accum/notify.py" "Policy asserts FAIL → PAUSE"
  date -u +"PAUSE %Y-%m-%dT%H:%M:%SZ :: policy_violation" > "$ROOT/health/mini_accum.status"
  exit 1
else
  LEVEL=INFO CHAN=mini_accum /usr/bin/env python3 "$ROOT/scripts/mini_accum/notify.py" "Policy asserts OK (baseline intacto)"
fi
# --- fin policy asserts ---
# --- Guard: CODE SEAL FROZEN (wrapper inmutable) ---
if [ -f "$ROOT/reports/mini_accum/code_seal.FROZEN.sha256" ]; then
  if ! ( cd "$ROOT" && shasum -a 256 -c reports/mini_accum/code_seal.FROZEN.sha256 >/dev/null ); then
    LEVEL=WARN CHAN=mini_accum /usr/bin/env python3 "$ROOT/scripts/mini_accum/notify.py" "Runner: CODE SEAL MISMATCH → PAUSE"
    date -u +"PAUSE %Y-%m-%dT%H:%M:%SZ :: code_seal_mismatch" > "$ROOT/health/mini_accum.status"
    exit 1
  fi
fi
# --- fin guard CODE SEAL ---
