#!/usr/bin/env bash
command -v caffeinate >/dev/null 2>&1 && caffeinate -dimsu -w $$ &
set -euo pipefail


# --- cron-safe PATH fallback ---
export PATH=${PATH:-/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin}
case ":$PATH:" in *:/opt/homebrew/bin:*) :;; *) export PATH="/opt/homebrew/bin:$PATH";; esac



# Fallback PATH si venimos de cron "vacío"
export PATH=${PATH:-/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin}
case ":$PATH:" in
  *:/opt/homebrew/bin:*) :;;
  *) export PATH="/opt/homebrew/bin:$PATH";;
esac

# --- paths & logging ---
PROJECT_DIR="${PROJECT_DIR:-$HOME/PycharmProjects/Bot_BTC}"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
# Loggea SIEMPRE a logs/cron.log (manual o vía cron)
exec >> "$LOG_DIR/cron.log" 2>&1

echo "==== $(date -u +'%FT%TZ') :: weekly_runner start ===="

# --- entorno Python ---
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  echo "[INFO] venv activada"
else
  echo "[WARN] venv no encontrada, sigo con Python del sistema"
fi

# --- env de ejecución ---
export RUN_MODE="${RUN_MODE:-paper}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export OVERRIDE_MODE="${OVERRIDE_MODE:-NORMAL}"

# Freshness/watchdog (horas)
export FRESHNESS_MAX_HOURS="${FRESHNESS_MAX_HOURS:-8}"
export WATCHDOG_HOURS="${WATCHDOG_HOURS:-8}"

# Ingesta por defecto
export EXCHANGE="${EXCHANGE:-binanceus}"
export SYMBOL="${SYMBOL:-BTC/USDT}"
export D1_CSV="${D1_CSV:-data/ohlc/1d/BTC-USD.csv}"
export H4_CSV="${H4_CSV:-data/ohlc/4h/BTC-USD.csv}"

# --- opcional: gating horario (desactivado porque cron ya dispara a 08:05 UTC) ---
# if [[ "${FORCE:-0}" != "1" ]]; then
#   hhmm="$(date -u +'%H%M')"
#   [[ "$hhmm" == "0805" ]] || { echo "[INFO] fuera de ventana 08:05Z, salgo"; exit 0; }
# fi

# --- 1) Fetch OHLCV fresco (1d/4h) ---
echo "[STEP] fetch OHLCV"
python3 scripts/mini_accum/fetch_ohlcv_ccxt.py --exchange "$EXCHANGE" --symbol "$SYMBOL" --tf 1d --out "$D1_CSV" || echo "[WARN] 1d fetch failed (continuing)"
python3 scripts/mini_accum/fetch_ohlcv_ccxt.py --exchange "$EXCHANGE" --symbol "$SYMBOL" --tf 4h --out "$H4_CSV" || echo "[WARN] 4h fetch failed (continuing)"

# --- 2) Botón semanal (pipeline + señal + KPIs + snapshot) ---
echo "[STEP] run_once_paper.sh"
 /bin/bash scripts/mini_accum/run_once_paper.sh || echo "[WARN] run_once_paper.sh terminó con aviso"

# --- 3) Notificación corta (resumen) ---
echo "[STEP] notify checkpoint"
AB_SUM="$(sed -n '1,80p' reports/mini_accum/ab_latest.md 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g' | cut -c1-400)"
LIVE_TAIL="$(tail -n 3 reports/mini_accum/live_kpis.csv 2>/dev/null | tr '\n' ' ')"
FLIPS_TAIL="$(tail -n 3 reports/mini_accum/flips_log.csv 2>/dev/null | tr '\n' ' ')"
HEALTH="$(tr '\n' ' ' < health/mini_accum.status 2>/dev/null || true)"

python3 scripts/mini_accum/notify.py "Checkpoint paper — LIVE: ${LIVE_TAIL} | FLIPS: ${FLIPS_TAIL} | HEALTH: ${HEALTH} | AB: ${AB_SUM}" || echo "[WARN] notify falló"


echo "==== $(date -u +'%FT%TZ') :: weekly_runner end ===="