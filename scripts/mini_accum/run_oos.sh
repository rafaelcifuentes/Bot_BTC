#!/usr/bin/env zsh
# Mini-Accum OOS runner (zsh-only). Safe header: nunca mata la shell si es 'sourced'.

emulate -L zsh
set -e
set -u
set -o pipefail

# Si fue "sourced" (no ejecutado), salir silenciosamente
if [[ ${ZSH_EVAL_CONTEXT-} == *:file ]]; then
  return 0
fi

# ---------- Helpers ----------
safe_exit() {
  local code=${1:-0}
  # sourced desde zsh
  if [[ ${ZSH_EVAL_CONTEXT-} == *:file ]]; then return $code; fi
  # sourced desde bash
  if [[ -n ${BASH_VERSION-} && ${BASH_SOURCE[0]-} != "$0" ]]; then return $code; fi
  exit $code
}

nonfatal() { "$@" || printf '[WARN] ignored: %s\n' "$*"; }

# ---------- Args ----------
REQ_S=${1:?Usage: $0 START(YYYY-MM-DD) END(YYYY-MM-DD) [CFG] [SUFFIX]}
REQ_E=${2:?Usage: $0 START(YYYY-MM-DD) END(YYYY-MM-DD) [CFG] [SUFFIX]}
CFG=${3:-${KISS_CFG:-configs/mini_accum/config.yaml}}
SUF=${4:-OOS}

# ---------- YAML & datos ----------
if ! command -v yq >/dev/null 2>&1; then
  echo "[ERR] yq no encontrado. Instálalo (p.ej., brew install yq)." >&2
  safe_exit 1
fi

H4=$(yq -r '.data.ohlc_4h_csv' "$CFG" 2>/dev/null)
: ${H4:=''}
if [[ -z "$H4" || ! -f "$H4" ]]; then
  echo "[ERR] No existe OHLC 4h CSV definido en $CFG: '$H4'" >&2
  safe_exit 1
fi

# Límites del CSV (asume primera columna 'ts' en ISO)
CSV_MIN=$(awk -F, 'NR==2{print substr($1,1,10); exit}' "$H4")
CSV_MAX=$(tail -n 1 "$H4" | awk -F, '{print substr($1,1,10)}')

# ---------- Intersección de rango ----------
read INT_S INT_E <<<"$(REQ_S=$REQ_S REQ_E=$REQ_E CSV_MIN=$CSV_MIN CSV_MAX=$CSV_MAX python3 - <<'PY'
from datetime import datetime
import os, sys
fmt='%Y-%m-%d'
CSV_MIN=os.environ.get('CSV_MIN','')[:10]
CSV_MAX=os.environ.get('CSV_MAX','')[:10]
REQ_S=os.environ.get('REQ_S','')
REQ_E=os.environ.get('REQ_E','')
try:
    csv_min=datetime.strptime(CSV_MIN,fmt)
    csv_max=datetime.strptime(CSV_MAX,fmt)
    req_s=datetime.strptime(REQ_S,fmt)
    req_e=datetime.strptime(REQ_E,fmt)
    s=max(csv_min, req_s); e=min(csv_max, req_e)
    print(s.strftime(fmt), e.strftime(fmt))
except Exception:
    print(REQ_S, REQ_E)
PY
)"

if [[ "$INT_S" > "$INT_E" ]]; then
  echo "[SKIP] Sin intersección con CSV ($CSV_MIN..$CSV_MAX)"
  safe_exit 0
fi

echo "[RUN] $INT_S → $INT_E  (CSV: $CSV_MIN..$CSV_MAX)  CFG=$CFG  SUF=$SUF"

# ---------- Ejecutar backtest ----------
REPORT_SUFFIX="$SUF" mini-accum-backtest --config "$CFG" --start "$INT_S" --end "$INT_E"

# Diagnósticos / reportes (no bloqueantes)
nonfatal bash scripts/mini_accum/diag_gate_mix.sh
nonfatal bash scripts/mini_accum/make_run_report.sh

safe_exit 0