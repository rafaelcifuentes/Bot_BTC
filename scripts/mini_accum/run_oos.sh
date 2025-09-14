#!/usr/bin/env bash
set -Eeuo pipefail
CFG="${KISS_CFG:-configs/mini_accum/config.yaml}"
H4=$(yq -r '.data.ohlc_4h_csv' "$CFG")
[[ -f "$H4" ]] || { echo "No existe $H4"; exit 1; }
CSV_MIN=$(awk -F, 'NR==2{print substr($1,1,10)}' "$H4")
CSV_MAX=$(tail -n 1 "$H4" | awk -F, '{print substr($1,1,10)}')

# args: START END SUFFIX
REQ_S=${1:?START YYYY-MM-DD}
REQ_E=${2:?END YYYY-MM-DD}
SUF=${3:-OOS}

# intersección (usa python para fechas)
read INT_S INT_E <<<"$(python3 - <<PY
from datetime import datetime
import sys
fmt='%Y-%m-%d'
csv_min=datetime.strptime('$CSV_MIN'[:10],fmt)
csv_max=datetime.strptime('$CSV_MAX'[:10],fmt)
req_s=datetime.strptime('$REQ_S',fmt)
req_e=datetime.strptime('$REQ_E',fmt)
s=max(csv_min, req_s); e=min(csv_max, req_e)
print(s.strftime(fmt), e.strftime(fmt))
PY
)"
if [[ "$INT_S" > "$INT_E" ]]; then
  echo "[SKIP] Sin intersección con CSV ($CSV_MIN..$CSV_MAX)"; exit 0
fi

echo "[RUN] $INT_S → $INT_E  (CSV: $CSV_MIN..$CSV_MAX)"
REPORT_SUFFIX="$SUF" mini-accum-backtest --config "$CFG" --start "$INT_S" --end "$INT_E"
bash scripts/mini_accum/diag_gate_mix.sh
# Auto-report si PASS
bash scripts/mini_accum/make_run_report.sh || true
