#!/usr/bin/env bash
set -Eeuo pipefail

CORE="configs/mini_accum/presets/CORE_2025.yaml"
BULL="configs/mini_accum/presets/BULL_H2_2024.yaml"

S="${1:-}"
E="${2:-}"

# Si no pasas fechas, usa últimos 120 días desde el CSV 4h
if [[ -z "${S}" || -z "${E}" ]]; then
  END=$(tail -n 1 data/ohlc/4h/BTC-USD.csv | awk -F, '{print substr($1,1,10)}')
  S=$(python3 - "$END" <<'PY'
import sys, datetime
end = datetime.date.fromisoformat(sys.argv[1])
print((end - datetime.timedelta(days=120)).isoformat())
PY
)
  E="$END"
fi

# Intersección con H2-2024 (zsh-friendly: usar ! (A || B) en vez de >=/<=)
H2S="2024-07-01"; H2E="2024-12-31"
use_bull="false"
if [[ ! ( "$E" < "$H2S" || "$S" > "$H2E" ) ]]; then
  use_bull="true"
fi

CFG="$CORE"; TAG="KISSv4_CORE_SMART"
if [[ "$use_bull" == "true" ]]; then CFG="$BULL"; TAG="KISSv4_BULL_SMART"; fi

echo "[INFO] start=$S end=$E preset=$([[ $use_bull == true ]] && echo BULL_H2_2024 || echo CORE_2025)"
REPORT_SUFFIX="${TAG}" mini-accum-backtest --config "$CFG" --start "$S" --end "$E" || true

# Diag opcional (no falla si no está)
bash scripts/mini_accum/diag_gate_mix.sh || true

# KPI corto
last_kpi=$(ls -t reports/mini_accum/*_kpis__${TAG}.csv 2>/dev/null | head -n1 || true)
if [[ -n "${last_kpi:-}" && -f "$last_kpi" ]]; then
  awk -F, -v F="$last_kpi" 'NR==2{printf "[KPI] %s  netBTC=%.4f  mdd=%.3f  fpy=%.2f\n",F,$1,$4,$7}' "$last_kpi"
fi
