#!/usr/bin/env bash
set -Eeuo pipefail

# Ventana: últimos 120 días según CSV 4h
END=$(tail -n 1 data/ohlc/4h/BTC-USD.csv | awk -F, '{print substr($1,1,10)}')
START=$(python3 - "$END" <<'PY'
import sys, datetime
end = datetime.date.fromisoformat(sys.argv[1])
print((end - datetime.timedelta(days=120)).isoformat())
PY
)

bash scripts/mini_accum/run_preset_smart.sh "$START" "$END"

# Reporte PASS (si aplica)
bash scripts/mini_accum/make_run_report.sh || true

# Mostrar PASS nuevos (si los hay)
ls docs/runs/*PASS.md 2>/dev/null || true
