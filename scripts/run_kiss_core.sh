#!/usr/bin/env bash
set -Eeuo pipefail
yq -e '.' configs/mini_accum/config.yaml >/dev/null
START=$(awk -F, 'NR==2{print substr($1,1,10); exit}' data/ohlc/4h/BTC-USD.csv)
END=$(tail -n 1 data/ohlc/4h/BTC-USD.csv | awk -F, '{print substr($1,1,10)}')
REPORT_SUFFIX="KISSv4_CORE" mini-accum-backtest --config configs/mini_accum/config.yaml --start "$START" --end "$END"
bash scripts/mini_accum/diag_gate_mix.sh
