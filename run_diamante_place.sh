# === run_diamante_place.sh ===
#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dry}"              # dry | live
EXCHANGE="${EXCHANGE:-binanceus}"
ASSETS="${ASSETS:-BTC-USD,ETH-USD}"   # sin SOL
THRESH="${THRESH:-0.60}"
WALK_K="${WALK_K:-3}"
HORIZON="${HORIZON:-60}"

# Capital / riesgo / pesos / stops
: "${CAPITAL_USD:?Define CAPITAL_USD, p.ej. export CAPITAL_USD=10000}"
RISK_PCT="${RISK_PCT:-0.0075}"
WEIGHT_BTC="${WEIGHT_BTC:-0.70}"
WEIGHT_ETH="${WEIGHT_ETH:-0.30}"
STOP_BTC_PCT="${STOP_BTC_PCT:-0.015}"
STOP_ETH_PCT="${STOP_ETH_PCT:-0.030}"

# Señales (WF3, variante segura)
BTC_SIG="${BTC_SIG:-reports/diamante_btc_wf3_week1_plus.csv}"
ETH_SIG="${ETH_SIG:-reports/diamante_eth_wf3_week1_plus.csv}"

RUN_ID="${RUN_ID:-wk1_wf${WALK_K}_th$(printf '%03d' "$(python - <<'PY'
t = 0.60
try:
    import os
    t = float(os.environ.get("THRESH","0.60"))
except: pass
print(int(round(t*100)))
PY
)"))}"

mkdir -p logs

echo "[CHECK] mode=$MODE | assets=$ASSETS | THRESH=$THRESH | K=$WALK_K | H=$HORIZON | CAP=$CAPITAL_USD | RISK=$RISK_PCT"
echo "[CHECK] signals: $BTC_SIG , $ETH_SIG"

# Opcional: refrescar señales al reloj actual (cierra barra 4h más reciente)
# Actívalo con: REFRESH=1 bash ./run_diamante_place.sh dry
if [[ "${REFRESH:-0}" == "1" ]]; then
  echo "[REFRESH] Regenerando señales con FREEZE=now (UTC)…"
  FREEZE="$(date -u '+%Y-%m-%d %H:00')" \
  THRESH="$THRESH" WF_LIST="$WALK_K" \
  bash ./run_diamante_day5.sh
fi

ARGS=( place
  --exchange "$EXCHANGE"
  --assets "$ASSETS"
  --signals "$BTC_SIG" "$ETH_SIG"
  --threshold "$THRESH"
  --walk-k "$WALK_K"
  --horizon "$HORIZON"
  --capital "$CAPITAL_USD"
  --risk "$RISK_PCT"
  --weights "BTC-USD:$WEIGHT_BTC,ETH-USD:$WEIGHT_ETH"
  --stops   "BTC-USD:$STOP_BTC_PCT,ETH-USD:$STOP_ETH_PCT"
  --log "logs/${RUN_ID}_${MODE}.log"
)

if [[ "$MODE" == "dry" ]]; then
  ARGS+=( --no-submit --debug )
else
  ARGS+=( --submit --confirm )
fi

python runner.py "${ARGS[@]}"

echo
echo "== Últimas líneas del log =="
tail -n 200 "logs/${RUN_ID}_${MODE}.log" || true
grep -E "ENTRY|STOP|TARGET|SIZE|RISK|WEIGHT|WARN|ERROR|SKIP" "logs/${RUN_ID}_${MODE}.log" || true