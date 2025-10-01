#!/usr/bin/env bash
# Corazón — FREEZE semanal (sombra)
set -euo pipefail

FREEZE_DATE="${1:-2025-09-15}"         # YYYY-MM-DD
FREEZE_TS="${FREEZE_DATE} 00:00"
ASSET="BTC-USD"
PERIOD_DAYS=730
HORIZONS="30,60,90"
SLIP=0.0002
COST=0.0004
OHLC4H="data/ohlc/4h/${ASSET}.csv"
RULES_SNAP="configs/heart_rules_${FREEZE_DATE}.yaml"
RULES_DEFAULT="configs/heart_rules.yaml"
WEIGHTS_OUT="corazon/weights_${FREEZE_DATE}.csv"
LQ_OUT="corazon/lq_${FREEZE_DATE}.csv"
BASE_CSV="reports/diamante_btc_costes_freeze_${FREEZE_DATE}.csv"
BASE_BARS="reports/diamante_btc_costes_freeze_${FREEZE_DATE}_bars.csv"
HEART_DIR="reports/heart"
OUT_SUMMARY_MD="${HEART_DIR}/summary_diamante_btc_costes_freeze_${FREEZE_DATE}_bars.md"
OUT_KPIS_CSV="${HEART_DIR}/kpis_diamante_btc_costes_freeze_${FREEZE_DATE}_bars.csv"
XI_LOG="corazon/daily_xi.csv"

mkdir -p "$(dirname "$BASE_CSV")" "$HEART_DIR" corazon signals

[[ -f "$OHLC4H" ]] || { echo "[ERR] Falta $OHLC4H"; exit 1; }

export CORAZON_EXPORT_BARS=1
python swing_4h_forward_diamond.py --skip_yf \
  --symbol "$ASSET" --period ${PERIOD_DAYS}d --horizons "$HORIZONS" \
  --freeze_end "$FREEZE_TS" --slip "$SLIP" --cost "$COST" \
  --out_csv "$BASE_CSV"

# alias si el exporter dejó week1_bars.csv
if [[ ! -f "$BASE_BARS" && -f reports/diamante_btc_costes_week1_bars.csv ]]; then
  ln -sf diamante_btc_costes_week1_bars.csv "$BASE_BARS"
fi
[[ -f "$BASE_BARS" ]] || { echo "[ERR] No existe $BASE_BARS"; exit 1; }

# reglas: usa snapshot si existe, si no default
RULES="$RULES_SNAP"
[[ -f "$RULES" ]] || RULES="$RULES_DEFAULT"
[[ -f "$RULES" ]] || { echo "[ERR] Faltan reglas: $RULES_SNAP o $RULES_DEFAULT"; exit 1; }

python scripts/corazon_weights_generator.py \
  --rules "$RULES" \
  --ohlc  "$OHLC4H" \
  --diamante signals/diamante.csv \
  --perla    signals/perla.csv \
  --out_weights "$WEIGHTS_OUT" \
  --out_lq     "$LQ_OUT"

cp "$WEIGHTS_OUT" "${HEART_DIR}/w_diamante.csv"

# overlay
source scripts/corazon_cmds.zsh
runH_apply_overlay "$BASE_BARS"

# KPIs (usa ts_col=timestamp para evitar parse_dates:'ts')
python scripts/report_heart_vs_baseline.py \
  --baseline_csv "$BASE_BARS" \
  --overlay_csv  "${HEART_DIR}/diamante_overlay_$(basename "$BASE_BARS")" \
  --out_md       "$OUT_SUMMARY_MD" \
  --out_csv      "$OUT_KPIS_CSV" \
  --ts_col timestamp

# ξ* + PASS/FAIL + bitácora
python scripts/heart_log_xi.py "$OUT_KPIS_CSV" "$FREEZE_DATE" "$XI_LOG"

# Mostrar resultados
echo "— KPIs del FREEZE —"
if command -v column >/dev/null 2>&1; then
  column -t -s, "$OUT_KPIS_CSV" || true
else
  cat "$OUT_KPIS_CSV" || true
fi
[[ -f "$XI_LOG" ]] && tail -n 1 "$XI_LOG" || true

echo "[OK] FREEZE sombra ${FREEZE_DATE} completado"
