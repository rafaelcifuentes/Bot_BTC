#!/usr/bin/env bash
set -euo pipefail

D_IN="${1:-${FREEZE:-}}"
if [[ -z "$D_IN" ]]; then
  echo "Uso: $0 YYYY-MM-DD  (o export FREEZE='YYYY-MM-DD 00:00' y llamar sin args)"
  exit 1
fi

# Normaliza fecha/hora
if [[ "$D_IN" == *" "* ]]; then DATE="${D_IN%% *}"; FE="$D_IN"; else DATE="$D_IN"; FE="$D_IN 00:00"; fi

mkdir -p reports/heart corazon

BASE="reports/diamante_btc_costes_freeze_${DATE}_bars.csv"
OVER="reports/heart/diamante_overlay_diamante_btc_costes_freeze_${DATE}_bars.csv"
KPI="reports/heart/kpis_diamante_btc_costes_freeze_${DATE}_bars.csv"
MD="reports/heart/summary_diamante_btc_costes_freeze_${DATE}_bars.md"

echo "[1] Export baseline FREEZE → $BASE"
export CORAZON_EXPORT_BARS=1
EXCHANGE="${EXCHANGE:-binanceus}" python swing_4h_forward_diamond.py --skip_yf \
  --symbol BTC-USD --period 730d --horizons 30,60,90 \
  --freeze_end "$FE" \
  --slip 0.0002 --cost 0.0004 \
  --out_csv "reports/diamante_btc_costes_freeze_${DATE}.csv" || true

# Fallback si el emitter dejó week1_bars
if [[ ! -f "$BASE" ]]; then
  if [[ -f reports/diamante_btc_costes_week1_bars.csv ]]; then
    cp reports/diamante_btc_costes_week1_bars.csv "$BASE"
    echo "[alias] $BASE <- reports/diamante_btc_costes_week1_bars.csv"
  else
    echo "[ERR] No encontré baseline *_bars.csv para FREEZE ${DATE}"
    exit 2
  fi
fi

echo "[2] Generar pesos (usa snapshot del YAML si existe)"
RULES="configs/heart_rules_${DATE}.yaml"
[[ -f "$RULES" ]] || RULES="configs/heart_rules.yaml"

OHLC4H_PATH="${OHLC4H:-$PWD/data/ohlc/4h/BTC-USD.csv}"
python scripts/corazon_weights_generator.py \
  --rules  "$RULES" \
  --ohlc   "$OHLC4H_PATH" \
  --diamante signals/diamante.csv \
  --perla    signals/perla.csv \
  --out_weights "corazon/weights_${DATE}.csv" \
  --out_lq     "corazon/lq_${DATE}.csv"

cp "corazon/weights_${DATE}.csv" reports/heart/w_diamante.csv

echo "[3] Overlay"
python scripts/apply_heart_overlay.py "$BASE" \
  --weights_csv reports/heart/w_diamante.csv \
  --out_csv "$OVER"

echo "[4] KPIs"
python scripts/report_heart_vs_baseline.py \
  --baseline_csv "$BASE" \
  --overlay_csv  "$OVER" \
  --out_md  "$MD" \
  --out_csv "$KPI" \
  --ts_col timestamp

echo "[5] ξ* + PASS/FAIL (log → corazon/daily_xi.csv)"
python - <<PY
import pandas as pd
D="${DATE}"
k=pd.read_csv(f"reports/heart/kpis_diamante_btc_costes_freeze_{D}_bars.csv").iloc[0]
mdd_ratio=abs(float(k["mdd_base"]))/max(1e-12,abs(float(k["mdd_overlay"])))
vol_ratio=float(k["vol_base"])/max(1e-12,float(k["vol_overlay"]))
xi=min(mdd_ratio,vol_ratio)*0.85
pf_ok=float(k["pf_overlay"])>=0.90*float(k["pf_base"])
mdd_ok=abs(float(k["mdd_overlay"]))<=abs(float(k["mdd_base"]))
vol_ok=float(k["vol_overlay"])<=float(k["vol_base"])
status="PASS" if (pf_ok and mdd_ok and vol_ok) else "FAIL"
row=pd.DataFrame([{
 "ts":D,"pf_base":k["pf_base"],"pf_overlay":k["pf_overlay"],
 "mdd_base":k["mdd_base"],"mdd_overlay":k["mdd_overlay"],
 "vol_base":k["vol_base"],"vol_overlay":k["vol_overlay"],
 "net_base":k["net_base"],"net_overlay":k["net_overlay"],
 "mdd_ratio":mdd_ratio,"vol_ratio":vol_ratio,"xi_star":xi,"status":status
}])
try:
  dx=pd.read_csv("corazon/daily_xi.csv"); dx=dx[dx["ts"]!=D]; dx=pd.concat([dx,row],ignore_index=True)
except FileNotFoundError:
  dx=row
dx.to_csv("corazon/daily_xi.csv",index=False)
print(f"[{status}] KPIs FREEZE {D} | ξ*={xi:.4f}x")
PY

echo "[OK] Artefactos:"
ls -lh "$OVER" "$KPI" "$MD" || true
