#!/usr/bin/env bash
set -euo pipefail

# =========================
# Día 3 — Micro-grid (BTC/ETH)
# =========================
EXCHANGE="${EXCHANGE:-binanceus}"
FREEZE="${FREEZE:-2025-08-05 00:00}"
PERIOD="${PERIOD:-730d}"
HORIZONS="${HORIZONS:-30,60,90}"
THRESH="${THRESH:-0.60}"         # Fijado desde Día 2
OUTDIR="reports"

echo "[INFO] EXCHANGE=$EXCHANGE | FREEZE=\"$FREEZE\" | THRESH=$THRESH | HORIZONS=$HORIZONS"

assets=("BTC-USD" "ETH-USD")

# Micro-grid corta alrededor de ATR/TP/partial
# (24 corridas por activo: 3 x 2 x 2 x 2)
sl_list=(1.1 1.3 1.5)
tp1_list=(0.8 1.0)
tp2_list=(3.0 5.0)
partial_list=(0.50 0.70)

for sym in "${assets[@]}"; do
  base=$(echo "$sym" | cut -d- -f1 | tr '[:upper:]' '[:lower:]')
  for sl in "${sl_list[@]}"; do
    for tp1 in "${tp1_list[@]}"; do
      for tp2 in "${tp2_list[@]}"; do
        for pp in "${partial_list[@]}"; do
          tag="${base}_mg_sl${sl}_tp1${tp1}_tp2${tp2}_p$(printf '%.0f' "$(echo "$pp*100" | bc)")"
          out_csv="${OUTDIR}/diamante_${tag}_week1.csv"
          echo "[RUN] ${sym} sl=${sl} tp1=${tp1} tp2=${tp2} partial=${pp} -> ${out_csv}"
          EXCHANGE="$EXCHANGE" python swing_4h_forward_diamond.py --skip_yf \
            --symbol "$sym" --period "$PERIOD" --horizons "$HORIZONS" \
            --freeze_end "$FREEZE" --threshold "$THRESH" \
            --sl_atr_mul "$sl" --tp1_atr_mul "$tp1" --tp2_atr_mul "$tp2" --partial_pct "$pp" \
            --out_csv "$out_csv"
        done
      done
    done
  done
done

# === Resumen y selección por activo (pf_60d) ===
python - <<'PY'
import pandas as pd, glob, os, re
paths = glob.glob("reports/diamante_*_mg_*.csv")
rows = []
for p in paths:
    base = os.path.basename(p)
    m = re.match(r"diamante_(\w+)_mg_sl([0-9.]+)_tp1([0-9.]+)_tp2([0-9.]+)_p(\d+)_week1\.csv", base)
    if not m:
        continue
    asset, sl, tp1, tp2, pcent = m.groups()
    df = pd.read_csv(p)
    row60 = df[df['days']==60]
    if row60.empty:
        continue
    r = row60.iloc[0].to_dict()
    rows.append({
        "asset": asset.upper(),
        "sl_atr_mul": float(sl),
        "tp1_atr_mul": float(tp1),
        "tp2_atr_mul": float(tp2),
        "partial_pct": float(pcent)/100.0,
        "pf_60d": r.get("pf", float('nan')),
        "wr_60d": r.get("win_rate", float('nan')),
        "trades_60d": r.get("trades", float('nan')),
        "mdd_60d": r.get("mdd", float('nan')),
        "net_60d": r.get("net", float('nan')),
        "src": base
    })
summary = pd.DataFrame(rows).sort_values(["asset","pf_60d"], ascending=[True, False])
os.makedirs("reports", exist_ok=True)
summary.to_csv("reports/diamante_day3_microgrid_full.csv", index=False)
best = summary.groupby("asset").head(1).reset_index(drop=True)
best.to_csv("reports/diamante_day3_microgrid_best.csv", index=False)
print("=== TOP por activo (pf_60d) ===")
print(best.to_string(index=False))
PY

echo "[DONE] Resúmenes -> reports/diamante_day3_microgrid_{full,best}.csv"