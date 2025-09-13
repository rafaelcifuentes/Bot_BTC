#!/usr/bin/env bash
set -euo pipefail

CFG="configs/mini_accum/config.yaml"

# Construye un sufijo base a partir del YAML actual (dw, xbuf, mbuf y, si aplica, ATR p/yb)
SUFBASE="$(python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/mini_accum/config.yaml"))
dw = c.get('anti_whipsaw',{}).get('dwell_bars_min_between_flips')
x  = c.get('signals',{}).get('cross_buffer_bps', 0)
m  = c.get('macro_buffer_bps', None)
atr = (c.get('modules',{}) or {}).get('atr_regime', c.get('atr_regime',{}))
atr_enabled = bool((atr or {}).get('enabled', False))
p  = (atr or {}).get('percentile_p')
yb = (atr or {}).get('yellow_band_pct')

parts = ["bb1-dynATR", f"dw{dw}", f"xbuf{x}", f"mbuf{(m if m is not None else 'NA')}"]
if atr_enabled and p is not None and yb is not None:
    parts.insert(1, f"p{p}")
    parts.insert(2, f"yb{yb}")
print("-".join(parts))
PY
)"

run_oos () {  # run_oos <WIN> <START> <END>
  local WIN="$1"; local START="$2"; local END="$3"
  local SUFFIX="${SUFBASE}-oos${WIN}"
  echo "== OOS ${WIN} (${START}..${END}) =="
  REPORT_SUFFIX="${SUFFIX}" mini-accum-backtest --config "${CFG}" --start "${START}" --end "${END}"
}

# Ventanas OOS
run_oos 23Q4 2023-10-01 2023-12-31
run_oos 24H1 2024-01-01 2024-06-30

# Log KPIs (append + dedupe por suffix)
python - <<'PY'
import os, glob, pandas as pd
log = "experiments_log.csv"

def latest_kpi(win):
    files = sorted(glob.glob(f"reports/mini_accum/*_kpis__*oos{win}.csv"))
    return files[-1] if files else None

rows = []
for win in ("23Q4", "24H1"):
    fp = latest_kpi(win)
    if not fp:
        print(f"[WARN] No KPI para {win}")
        continue
    k = pd.read_csv(fp)
    suf = os.path.basename(fp).split("__",1)[-1].replace("_kpis.csv","")
    k['suffix'] = suf
    k['window'] = win
    rows.append(k[['suffix','net_btc_ratio','mdd_model_usd','mdd_hodl_usd',
                   'mdd_vs_hodl_ratio','flips_total','flips_per_year','window']])

if rows:
    df = pd.concat(rows, ignore_index=True)
    if os.path.exists(log):
        base = pd.read_csv(log)
        out = pd.concat([base, df], ignore_index=True)
        out = out.drop_duplicates(subset=['suffix'], keep='last')
    else:
        out = df
    out.to_csv(log, index=False)
    print(f"[OK] {log} actualizado (+{len(df)} filas nuevas, deduplicado por suffix)")
else:
    print("[WARN] Nada que loguear")
PY