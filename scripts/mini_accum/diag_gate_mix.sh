#!/usr/bin/env bash
set -Eeuo pipefail

need() { command -v "$1" >/dev/null 2>&1 || { echo "Falta $1"; exit 1; }; }
need yq
need python3

YAML="configs/mini_accum/config.yaml"
H4=$(yq -r '.data.ohlc_4h_csv' "$YAML")
D1=$(yq -r '.data.ohlc_d1_csv' "$YAML")
DW=$(yq -r '.anti_whipsaw.dwell_bars_min_between_flips' "$YAML")
ADX_MIN=$(yq -r '.filters.adx.min' "$YAML")

[[ -f "$H4" && -f "$D1" ]] || { echo "No encuentro $H4 o $D1"; exit 1; }

python3 - "$H4" "$D1" "$DW" "$ADX_MIN" <<'PY'
import sys, glob, math, os, io, csv
import pandas as pd

h4_path, d1_path, dwell_s, adx_min_s = sys.argv[1:5]
dwell = max(int(dwell_s), 1)
adx_min = float(adx_min_s)

def newest(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None

def sniff_sep(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[',',';','\t','|'])
        return dialect.delimiter
    except Exception:
        return ','

# -------- load data --------
h4 = pd.read_csv(h4_path)
d1 = pd.read_csv(d1_path)
ts4 = h4.columns[0]; ts1 = d1.columns[0]
for df, ts in [(h4, ts4), (d1, ts1)]:
    df[ts] = pd.to_datetime(df[ts], utc=True, errors='coerce')
    df.sort_values(ts, inplace=True)
    df.dropna(subset=[ts], inplace=True)

# -------- derived features --------
ema = lambda s, span: s.ewm(span=span, adjust=False).mean()
h4['ema21'] = ema(h4['close'].astype(float), 21)
h4['ema55'] = ema(h4['close'].astype(float), 55)
h4['trend_up'] = h4['ema21'] > h4['ema55']

hi, lo, cl = [h4[c].astype(float) for c in ['high','low','close']]
up = hi.diff(); down = (-lo.diff(-1))
plus_dm = ((up > down) & (up > 0)).astype(float) * up.clip(lower=0)
minus_dm = ((down > up) & (down > 0)).astype(float) * down.clip(lower=0)
tr = pd.concat([(hi-lo).abs(), (hi-cl.shift(1)).abs(), (lo-cl.shift(1)).abs()], axis=1).max(axis=1)
period = 14
atr = tr.ewm(alpha=1/period, adjust=False).mean()
plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0,pd.NA))
minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0,pd.NA))
dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0.0)
h4['adx14'] = dx.ewm(alpha=1/period, adjust=False).mean()

# Macro daily → 4h
d1['ema200'] = ema(d1['close'].astype(float), 200)
d = d1[[ts1, 'close', 'ema200']].rename(columns={'close':'d_close'})
m = pd.merge_asof(h4[[ts4, 'trend_up', 'adx14', 'close']].sort_values(ts4),
                  d.sort_values(ts1), left_on=ts4, right_on=ts1, direction='backward')
m['macro_green'] = m['d_close'] > m['ema200']

T = len(m)
get_pct = lambda s: (int(s.sum()), T, (100*int(s.sum())/T if T else 0.0))
_, _, p_tr = get_pct(m['trend_up'])
_, _, p_mg = get_pct(m['macro_green'].fillna(False))
_, _, p_ad = get_pct(m['adx14'] >= adx_min)
all_mask = m['trend_up'] & (m['adx14'] >= adx_min) & m['macro_green'].fillna(False)
N_all, _, p_all = get_pct(all_mask)

# barras/año y FPY teórico
if T >= 2:
    span_days = (m[ts4].iloc[-1] - m[ts4].iloc[0]).total_seconds()/86400.0
    bars_per_day = T/span_days if span_days>0 else 6.0
else:
    bars_per_day = 6.0
bars_per_year = bars_per_day * 365.25
fpy_theo = (bars_per_year * (N_all/T if T else 0.0)) / dwell

# -------- read latest KPIs / flips robustly --------
last_kpi = newest('reports/mini_accum/*_kpis*.csv')
last_flp = newest('reports/mini_accum/*_flips*.csv')
nb=mdd=fpy=float('nan'); fpy_drv=float('nan')
used_kpi = os.path.basename(last_kpi) if last_kpi else '-'
used_flp = os.path.basename(last_flp) if last_flp else '-'

if last_kpi:
    sep = sniff_sep(last_kpi)
    try:
        k = pd.read_csv(last_kpi, engine='python', sep=sep)
        if len(k) >= 1:
            row = k.iloc[-1]
            nb = float(row.get('net_btc_ratio', row.iloc[0]))
            mdd = float(row.get('mdd_vs_hodl_ratio', row.iloc[3]))
            fpy = float(row.get('flips_per_year', row.iloc[6]))
    except Exception:
        pass

if not (fpy==fpy) and last_flp:
    sep = sniff_sep(last_flp)
    try:
        fl = pd.read_csv(last_flp, engine='python', sep=sep)
        if len(fl) >= 2:
            tcol = fl.columns[0]
            fl[tcol] = pd.to_datetime(fl[tcol], utc=True, errors='coerce')
            fl = fl.dropna(subset=[tcol]).sort_values(tcol)
            n = len(fl)
            if n >= 2:
                span_days = (fl[tcol].iloc[-1] - fl[tcol].iloc[0]).total_seconds()/86400.0
                years = span_days/365.25 if span_days>0 else float('nan')
                fpy_drv = (n/years) if years==years and years>0 else float('nan')
    except Exception:
        pass

suffix = ""
if not (fpy==fpy) and (fpy_drv==fpy_drv):
    suffix = f"  (fpy≈{fpy_drv:.2f} derivado de flips)"

print(
  f"trend={p_tr:4.1f}%  macro={p_mg:4.1f}%  adx≥{adx_min:g}={p_ad:4.1f}%  "
  f"ALL={p_all:4.1f}%  dwell={dwell}  FPY_theo≈{fpy_theo:5.2f}  "
  f"[actual  netBTC={nb:.4f}  mdd_vs_HODL={mdd:.3f}  fpy={fpy:.2f}]" + suffix +
  f"  | kpi={used_kpi}  flips={used_flp}"
)
PY
