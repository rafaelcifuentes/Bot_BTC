#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import glob, pandas as pd
from math import prod
from pathlib import Path

def latest(pats):
    xs=[]
    for p in pats: xs+=glob.glob(p, recursive=True)
    return sorted(xs)[-1] if xs else None

def read_kpi(p):
    df=pd.read_csv(p)
    cols={c.lower():c for c in df.columns}
    g=lambda *ks: next((df[cols[k]].iloc[0] for k in ks if k in cols), None)
    return dict(file=Path(p).name,
                net=float(g('net_btc_ratio','net_btc')),
                mdd=float(g('mdd_vs_hodl_ratio','mdd_vs_hodl')),
                flips=g('flips_total','flips'))

P2022=latest(["reports/mini_accum/**/kpis__OOS_2022_E1_Y2*.csv","reports/mini_accum/**/kpis__OOS_2022_*REGIME*.csv"])
P2023=latest(["reports/mini_accum/**/kpis__OOS_2023*REGIME*.csv"])
P2024=latest(["reports/mini_accum/**/kpis__OOS_2024*REGIME*.csv"])
P2025=latest(["reports/mini_accum/**/kpis__OOS_2025H1*REGIME*.csv"])

found={}
if P2022: found["2022"]=("E1_Y2", read_kpi(P2022))
if P2023: found["2023"]=("V1 TOP", read_kpi(P2023))
if P2024: found["2024"]=("V1 TOP", read_kpi(P2024))
if P2025: found["2025H1"]=("V1 TOP", read_kpi(P2025))

print("| Año | Preset | sats_mult | mdd_vs_hodl | flips | KPI |")
print("|:--:|:--|--:|--:|--:|:--|")
for yr in ["2022","2023","2024","2025H1"]:
    if yr in found:
        preset, r = found[yr]
        fl = "" if r['flips'] is None else int(r['flips'])
        print(f"| {yr} | {preset} | {r['net']:.6f} | {r['mdd']:.6f} | {fl} | {r['file']} |")

vals=[found[k][1]['net'] for k in ["2022","2023","2024"] if k in found]
if vals:
    acc2024=prod(vals)
    print(f"\nBTC fin 2024 (desde 1 BTC): ~{acc2024:.6f} BTC")
    if "2025H1" in found:
        h1=found["2025H1"][1]['net']
        print(f"BTC fin H1-2025: ~{(acc2024*h1):.6f} BTC")
        print(f"BTC fin 2025 (neutral H2≈H1): ~{(acc2024*(h1**2)):.6f} BTC")
PY
