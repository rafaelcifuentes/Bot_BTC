#!/usr/bin/env python3
# Cálculo de ξ* y log PASS/FAIL para Corazón sombra
import sys
import pandas as pd

if len(sys.argv) < 3:
    print("Uso: heart_log_xi.py <kpis_csv> <freeze_date YYYY-MM-DD> [xi_log_csv]")
    sys.exit(1)

kpis_csv = sys.argv[1]
date_str = sys.argv[2]
xi_log = sys.argv[3] if len(sys.argv) > 3 else "corazon/daily_xi.csv"

k = pd.read_csv(kpis_csv).iloc[0]
cols = {c.lower(): c for c in k.index}
get = lambda name: float(k[cols[name]])

mdd_base = abs(get('mdd_base'))
mdd_overlay = abs(get('mdd_overlay'))
vol_base = get('vol_base')
vol_overlay = get('vol_overlay')
pf_base = get('pf_base')
pf_overlay = get('pf_overlay')
net_base = get('net_base')
net_overlay = get('net_overlay')

EPS = 1e-12
mdd_ratio = (mdd_base + EPS) / max(EPS, mdd_overlay)
vol_ratio = (vol_base + EPS) / max(EPS, vol_overlay)
xi_star = min(mdd_ratio, vol_ratio) * 0.85

status = ("PASS" if (
    pf_overlay >= 0.90 * pf_base and
    mdd_overlay <= mdd_base and
    vol_overlay <= vol_base
) else "FAIL")

row = {
    "ts": date_str,
    "mdd_ratio": mdd_ratio,
    "vol_ratio": vol_ratio,
    "xi_star": xi_star,
    "status": status,
    "pf_base": pf_base, "pf_overlay": pf_overlay,
    "mdd_base": mdd_base, "mdd_overlay": mdd_overlay,
    "vol_base": vol_base, "vol_overlay": vol_overlay,
    "net_base": net_base, "net_overlay": net_overlay,
}

try:
    dx = pd.read_csv(xi_log)
    dx = dx[dx["ts"] != date_str]
    dx = pd.concat([dx, pd.DataFrame([row])], ignore_index=True)
except FileNotFoundError:
    dx = pd.DataFrame([row])

try:
    dx = dx.sort_values("ts").reset_index(drop=True)
except Exception:
    pass

dx.to_csv(xi_log, index=False)
print(f"[DONE] ξ*={xi_star:.4f}x | {status}")
