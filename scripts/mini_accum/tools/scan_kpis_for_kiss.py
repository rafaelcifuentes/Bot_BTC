#!/usr/bin/env python3
import glob, csv, math
CAND_SATS = ["sats_mult","net_btc_mult","net_btc_ratio","sats_multiplier","btc_mult","roi_btc_mult","netBTC_mult"]
CAND_MDDR = ["mdd_vs_hodl","mdd_vs_hodl_ratio","mdd_ratio_vs_hodl","mdd_model_vs_hodl"]
CAND_FPY  = ["fpy","flips_per_year","flips/yr","fpy_est","fpy_oos","fpy_wf"]
def to_float(x):
    s = str(x).strip()
    return float(s[:-1])/100.0 if s.endswith('%') else float(s)
def pick(row, keys):
    for k in keys:
        if k in row and str(row[k]).strip() not in ("","nan","None"):
            try:
                v = to_float(row[k]); 
                if not math.isnan(v): return v
            except: pass
    return None
rows = []
for p in glob.glob("reports/mini_accum/*kpis*.csv"):
    try:
        with open(p, newline="") as fh:
            r = list(csv.DictReader(fh))
        if not r: continue
        row = r[-1]
        sats = pick(row, CAND_SATS)
        mddr = pick(row, CAND_MDDR)
        fpy  = pick(row, CAND_FPY)
        rows.append((sats if sats is not None else -1, mddr, fpy, p))
    except: 
        pass
rows.sort(key=lambda x: (x[0] if x[0] is not None else -1), reverse=True)
print("== TOP por sats_mult ==")
for sats, mddr, fpy, p in rows[:20]:
    print(f"{sats:>8.4f}  mdd_vs_hodl={mddr!s:>8}  fpy={fpy!s:>6}  | {p}")
