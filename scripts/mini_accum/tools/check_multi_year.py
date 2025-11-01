#!/usr/bin/env python3
import argparse, json, glob, csv, math, sys

def first_num(row, keys):
    for k in keys:
        if k in row and str(row[k]).strip() not in ("","nan","None"):
            try:
                s = str(row[k]).strip()
                if s.endswith('%'): return float(s[:-1])/100.0
                return float(s)
            except: pass
    return math.nan

def scan(patterns):
    files=[]
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(set(files))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-map', required=True, help='JSON mapping año->NetBTC factor')
    ap.add_argument('--glob', action='append', required=True, help='Glob(s) para *_kpis__WF_* o *_OOS_*')
    ap.add_argument('--eps', type=float, default=0.001, help='ε de mejora (0.001=0.10%)')
    args = ap.parse_args()

    base = json.loads(args.base_map)
    files = scan(args.glob)
    if not files:
        print("[SKIP] no hay archivos que coincidan con los patrones")
        sys.exit(0)

    keys = ['sats_mult','net_btc_vs_hodl','net_btc_ratio','net_btc','netBTC','net_btc_oos']
    errs=[]
    for year, need_base in base.items():
        cand_files = [f for f in files if year in f]
        if not cand_files:
            print(f"[SKIP] {year}: sin KPI candidato")
            continue
        f = sorted(cand_files)[-1]
        with open(f, newline='') as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            print(f"[FAIL] {year}: CSV vacío {f}"); errs.append(year); continue
        cand = first_num(rows[-1], keys)
        if not (cand==cand):  # NaN
            print(f"[FAIL] {year}: KPI sin sats en {f}"); errs.append(year); continue
        req = float(need_base)*(1.0+args.eps)
        status = "PASS" if cand >= req else "FAIL"
        print(f"[{status}] {year}: cand={cand:.6f}  base={float(need_base):.6f}  req≥{req:.6f}  file={f}")
        if status=="FAIL": errs.append(year)

    sys.exit(1 if errs else 0)

if __name__ == "__main__":
    main()
