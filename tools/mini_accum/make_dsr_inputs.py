#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, glob, os, re
import pandas as pd
from pathlib import Path

RE_RET_FN = re.compile(r"ret__([^/]+)__([^/]+)\.csv$", re.IGNORECASE)

def load_returns(ret_glob: str) -> pd.DataFrame:
    p = Path(ret_glob)
    if p.is_file():
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        assert {"config_id","window","ret"}.issubset(set(cols)), \
            "Single CSV must have columns: config_id, window, ret"
        out = df[[cols["config_id"], cols["window"], cols["ret"]]].copy()
        out.columns = ["config_id","window","ret"]
        return out

    files = sorted(glob.glob(ret_glob))
    if not files:
        raise FileNotFoundError(f"No returns found for pattern: {ret_glob}")
    rows = []
    for f in files:
        base = os.path.basename(f)
        m = RE_RET_FN.search(base)
        if not m:
            print(f"[WARN] skip non-matching returns: {base}")
            continue
        cfg, win = m.group(1), m.group(2)
        if not win.startswith("WF_"):
            win = f"WF_{win}"
        df = pd.read_csv(f)
        lc = {c.lower(): c for c in df.columns}
        if "ret" not in lc:
            print(f"[WARN] no 'ret' col in: {base}")
            continue
        r = df[[lc["ret"]]].copy()
        r["config_id"] = cfg
        r["window"] = win
        r = r[["config_id","window","ret"]]
        rows.append(r)
    if not rows:
        raise ValueError("No valid returns files parsed.")
    out = pd.concat(rows, ignore_index=True)
    out["ret"] = pd.to_numeric(out["ret"], errors="coerce")
    out = out.dropna(subset=["ret"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ret_glob", required=True, help="e.g., reports/mini_accum/kiss_v1/ret__*.csv or a single CSV")
    ap.add_argument("--summary_in", required=True)
    ap.add_argument("--summary_out", required=True)
    args = ap.parse_args()

    rets = load_returns(args.ret_glob)
    agg = rets.groupby(["config_id","window"])["ret"].agg(ret_mean="mean", ret_std="std", n_obs="count").reset_index()

    summ = pd.read_csv(args.summary_in)
    merged = pd.merge(summ, agg, on=["config_id","window"], how="left")
    merged.to_csv(args.summary_out, index=False)
    print("[OK] DSR inputs merged into", args.summary_out)

if __name__ == "__main__":
    main()