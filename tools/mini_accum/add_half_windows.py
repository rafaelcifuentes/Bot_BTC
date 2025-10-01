#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera filas adicionales para wf_summary_kpis.csv dividiendo ventanas anuales en H1/H2
usando CSVs de equity. Calcula: sats_mult, mdd_vs_hodl, flips_total, fpy y run_ok.

Asume columnas típicas en equity CSV:
- ts, model_equity_btc, hodl_equity (si hodl_equity no es BTC, igual sirve para ratio MDD)
- executed (0/1) opcional para contar flips.
"""

import argparse, glob, os, re
from pathlib import Path
import pandas as pd
import numpy as np

RE_YEAR = re.compile(r"(?:WF_)?(20\d{2})(?:H[12])?", re.IGNORECASE)
RE_DD   = re.compile(r"DD(\d+)", re.IGNORECASE)
RE_RB   = re.compile(r"RB(\d+)", re.IGNORECASE)
RE_H    = re.compile(r"H(\d+)",  re.IGNORECASE)
RE_G    = re.compile(r"G(\d+)",  re.IGNORECASE)
RE_BULL = re.compile(r"BULL(\d+)", re.IGNORECASE)

def parse_year_from_name(base: str):
    m = RE_YEAR.findall(base)
    return int(m[-1]) if m else None

def parse_cfg_from_name(base: str):
    def g(rx):
        m = rx.search(base)
        return m.group(1) if m else None
    DD, RB, H, G, BULL = g(RE_DD), g(RE_RB), g(RE_H), g(RE_G), g(RE_BULL)
    if all([DD, RB, H, G, BULL]):
        return f"DD{DD}_RB{RB}_H{H}_G{G}_BULL{BULL}"
    return None

def mdd(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return np.nan
    peak = s.cummax()
    dd = (peak - s) / peak.replace(0, np.nan)
    return float(dd.max())

def sats_mult_from_equity_btc(eq: pd.Series) -> float:
    x = pd.to_numeric(eq, errors="coerce").dropna()
    if len(x) < 2: return np.nan
    return float(x.iloc[-1] / x.iloc[0])

def count_flips(executed: pd.Series) -> int:
    if executed is None: return 0
    e = pd.to_numeric(executed, errors="coerce").fillna(0).astype(int)
    return int((e == 1).sum())

def half_bounds(year: int):
    return (
        (pd.Timestamp(year=year, month=1, day=1),  pd.Timestamp(year=year, month=6, day=30, hour=23, minute=59, second=59)),
        (pd.Timestamp(year=year, month=7, day=1),  pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)),
    )

def process_file(f, years_set):
    base = os.path.basename(f)
    y = parse_year_from_name(base)
    if y not in years_set:
        return []  # no hacemos nada con otros años
    cfg = parse_cfg_from_name(base)

    try:
        df = pd.read_csv(f)
    except Exception:
        return []
    cols = {c.lower(): c for c in df.columns}
    if "ts" not in cols:
        return []
    ts = pd.to_datetime(df[cols["ts"]], errors="coerce", utc=True).dt.tz_localize(None)
    me = df[cols.get("model_equity_btc","model_equity_btc")] if "model_equity_btc" in cols else None
    he = df[cols.get("hodl_equity","hodl_equity")] if "hodl_equity" in cols else None
    ex = df[cols.get("executed","executed")] if "executed" in cols else None

    if me is None or he is None:
        return []

    rows = []
    for (start, end), tag in zip(half_bounds(y), ("H1","H2")):
        mask = (ts >= start) & (ts <= end)
        if mask.sum() < 3:
            continue
        me_h = me[mask]
        he_h = he[mask]
        ex_h = ex[mask] if ex is not None else None

        sats = sats_mult_from_equity_btc(me_h)
        mdd_model = mdd(me_h)
        mdd_hodl  = mdd(he_h)
        mdd_ratio = float(mdd_model / mdd_hodl) if (mdd_hodl and not np.isnan(mdd_hodl) and mdd_hodl>0) else np.nan
        flips = count_flips(ex_h)
        fpy = flips / 0.5  # medio año

        rows.append({
            "window": f"WF_{y}{tag}",
            "config_id": cfg if cfg else "",
            "sats_mult": sats,
            "mdd_vs_hodl": mdd_ratio,
            "fpy": fpy,
            "flips_total": flips,
            "run_ok": bool((sats > 1.0) and (flips > 0)),
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity_glob", required=True, help="p.ej. 'reports/mini_accum/kiss_v1/*equity*.csv'")
    ap.add_argument("--summary_in", required=True)
    ap.add_argument("--summary_out", required=True)
    ap.add_argument("--years", default="2023,2024", help="años a dividir en H1/H2, coma-separado")
    args = ap.parse_args()

    years_set = set(int(x.strip()) for x in args.years.split(",") if x.strip().isdigit())
    files = sorted(glob.glob(args.equity_glob))
    if not files:
        print("[ERR] No equity files matched."); return

    new_rows = []
    for f in files:
        new_rows.extend(process_file(f, years_set))

    if not new_rows:
        print("[ERR] No se generaron subventanas. Revisa patrones y columnas.")
        return

    df_new = pd.DataFrame(new_rows).dropna(subset=["config_id","window","sats_mult"], how="any")
    df_old = pd.read_csv(args.summary_in)

    # Evita duplicados si ya corriste antes
    keys = set((r["config_id"], r["window"]) for _, r in df_new.iterrows())
    df_old = df_old[~df_old.apply(lambda r: (r["config_id"], r["window"]) in keys, axis=1)]

    merged = pd.concat([df_old, df_new], ignore_index=True)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.summary_out, index=False)
    print(f"[OK] Añadidas {len(df_new)} filas H1/H2 en {args.summary_out}")

if __name__ == "__main__":
    main()