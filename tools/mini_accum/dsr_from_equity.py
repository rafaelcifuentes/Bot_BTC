
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcula ret_mean, ret_std, n_obs por (config_id, window) leyendo directamente
los CSVs de equity y los inserta en wf_summary_kpis.csv.

Robusto a nombres tipo:
  base_..._equity__WF_WF_2025H1_PT_G200_DD16_RB2_H30_BULL0.csv
donde la ventana y los parámetros no están en orden "clásico".
"""

import argparse, glob, os, re, sys
from pathlib import Path
import pandas as pd

# Ventana: captura la ÚLTIMA ocurrencia de 2025 / 2025H1, con o sin prefijo WF_
RE_WINDOW_ANY = re.compile(r"(?:WF_)?([0-9]{4}(?:H[12])?)", re.IGNORECASE)

# Extrae cada pieza de la config sin asumir orden
RE_DD   = re.compile(r"DD(\d+)", re.IGNORECASE)
RE_RB   = re.compile(r"RB(\d+)", re.IGNORECASE)
RE_H    = re.compile(r"H(\d+)",  re.IGNORECASE)
RE_G    = re.compile(r"G(\d+)",  re.IGNORECASE)
RE_BULL = re.compile(r"BULL(\d+)", re.IGNORECASE)

TIME_LIKE = {"ts","timestamp","time","date","datetime"}

def parse_window_from_name(base: str):
    hits = RE_WINDOW_ANY.findall(base)
    if not hits: return None
    last = hits[-1]
    return f"WF_{last}"

def parse_config_from_name(base: str):
    def g(rx):
        m = rx.search(base)
        return m.group(1) if m else None
    DD   = g(RE_DD)
    RB   = g(RE_RB)
    H    = g(RE_H)
    G    = g(RE_G)
    BULL = g(RE_BULL)
    if all([DD, RB, H, G, BULL]):
        return f"DD{DD}_RB{RB}_H{H}_G{G}_BULL{BULL}"
    return None

def find_cfg_win_inside_df(df: pd.DataFrame):
    lower = {c.lower(): c for c in df.columns}
    cfg = df[lower["config_id"]].iloc[0] if "config_id" in lower else None
    win = df[lower["window"]].iloc[0] if "window" in lower else None
    if isinstance(win, str) and not win.upper().startswith("WF_"):
        win = f"WF_{win}"
    return cfg, win

def detect_equity_col(df: pd.DataFrame, override: str | None):
    if override and override in df.columns:
        return override
    candidates = [
        "equity_btc","equity","nav_btc","nav","net_btc",
        "balance_btc","btc_equity","btc_balance","equityBTC","equity_value_btc"
    ]
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower and pd.api.types.is_numeric_dtype(df[lower[c.lower()]]):
            return lower[c.lower()]
    # heurística: primera numérica con varianza>0 que no sea timestamp
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in TIME_LIKE]
    if numeric:
        vs = [(c, pd.to_numeric(df[c], errors="coerce").var()) for c in numeric]
        vs = [(c,v) for c,v in vs if pd.notnull(v) and v>0]
        if vs:
            vs.sort(key=lambda x: x[1], reverse=True)
            return vs[0][0]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity_glob", required=True, help="p.ej. 'reports/mini_accum/kiss_v1/*equity*.csv'")
    ap.add_argument("--summary_in", required=True)
    ap.add_argument("--summary_out", required=True)
    ap.add_argument("--equity_col", default=None, help="override del nombre de columna de equity si hace falta")
    ap.add_argument("--min_points", type=int, default=3)
    args = ap.parse_args()

    files = sorted(glob.glob(args.equity_glob))
    if not files:
        print(f"[ERR] No equity files match: {args.equity_glob}")
        sys.exit(1)

    rows = []
    skipped = 0
    for f in files:
        base = os.path.basename(f)
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] skip (read): {base} -> {e}")
            skipped += 1; continue

        # Detect window/config primero por nombre, luego por columnas internas
        win = parse_window_from_name(base)
        cfg = parse_config_from_name(base)
        if not cfg or not win:
            cfg2, win2 = find_cfg_win_inside_df(df)
            cfg = cfg or cfg2
            win = win or win2
        if not cfg or not win:
            print(f"[WARN] skip (no cfg/win): {base}")
            skipped += 1; continue

        col = detect_equity_col(df, args.equity_col)
        if not col:
            print(f"[WARN] skip (no equity col): {base} ; cols={list(df.columns)}")
            skipped += 1; continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.size < args.min_points:
            print(f"[WARN] skip (too short): {base} size={s.size}")
            skipped += 1; continue

        ret = s.pct_change().dropna()
        if ret.empty:
            print(f"[WARN] skip (no returns): {base}")
            skipped += 1; continue

        rows.append({"config_id": cfg, "window": win,
                     "ret_mean": float(ret.mean()),
                     "ret_std":  float(ret.std(ddof=1)),
                     "n_obs":    int(ret.count())})

    if not rows:
        print("[ERR] No se pudo derivar ningún retorno. Revisa patrones y columnas.")
        sys.exit(2)

    agg = pd.DataFrame(rows).groupby(["config_id","window"], as_index=False)\
                            .agg({"ret_mean":"mean","ret_std":"mean","n_obs":"sum"})

    summ = pd.read_csv(args.summary_in)
    merged = pd.merge(summ, agg, on=["config_id","window"], how="left")
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.summary_out, index=False)
    print(f"[OK] DSR inputs añadidos a {args.summary_out} (filas con datos nuevos: {(~merged['ret_mean'].isna()).sum()})")

if __name__ == "__main__":
    main()
