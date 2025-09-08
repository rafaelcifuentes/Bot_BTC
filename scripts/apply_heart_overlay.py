#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplica el overlay de Corazón sobre un baseline por-barra (Diamante).
- Entrada baseline: CSV con timestamp (o ts) y retorno base (ret_4h / ret / ret_base).
- Pesos: reports/heart/w_diamante.csv por defecto (timestamp, w_diamante).
- Salida overlay: reports/heart/diamante_overlay_<basename(baseline)>
- Métricas rápidas: MDD base vs overlay.
"""

import os, sys, argparse
import numpy as np
import pandas as pd

TS_CANDS = ("timestamp","ts")
RET_CANDS_BASE = ("ret_base","ret_4h","ret","return")

def _pick_col(cols, cands, name):
    for c in cands:
        if c in cols:
            return c
    raise ValueError(f"No encontré columna {name}. Busqué {cands} en {list(cols)}")

def _norm_ts(df, ts_col):
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    # UTC naive + rejilla 4h
    df[ts_col] = df[ts_col].dt.tz_convert(None).dt.floor("4h")
    return df

def _mdd_from_equity(equity: pd.Series) -> float:
    s = equity.astype(float)
    if s.empty:
        return 0.0
    roll = s.cummax()
    dd = (s - roll) / roll.replace(0, np.nan)
    dd = dd.fillna(0.0)
    return float(-dd.min())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_csv", help="CSV por-barra de Diamante (LIVE o FREEZE) con timestamp y retorno base.")
    ap.add_argument("--weights_csv", default="reports/heart/w_diamante.csv",
                    help="CSV de pesos (timestamp, w_diamante). Default: reports/heart/w_diamante.csv")
    ap.add_argument("--out_csv", default=None, help="Ruta de salida overlay. Si no, usa reports/heart/diamante_overlay_<basename(baseline)>.csv")
    args = ap.parse_args()

    # --- Baseline ---
    if not os.path.exists(args.baseline_csv):
        raise FileNotFoundError(f"Baseline no encontrado: {args.baseline_csv}")
    b = pd.read_csv(args.baseline_csv)
    ts_b = _pick_col(b.columns, TS_CANDS, "timestamp (baseline)")
    b = _norm_ts(b, ts_b)
    ret_col = None
    for cand in RET_CANDS_BASE:
        if cand in b.columns:
            ret_col = cand; break
    if ret_col is None:
        raise ValueError(f"No encontré columna de retorno en baseline. Busqué {RET_CANDS_BASE}")

    # --- Pesos ---
    if not os.path.exists(args.weights_csv):
        raise FileNotFoundError(f"Pesos no encontrados: {args.weights_csv}")
    w = pd.read_csv(args.weights_csv)
    ts_w = _pick_col(w.columns, TS_CANDS, "timestamp (weights)")
    # normaliza nombre w_diamante si viene distinto
    if "w_diamante" not in w.columns:
        # intenta detectar alguna columna numérica candidata
        cand_w = [c for c in w.columns if c.lower().startswith("w_")]
        if cand_w:
            w = w.rename(columns={cand_w[0]: "w_diamante"})
        else:
            raise ValueError("No encontré columna 'w_diamante' en pesos.")
    w = _norm_ts(w, ts_w)
    w = w[[ts_w, "w_diamante"]].dropna()
    w = w.sort_values(ts_w)

    # --- Merge asof ---
    b = b.sort_values(ts_b)
    m = pd.merge_asof(
        b, w.rename(columns={ts_w: ts_b}),
        on=ts_b, direction="nearest", tolerance=pd.Timedelta("2h")
    )
    # si faltan pesos en los extremos, usa 1.0 (no cap)
    m["w_diamante"] = m["w_diamante"].fillna(1.0).clip(lower=0.0, upper=1.0)

    # --- Overlay ---
    m["ret_base"] = m[ret_col].astype(float)
    m["ret_overlay"] = m["ret_base"] * m["w_diamante"]

    # Curvas (normalizadas a 1.0)
    m["eq_base"] = (1.0 + m["ret_base"].fillna(0.0)).cumprod()
    m["eq_overlay"] = (1.0 + m["ret_overlay"].fillna(0.0)).cumprod()

    # --- Salida ---
    out_csv = args.out_csv
    if not out_csv:
        base = os.path.basename(args.baseline_csv)
        name = os.path.splitext(base)[0]
        out_dir = "reports/heart"
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"diamante_overlay_{name}.csv")
    m.to_csv(out_csv, index=False)

    # --- Métricas rápidas ---
    mdd_b = _mdd_from_equity(m["eq_base"])
    mdd_o = _mdd_from_equity(m["eq_overlay"])
    print(f"[OK] Overlay escrito en {out_csv}")
    print(f"  MDD_base    : {mdd_b:.4%}")
    print(f"  MDD_overlay : {mdd_o:.4%}")

if __name__ == "__main__":
    main()
