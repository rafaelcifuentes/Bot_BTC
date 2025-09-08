#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perla MVP grid (Donchian) — listo para el allocator
- Explora (up, dn), evalúa PF/NET/VOL/MDD/WR sin look-ahead
- Modos:
    * longflat  -> sP=+1 siempre, w_perla_raw in {0,1}
    * longshort -> sP in {-1,+1}, w_perla_raw=1.0 (sin 0s)
- Escribe:
    * reports/heart/perla_grid_results.csv  (ranking del grid)
    * signals/perla.csv                     (señal final)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import math

# ----------------- Utils -----------------
def ensure_utc_index(df: pd.DataFrame, ts_col="timestamp") -> pd.DataFrame:
    if ts_col in df.columns:
        idx = pd.to_datetime(df[ts_col], utc=True)
        df = df.drop(columns=[ts_col]); df.index = idx
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    return df[~df.index.duplicated(keep="last")].sort_index()

def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("4h").last().ffill()

def derive_returns(df: pd.DataFrame) -> pd.Series:
    # busca columnas de retornos
    for cand in ["retP_btc","retD_btc","ret","returns","r"]:
        if cand in df.columns:
            s = pd.to_numeric(df[cand], errors="coerce").fillna(0.0)
            if s.std() > 0:
                return s
    # deriva de precio si hay
    for px in ["close","price","px","Close","c"]:
        if px in df.columns:
            s = pd.to_numeric(df[px], errors="coerce")
            r = s.pct_change().fillna(0.0)
            if r.std() > 0:
                return r
    raise ValueError("No se pudieron derivar retornos (faltan ret* o close/price).")

def ann_vol_median_ewm(r: pd.Series, span=60, bars_per_year=6*365) -> float:
    return float(r.ewm(span=span, adjust=False).std().median() * math.sqrt(bars_per_year))

def max_dd(r: pd.Series) -> float:
    eq = (1 + r.fillna(0)).cumprod()
    peak = eq.cummax()
    dd = eq/peak - 1.0
    return float(dd.min()) if len(dd) else float("nan")

def profit_factor(r: pd.Series) -> float:
    g = r[r>0].sum(); l = -r[r<0].sum()
    return float(g/l) if l>0 else (float("inf") if g>0 else 0.0)

def kpis(r: pd.Series) -> dict:
    return dict(
        vol=ann_vol_median_ewm(r),
        mdd=max_dd(r),
        pf=profit_factor(r),
        wr=float((r>0).mean()),
        net=float((1+r).prod()-1)
    )

# ---------- Señal Donchian ----------
def _pick_close_like(df: pd.DataFrame) -> pd.Series | None:
    for name in ["close","price","px","Close","c"]:
        if name in df.columns:
            s = pd.to_numeric(df[name], errors="coerce")
            if s.notna().any():
                return s
    return None

def donchian_events(df: pd.DataFrame, up: int, dn: int) -> pd.Series:
    """
    Eventos base: +1 => entrada long, -1 => entrada short, 0 => nada.
    Si no hay OHLC ni close/price, sintetiza 'close' desde retornos (cumprod).
    """
    has_ohlc = {"high","low","close"}.issubset(df.columns)
    if has_ohlc:
        h = pd.to_numeric(df["high"], errors="coerce")
        l = pd.to_numeric(df["low"],  errors="coerce")
        c = pd.to_numeric(df["close"],errors="coerce")
    else:
        c_like = _pick_close_like(df)
        if c_like is None:
            # Fallback robusto: sintetizar “precio” desde retornos
            r = derive_returns(df)
            c = (1.0 + r).cumprod()
        else:
            c = c_like
        h, l = c, c

    # Canales usando histórico hasta t-1 (no look-ahead)
    ch_up = h.shift(1).rolling(up, min_periods=up).max()
    ch_dn = l.shift(1).rolling(dn, min_periods=dn).min()

    evt = pd.Series(0.0, index=c.index)
    evt[c > ch_up] =  1.0
    evt[c < ch_dn] = -1.0
    return evt

def make_position(evt: pd.Series, mode: str) -> pd.Series:
    """Convierte eventos (-1/0/+1) en posición stateful según modo."""
    pos = pd.Series(0.0, index=evt.index)
    cur = 0.0
    for t, e in evt.items():
        if e == 1.0:
            cur =  1.0
        elif e == -1.0:
            cur = -1.0 if mode == "longshort" else 0.0  # en longflat, -1 => flat
        # si e == 0 => mantener
        pos.loc[t] = cur
    if mode == "longshort":
        # evita 0s; rellena al arranque con +1 si aún 0
        pos = pos.replace(0.0, np.nan).ffill().fillna(1.0)
    return pos

def eval_no_lookahead(r_spot: pd.Series, pos: pd.Series) -> pd.Series:
    """Retornos simulados sin look-ahead: usar pos.shift(1) * r."""
    return pos.shift(1).fillna(0.0) * r_spot.fillna(0.0)

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="Perla MVP Grid (Donchian)")
    ap.add_argument("--ohlc", required=True, help="CSV con timestamp y close/ret (puede ser signals/diamante.csv)")
    ap.add_argument("--freeze_end", default=None, help="YYYY-MM-DD (UTC); corta datos al final indicado")
    ap.add_argument("--mode", choices=["longflat","longshort"], default="longflat")
    ap.add_argument("--ups", default="20,30,40", help="lista up (ej: 20,30,40)")
    ap.add_argument("--dns", default="10,15,20", help="lista dn (ej: 10,15,20)")
    ap.add_argument("--out_csv", default="reports/heart/perla_grid_results.csv")
    ap.add_argument("--write_best", default="signals/perla.csv", help="ruta de la señal final")
    args = ap.parse_args()

    Path("reports/heart").mkdir(parents=True, exist_ok=True)
    Path("signals").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.ohlc)
    df = ensure_utc_index(df)
    df = resample_4h(df)
    r = derive_returns(df)

    if args.freeze_end:
        cutoff = pd.to_datetime(args.freeze_end, utc=True)
        df = df[df.index <= cutoff]; r = r.loc[df.index]

    ups = [int(x) for x in args.ups.split(",") if x.strip()]
    dns = [int(x) for x in args.dns.split(",") if x.strip()]

    rows = []
    best_key = None
    best_score = (-np.inf, -np.inf)  # (pf, net)

    for u in ups:
        for d in dns:
            evt = donchian_events(df, u, d)
            pos = make_position(evt, args.mode)
            ret = eval_no_lookahead(r, pos)

            m = kpis(ret)
            rows.append(dict(up=u, dn=d, **m))

            score = (m["pf"], m["net"])
            if score > best_score:
                best_score = score
                best_key = (u, d, pos)

    res = pd.DataFrame(rows).sort_values(["pf","net"], ascending=[False, False])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out_csv, index=False)
    print("[OK] grid →", args.out_csv)
    print(res.head(10).to_string(index=False))

    # --- escribir señal final para el allocator ---
    if best_key is not None:
        u, d, pos = best_key
        idx = pos.index

        if args.mode == "longflat":
            sP = pd.Series(1.0, index=idx)                 # dirección fija +1
            w_perla_raw = (pos > 0).astype(float)          # on/off
        else:  # longshort
            sP = np.sign(pos).replace(0.0, 1.0)            # ±1, sin 0s
            w_perla_raw = pd.Series(1.0, index=idx)        # magnitud fija 1

        out = pd.DataFrame({
            "timestamp": idx,
            "sP": sP.values,
            "w_perla_raw": w_perla_raw.values,
            "retP_btc": r.reindex(idx).fillna(0.0).values
        })
        out.to_csv(args.write_best, index=False)
        print(f"[OK] {args.write_best} escrito con la mejor config. (up={u}, dn={d}, mode={args.mode})")

if __name__ == "__main__":
    main()