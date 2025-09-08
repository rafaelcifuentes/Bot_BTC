#!/usr/bin/env python3
# scripts/gen_proxy_signals.py
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Devuelve un Timestamp tz-aware en UTC."""
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

def parse_windows(win_list):
    out = []
    for w in win_list:
        try:
            name, start, end = w.split(":")
        except ValueError:
            raise ValueError(f"Formato inválido: {w} (usa NAME:YYYY-MM-DD:YYYY-MM-DD)")
        out.append((name, to_utc(pd.Timestamp(start)), to_utc(pd.Timestamp(end))))
    return out

def ensure_utc_series(s: pd.Series) -> pd.Series:
    """Asegura que la serie de fechas es tz-aware UTC (sin convertir valores)."""
    if getattr(s.dt, "tz", None) is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")

def ensure_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")

def build_proba_from_momentum(df_1m: pd.DataFrame, resample_rule: str = "4h", lookback: int = 20) -> pd.DataFrame:
    """Resample a marco swing y crea proxy de probabilidad."""
    g = df_1m.set_index("ts").resample(resample_rule).last().dropna()
    g.index = ensure_utc_index(g.index)
    ret = g["close"].pct_change()
    z = (ret - ret.rolling(lookback).mean()) / (ret.rolling(lookback).std() + 1e-12)
    proba = (1.0 / (1.0 + np.exp(-z))).fillna(0.5).clip(0, 1)
    return pd.DataFrame({"ts": g.index, "proba": proba.values})

def main():
    ap = argparse.ArgumentParser(description="Genera señales proxy (proba) por ventana a partir de OHLC 1m.")
    ap.add_argument("--ohlc_csv", required=True)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--windows", required=True, nargs="+", help='NAME:START:END')
    ap.add_argument("--out_root", default="reports/windows")
    ap.add_argument("--resample", default="4h")
    ap.add_argument("--lookback", type=int, default=20)
    args = ap.parse_args()

    if not os.path.exists(args.ohlc_csv):
        raise FileNotFoundError(f"No existe {args.ohlc_csv}")

    ohlc = pd.read_csv(args.ohlc_csv, parse_dates=["ts"])
    if "ts" not in ohlc.columns or "close" not in ohlc.columns:
        raise ValueError("OHLC inválido: requiere columnas ts y close")
    ohlc["ts"] = ensure_utc_series(pd.to_datetime(ohlc["ts"], utc=True))
    ohlc = ohlc.sort_values("ts").reset_index(drop=True)

    windows = parse_windows(args.windows)

    for name, start, end in windows:
        sub = ohlc[(ohlc["ts"] >= start) & (ohlc["ts"] <= end)].copy()
        out_dir = Path(args.out_root) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{args.asset}.csv"

        if sub.empty:
            print(f"[WARN] Ventana {name} sin datos OHLC. Escribo CSV vacío.")
            pd.DataFrame(columns=["ts", "proba"]).to_csv(out_file, index=False)
            continue

        sig = build_proba_from_momentum(sub, args.resample, args.lookback)
        sig.to_csv(out_file, index=False)
        print(f"[OK] {out_file} rows={len(sig)}")

if __name__ == "__main__":
    main()