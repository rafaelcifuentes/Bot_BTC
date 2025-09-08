#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_diamante_shadow.py  —  genera una 'Diamante' sombra con señal dinámica
a partir de retD_btc cuando la original está plana (sD constante o w_diamante_raw std≈0).

- No modifica tu signals/diamante.csv
- Escribe reports/allocator/diamante_for_allocator.csv con columnas:
  timestamp, sD, w_diamante_raw, retD_btc
- sD se deriva de un crossover EMA(fast, slow) sobre la equity reconstruida
- w_diamante_raw ∈ [0,1] se deriva de la magnitud del momentum (|EMA_fast/EMA_slow - 1|)

Uso:
  python3 scripts/make_diamante_shadow.py \
    --diamante signals/diamante.csv \
    --out reports/allocator/diamante_for_allocator.csv \
    --ema_fast 20 --ema_slow 50 \
    --flat_band_bp 10 \
    --mom_scale 8.0 \
    [--force]  # reescribe aunque la señal original ya sea dinámica
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        idx = pd.to_datetime(df["timestamp"], utc=True)
        df = df.drop(columns=["timestamp"])
        df.index = idx
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

def derive_returns(df: pd.DataFrame) -> pd.Series:
    for col in ["retD_btc","ret","returns","r"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            if s.std() > 0:
                return s
    # Si no hay retornos, intenta con close/price
    for col in ["close","price","px","Close"]:
        if col in df.columns:
            px = pd.to_numeric(df[col], errors="coerce")
            r = px.pct_change().fillna(0.0)
            if r.std() > 0:
                return r
    return pd.Series(0.0, index=df.index)

def build_shadow(d: pd.DataFrame, ema_fast: int, ema_slow: int,
                 flat_band_bp: float, mom_scale: float):
    r = derive_returns(d)
    if r.std() == 0:
        raise RuntimeError("retD_btc/ret está plano: no se puede derivar señal dinámica.")

    # Reconstruye equity y EMAs
    eq = (1 + r).cumprod()
    ema_f = eq.ewm(span=ema_fast, adjust=False).mean()
    ema_s = eq.ewm(span=ema_slow, adjust=False).mean()

    # Momentum relativo y banda muerta (en basis points)
    mom = (ema_f / ema_s) - 1.0
    band = flat_band_bp / 10000.0
    sD = np.where(mom > band,  1,
         np.where(mom < -band, -1, 0)).astype(float)

    # Peso crudo según magnitud del momentum, recortado a [0,1]
    w_raw = np.clip(np.abs(mom) * mom_scale, 0.0, 1.0)

    out = pd.DataFrame({
        "sD": sD,
        "w_diamante_raw": w_raw,
        "retD_btc": r
    }, index=d.index)
    out = out.reset_index().rename(columns={"index":"timestamp"})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diamante", required=True, help="Ruta a signals/diamante.csv")
    ap.add_argument("--out", default="reports/allocator/diamante_for_allocator.csv")
    ap.add_argument("--ema_fast", type=int, default=20)
    ap.add_argument("--ema_slow", type=int, default=50)
    ap.add_argument("--flat_band_bp", type=float, default=10.0,
                    help="Banda muerta para sD en basis points (10 = 0.10%)")
    ap.add_argument("--mom_scale", type=float, default=8.0,
                    help="Escala de |momentum| → w_diamante_raw (0..1)")
    ap.add_argument("--force", action="store_true",
                    help="Reemplaza incluso si la señal original ya era dinámica")
    args = ap.parse_args()

    d = pd.read_csv(args.diamante)
    d = ensure_utc_index(d)

    # Detecta si la señal original ya es dinámica
    sD_const = (("sD" not in d.columns) or (d["sD"].nunique(dropna=False) <= 1))
    w_std = float(pd.to_numeric(d.get("w_diamante_raw", pd.Series(np.nan, index=d.index)),
                                errors="coerce").std(skipna=True))
    dynamic_already = (not sD_const) and (w_std > 1e-12)

    if dynamic_already and not args.force:
        print("[SKIP] Señal original ya parece dinámica (usa --force para sobreescribir).")
        out = d.reset_index().rename(columns={"index":"timestamp"})
        # Asegura columnas mínimas
        if "retD_btc" not in out.columns:
            out["retD_btc"] = derive_returns(d).values
    else:
        out = build_shadow(d, args.ema_fast, args.ema_slow, args.flat_band_bp, args.mom_scale)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, date_format="%Y-%m-%d %H:%M:%S%z")

    # Resumen
    vc = out["sD"].value_counts(dropna=False).to_dict()
    ch = int((out["sD"].shift(1) != out["sD"]).sum())
    print(f"[DONE] {args.out}")
    print(f" sD states: {vc} | cambios: {ch}")
    print(f" std(w_diamante_raw): {float(pd.to_numeric(out['w_diamante_raw'], errors='coerce').std()):.6f}")
    print(f" std(retD_btc): {float(pd.to_numeric(out['retD_btc'], errors='coerce').std()):.6f}")

if __name__ == "__main__":
    main()