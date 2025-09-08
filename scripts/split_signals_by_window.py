#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import pandas as pd

def to_utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

def parse_windows(win_list):
    out=[]
    for w in win_list:
        try:
            name, s, e = w.split(":")
        except ValueError:
            raise ValueError(f"Formato inválido: {w} (usa NAME:YYYY-MM-DD:YYYY-MM-DD)")
        out.append((name, to_utc(s), to_utc(e)))
    return out

def read_master(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    # Normaliza el nombre de la columna de probabilidad
    cand = [c for c in df.columns if c.lower() in ("proba","score","p","signal","prob","probability")]
    if not cand:
        raise ValueError(f"No encuentro columna de probabilidad en {path}")
    col = cand[0]
    s = df.rename(columns={col:"proba"})[["ts","proba"]].copy()
    # Si es binaria (e.g., 0/1, True/False) la forzamos a float
    s["proba"] = s["proba"].astype(float)
    return s.sort_values("ts").reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description="Corta señal maestra en ventanas para el grid.")
    ap.add_argument("--master_csv", required=True, help="CSV con ts y proba/score/p/signal")
    ap.add_argument("--asset", required=True)
    ap.add_argument("--windows", required=True, nargs="+", help='NAME:START:END')
    ap.add_argument("--out_root", default="reports/windows")
    args = ap.parse_args()

    sig = read_master(args.master_csv)
    wins = parse_windows(args.windows)

    for name, s, e in wins:
        sub = sig[(sig["ts"]>=s) & (sig["ts"]<=e)].copy()
        out_dir = Path(args.out_root)/name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir/f"{args.asset}.csv"
        sub.to_csv(out_file, index=False)
        print(f"[OK] {out_file} rows={len(sub)}")

if __name__ == "__main__":
    main()
