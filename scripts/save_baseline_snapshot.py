#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, time, os, glob
from pathlib import Path
import pandas as pd

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # unifica nombres típicos que cambian entre outputs
    rename_map = {
        "asset": "symbol",
        "winrate": "wr", "win_rate": "wr", "win%": "wr",
        "n_trades": "trades", "num_trades": "trades",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    # fuerza numéricas si existen
    for c in ["pf","wr","trades","sortino","roi","mdd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def topN(path: Path, n: int = 3):
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    df = normalize_cols(df)

    # columnas a devolver si existen
    base_cols = [c for c in ["symbol","asset","horizon","threshold","pf","wr","trades","sortino","roi","mdd","_file"] if c in df.columns]
    if not base_cols:
        base_cols = list(df.columns)

    # criterio de ranking según disponibilidad
    rank_order = [c for c in ["pf","wr","trades","sortino","roi"] if c in df.columns]
    if rank_order:
        df = df.sort_values(rank_order, ascending=[False]*len(rank_order), na_position="last")
    return df[base_cols].head(n).to_dict(orient="records")

def expand_inputs(items):
    out = []
    for it in items:
        # permite rutas exactas y patrones glob
        matched = glob.glob(it)
        out.extend(matched if matched else [it])
    # dedup preservando orden
    seen = set(); dedup = []
    for p in out:
        if p not in seen:
            seen.add(p); dedup.append(p)
    return [Path(p) for p in dedup]

def autodiscover_reports():
    pats = [
        "reports/diamante_*_week1.csv",
        "reports/val_*p5050.csv",
        "reports/val_*_poda*.csv",
        "reports/val_*_selected*.csv",
        "reports/*.csv",
    ]
    return expand_inputs(pats)

def main():
    ap = argparse.ArgumentParser(description="Snapshot baseline (top-N por archivo CSV) para Diamante/validaciones.")
    ap.add_argument("--inputs", nargs="+", help="Lista de CSVs o globs (ej: reports/diamante_*_week1.csv). Si se omite, autodiscovery en reports/*.csv")
    ap.add_argument("--out_json", required=True, help="Ruta de salida JSON (ej: reports/baseline/diamante_week0.json)")
    ap.add_argument("--topn", type=int, default=3, help="Cantidad de filas top por archivo (default: 3)")
    args = ap.parse_args()

    files = expand_inputs(args.inputs) if args.inputs else autodiscover_reports()
    files = [p for p in files if p.suffix.lower() == ".csv" and p.exists()]

    snapshot = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_files": [str(p) for p in files],
        "topN": args.topn,
        "top_by_file": {}
    }

    for f in files:
        try:
            snapshot["top_by_file"][os.path.basename(str(f))] = topN(f, n=args.topn)
        except Exception as e:
            snapshot["top_by_file"][os.path.basename(str(f))] = {"error": str(e)}

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2, ensure_ascii=False)

    print(f"[baseline] escrito {out_path} con {len(snapshot['source_files'])} archivos")

if __name__ == "__main__":
    main()
