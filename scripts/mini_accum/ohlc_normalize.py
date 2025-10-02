# scripts/mini_accum/ohlc_normalize.py
import argparse, pandas as pd
from kpi_utils import normalize_ohlc_columns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="CSV OHLC de entrada")
    ap.add_argument("--out", dest="out", required=True, help="CSV de salida normalizado")
    ap.add_argument("--head", type=int, default=0, help="Muestra primeras N filas y sale")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    df2 = normalize_ohlc_columns(df)
    if args.head > 0:
        print(df2.head(args.head).to_string(index=False))
        return
    df2.to_csv(args.out, index=False)
    print(f"[OK] escrito: {args.out}")

if __name__ == "__main__":
    main()
