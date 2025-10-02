# scripts/mini_accum/recalc_fpy.py
import argparse, pandas as pd
from kpi_utils import compute_fpy, sanity_kpis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flips", required=True, help="CSV con flips ejecutados")
    ap.add_argument("--start", help="Fecha inicio (YYYY-MM-DD)")
    ap.add_argument("--end", help="Fecha fin (YYYY-MM-DD)")
    ap.add_argument("--write", help="Ruta opcional para escribir un CSV con el FPY calculado")
    args = ap.parse_args()

    df = pd.read_csv(args.flips)
    fpy = compute_fpy(df, start_ts=args.start, end_ts=args.end)
    sanity_kpis(fpy, len(df))

    print(f"FPY_calculado={fpy:.2f}")
    if args.write:
        out = pd.DataFrame([{"fpy": fpy, "flips": len(df), "start": args.start, "end": args.end}])
        out.to_csv(args.write, index=False)
        print(f"[OK] escrito: {args.write}")

if __name__ == "__main__":
    main()
