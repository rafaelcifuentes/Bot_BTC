#!/usr/bin/env python3
import argparse, pandas as pd, datetime as dt, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--src",   required=True, help="CSV *_plus.csv con columnas pf y win_rate")
    ap.add_argument("--out",   required=True, help="CSV de salida con columnas asset,ts,proba,side")
    ap.add_argument("--pf_min", type=float, default=1.5)
    ap.add_argument("--wr_min", type=float, default=0.60)
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.src)
    except Exception as e:
        print(f"[ERR] No pude leer {args.src}: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty or not {"pf","win_rate"}.issubset(df.columns):
        print(f"[ERR] {args.src} no tiene columnas esperadas (pf, win_rate) o está vacío.", file=sys.stderr)
        sys.exit(2)

    row = df.iloc[-1]
    pf = float(row["pf"])
    wr = float(row["win_rate"])
    # normaliza win_rate si viene en porcentaje (>1.0)
    if wr > 1.0: wr = wr / 100.0

    gate_ok = (pf >= args.pf_min) and (wr >= args.wr_min)
    proba = 1.0 if gate_ok else 0.0

    ts = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    out = pd.DataFrame([{"asset": args.asset, "ts": ts, "proba": proba, "side": "long", "source": args.src}])
    out.to_csv(args.out, index=False)
    print(f"[OK] {args.out} -> proba={proba:.2f} (pf={pf:.3f}, wr={wr:.3f})")

if __name__ == "__main__":
    main()
