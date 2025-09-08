import argparse, pandas as pd
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)   # reports/ohlc_4h/BTC-USD.csv
    ap.add_argument("--out_csv", required=True)  # reports/diamante_btc_week1_bars.csv
    args = ap.parse_args()
    df = pd.read_csv(args.in_csv)
    for c in ["timestamp","ts","date","datetime","time"]:
        if c in df.columns:
            ts = c; break
    else:
        raise ValueError(f"No timestamp column in {df.columns}")
    df[ts] = pd.to_datetime(df[ts], utc=True, errors="coerce")
    df = df.dropna(subset=[ts]).sort_values(ts)
    if "close" not in df.columns:
        raise ValueError("close column not found")
    df["ret_4h"] = df["close"].pct_change()
    out = df[[ts,"ret_4h"]].dropna().rename(columns={ts:"timestamp"})
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Dummy baseline escrito en {args.out_csv} con {len(out)} filas")
if __name__ == "__main__":
    main()
