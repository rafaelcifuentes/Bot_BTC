#!/usr/bin/env python3
from __future__ import annotations
import argparse, time
from datetime import datetime, timezone
import pandas as pd

TF_MS = {
    "1m":60_000, "5m":300_000, "15m":900_000,
    "1h":3_600_000, "4h":14_400_000, "6h":21_600_000, "1d":86_400_000
}

def load_exchange(exchange_id: str):
    import ccxt
    cls = getattr(ccxt, exchange_id)
    ex = cls()
    ex.load_markets()
    return ex

def fetch_ohlcv_all(ex, symbol: str, timeframe: str, since_ms: int, limit: int = 1500):
    tfms = TF_MS[timeframe]
    out, last = [], since_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=last, limit=limit)
        if not batch: break
        out += batch
        next_since = batch[-1][0] + tfms
        if next_since <= last: break
        last = next_since
        time.sleep(getattr(ex, "rateLimit", 250)/1000)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", required=True, choices=list(TF_MS.keys()))
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    ex = load_exchange(args.exchange)
    since_ms = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp()*1000)

    rows = fetch_ohlcv_all(ex, args.symbol, args.timeframe, since_ms)
    if not rows: raise SystemExit("No se obtuvo data. Verifica exchange/símbolo/timeframe/fecha.")
    df = pd.DataFrame(rows, columns=["ms","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["ms"], unit="ms", utc=True)
    df = (df.drop(columns=["ms"]).sort_values("timestamp").drop_duplicates("timestamp"))
    df = df[["timestamp","open","high","low","close","volume"]]
    df.to_csv(args.outfile, index=False)
    print(f"[OK] Guardado → {args.outfile} ({len(df)} velas)")
    print(df.head(3))
if __name__ == "__main__":
    main()
