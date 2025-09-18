#!/usr/bin/env python3
import os, sys
from pathlib import Path
import pandas as pd
import ccxt
import argparse

def fetch(exchange, symbol, timeframe, limit=1500):
    ex = getattr(ccxt, exchange)({'enableRateLimit': True})
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts_ms","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df[["ts","close"]].sort_values("ts")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="binanceus")
    ap.add_argument("--symbol",   default="BTC/USDT")
    ap.add_argument("--tf",       required=True, choices=["1d","4h"])
    ap.add_argument("--out",      required=True)
    args = ap.parse_args()

    df = fetch(args.exchange, args.symbol, args.tf, limit=1500)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[OK] {args.tf} -> {args.out}  last={df['ts'].iloc[-1]} close={df['close'].iloc[-1]}")

if __name__ == "__main__":
    raise SystemExit(main())
