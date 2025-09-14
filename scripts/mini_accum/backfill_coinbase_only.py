#!/usr/bin/env python3
import argparse, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd, ccxt

def to_ms(s): return int(datetime.strptime(s,"%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()*1000)
def write_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.sort_index().to_csv(path, index_label="timestamp")
    print(f"[OK] {path}  MIN={df.index.min()}  MAX={df.index.max()}  rows={len(df)}")

def fetch_coinbase_1h(symbol, since_ms, end_ms, limit=1000):
    ex = ccxt.coinbase({"enableRateLimit": True})
    ex.load_markets()
    rows, cursor = [], since_ms
    while cursor < end_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe="1h", since=cursor, limit=limit)
        if not batch: break
        rows += batch
        nxt = batch[-1][0] + 3600_000
        cursor = max(cursor + 3600_000, nxt)
    if not rows: return None
    df = pd.DataFrame(rows, columns=["ms","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["ms"], unit="ms", utc=True)
    return df.set_index("timestamp")[["open","high","low","close","volume"]].sort_index()

def resample(h1):
    agg={"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    h4 = h1.resample("4h", label="left", closed="left").agg(agg).dropna(how="any")
    d1 = h1.resample("1d", label="left", closed="left").agg(agg).dropna(how="any")
    return h4, d1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sym", default="BTC/USD")
    ap.add_argument("--out4h", default="data/ohlc_cb/4h/BTC-USD.csv")
    ap.add_argument("--out1d", default="data/ohlc_cb/1d/BTC-USD.csv")
    a=ap.parse_args()
    h1 = fetch_coinbase_1h(a.sym, to_ms(a.start), to_ms(a.end))
    if h1 is None or h1.empty:
        raise SystemExit("[ERROR] Coinbase vacío")
    h4, d1 = resample(h1)
    write_csv(h4, a.out4h)
    write_csv(d1, a.out1d)
    print("[OK] Coinbase-only listo.")
if __name__=="__main__": main()
