#!/usr/bin/env python3
import argparse, time, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

import ccxt

def to_ms(s):  # 'YYYY-MM-DD' -> ms UTC
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()*1000)

def write_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.sort_index().to_csv(path, index_label="timestamp")
    # resumen
    try:
        with open(path) as f:
            next(f)
            first = next(f).split(",")[0]
            last = first
            for line in f: last = line.split(",")[0]
        print(f"[OK] {path}  MIN={first}  MAX={last}")
    except Exception:
        print(f"[OK] {path} filas={len(df)}")

def resample_ohlc(h1):
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    h4 = h1.resample("4h", label="left", closed="left").agg(agg).dropna(how="any")
    d1 = h1.resample("1d", label="left", closed="left").agg(agg).dropna(how="any")
    return h4, d1

def fetch_ccxt_1h(xchg_name, symbol, since_ms, end_ms, tf="1h", limit=1000, pause=0.2):
    try:
        xcls = getattr(ccxt, xchg_name)
        ex = xcls({"enableRateLimit": True})
        if symbol not in ex.load_markets():
            return None
        rows = []
        cursor = since_ms
        while cursor < end_ms:
            batch = ex.fetch_ohlcv(symbol, timeframe=tf, since=cursor, limit=limit)
            if not batch:
                break
            rows.extend(batch)
            last = batch[-1][0]
            nxt = last + 60*60*1000  # +1h
            if nxt <= cursor:
                cursor += 60*60*1000
            else:
                cursor = nxt
            if cursor >= end_ms:
                break
            time.sleep(pause)
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["ms","open","high","low","close","volume"]).drop_duplicates("ms")
        df["timestamp"] = pd.to_datetime(df["ms"], unit="ms", utc=True)
        return df.set_index("timestamp")[["open","high","low","close","volume"]].sort_index()
    except Exception as e:
        print(f"[WARN] {xchg_name} falló: {e}")
        return None

def read_existing(path):
    try:
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        return df[["open","high","low","close","volume"]].dropna(how="any").sort_index()
    except Exception:
        return None

def union_with_disk(h4, d1, out4h, out1d):
    for new_df, path in [(h4,out4h),(d1,out1d)]:
        old = read_existing(path)
        if old is not None and not old.empty:
            merged = pd.concat([old, new_df]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            write_csv(merged, path)
        else:
            write_csv(new_df, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out4h", default="data/ohlc/4h/BTC-USD.csv")
    ap.add_argument("--out1d", default="data/ohlc/1d/BTC-USD.csv")
    a = ap.parse_args()

    s_ms, e_ms = to_ms(a.start), to_ms(a.end)
    # lista de exchanges y símbolos alternativos
    candidates = [
        ("kraken",   "BTC/USD"),
        ("coinbase", "BTC/USD"),
        ("bitstamp", "BTC/USD"),
        ("bitfinex", "BTC/USD"),
        ("bybit",    "BTC/USDT"),
        ("okx",      "BTC/USDT"),
    ]
    dfs = []
    for name, sym in candidates:
        print(f"[INFO] {name} {sym} 1h…")
        d = fetch_ccxt_1h(name, sym, s_ms, e_ms)
        if d is not None and not d.empty:
            print(f"[OK] {name}: {len(d)} filas  ({d.index.min()} → {d.index.max()})")
            dfs.append(d)
        else:
            print(f"[WARN] {name} vacío")
    if not dfs:
        print("[ERROR] Sin datos en ninguna fuente CCXT"); sys.exit(2)

    h1 = pd.concat(dfs).sort_index()
    h1 = h1[~h1.index.duplicated(keep="last")]
    h4, d1 = resample_ohlc(h1)
    union_with_disk(h4, d1, a.out4h, a.out1d)
    print("[OK] Backfill CCXT fusionado y unido con disco.")
if __name__ == "__main__":
    main()
