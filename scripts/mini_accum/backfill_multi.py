#!/usr/bin/env python3
import argparse, time, sys
from datetime import datetime, timezone
from pathlib import Path
import requests, pandas as pd

def to_unix(s): return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
def to_iso(ts): return datetime.utcfromtimestamp(ts).replace(tzinfo=timezone.utc).isoformat()

def write_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.sort_index().to_csv(path, index_label="timestamp")
    # resumen
    try:
        with open(path) as f:
            hdr = next(f, None)
            first = next(f).split(",")[0]
            last = first
            for line in f: last = line.split(",")[0]
        print(f"[OK] {path}  MIN={first}  MAX={last}")
    except Exception:
        print(f"[OK] {path}  filas={len(df)}")

def resample_ohlc(h1):
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    h4 = h1.resample("4h", label="left", closed="left").agg(agg).dropna(how="any")
    d1 = h1.resample("1d", label="left", closed="left").agg(agg).dropna(how="any")
    return h4, d1

# ----- Providers (1h) -----
def fetch_kraken_1h(s, e):
    url = "https://api.kraken.com/0/public/OHLC"
    since = s; rows=[]; tries=0
    while since < e and tries < 4000:
        tries += 1
        r = requests.get(url, params={"pair":"XBTUSD","interval":60,"since":since}, timeout=20)
        j = r.json()
        if j.get("error"): break
        res = j.get("result", {}); keys=[k for k in res if k!="last"]
        if not keys: break
        kp = keys[0]; data=res.get(kp,[])
        if not data: break
        for k in data:
            t=int(k[0]); 
            if t>=e: break
            rows.append((t, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[6])))
        since = int(res.get("last", since+3600))+1
        time.sleep(0.15)
    if not rows: return None
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"]).drop_duplicates("ts")
    df["timestamp"]=pd.to_datetime(df["ts"], unit="s", utc=True)
    return df.set_index("timestamp").drop(columns=["ts"]).sort_index()

def fetch_coinbase_1h(s, e):
    base="https://api.exchange.coinbase.com/products/BTC-USD/candles"
    step=300*3600; cur=s; rows=[]
    while cur < e:
        nxt=min(cur+step, e)
        r=requests.get(base, params={"granularity":3600,"start":to_iso(cur),"end":to_iso(nxt)}, timeout=20)
        if r.status_code!=200: break
        arr=r.json() or []
        for a in arr: # [ time, low, high, open, close, volume ]
            t=int(a[0])
            if t<cur or t>=nxt: continue
            rows.append((t,float(a[3]),float(a[2]),float(a[1]),float(a[4]),float(a[5])))
        cur=nxt; time.sleep(0.2)
    if not rows: return None
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"]).drop_duplicates("ts")
    df["timestamp"]=pd.to_datetime(df["ts"], unit="s", utc=True)
    return df.set_index("timestamp").drop(columns=["ts"]).sort_index()

def fetch_bitstamp_1h(s, e):
    url="https://www.bitstamp.net/api/v2/ohlc/btcusd/"; cur=s; rows=[]
    step=1000*3600
    while cur < e:
        nxt=min(cur+step, e)
        r=requests.get(url, params={"step":3600,"limit":1000,"start":cur,"end":nxt}, timeout=20)
        j=r.json(); data=(j.get("data",{}) or {}).get("ohlc",[])
        for k in data:
            t=int(k["timestamp"])
            if t<cur or t>=nxt: continue
            rows.append((t,float(k["open"]),float(k["high"]),float(k["low"]),float(k["close"]),float(k["volume"])))
        cur=nxt; time.sleep(0.2)
    if not rows: return None
    df=pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"]).drop_duplicates("ts")
    df["timestamp"]=pd.to_datetime(df["ts"], unit="s", utc=True)
    return df.set_index("timestamp").drop(columns=["ts"]).sort_index()

def merge_sources(dfs):
    dfs = [d for d in dfs if d is not None and not d.empty]
    if not dfs: return None
    df = pd.concat(dfs, axis=0).sort_index()
    # colapsar timestamps duplicados priorizando últimas filas
    df = df[~df.index.duplicated(keep="last")]
    return df

def read_existing(path):
    try:
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[["open","high","low","close","volume"]].dropna(how="any").sort_index()
        return df
    except Exception:
        return None

def union_with_disk(new4h, new1d, out4h, out1d):
    for new_df, path in [(new4h,out4h),(new1d,out1d)]:
        old = read_existing(path)
        if old is not None and not old.empty:
            merged = pd.concat([old, new_df]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            write_csv(merged, path)
        else:
            write_csv(new_df, path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out4h", default="data/ohlc/4h/BTC-USD.csv")
    ap.add_argument("--out1d", default="data/ohlc/1d/BTC-USD.csv")
    a=ap.parse_args()
    s,e = to_unix(a.start), to_unix(a.end)

    dfs=[]
    for name,fn in [("Kraken",fetch_kraken_1h),("Coinbase",fetch_coinbase_1h),("Bitstamp",fetch_bitstamp_1h)]:
        try:
            print(f"[INFO] {name} 1h…")
            d=fn(s,e)
            if d is not None and not d.empty:
                print(f"[OK] {name}: {len(d)} filas  ({d.index.min()} → {d.index.max()})")
                dfs.append(d)
            else:
                print(f"[WARN] {name} vacío")
        except Exception as ex:
            print(f"[WARN] {name} falló: {ex}")

    merged_h1 = merge_sources(dfs)
    if merged_h1 is None or merged_h1.empty:
        print("[ERROR] Sin datos en ninguna fuente."); sys.exit(2)

    h4,d1 = resample_ohlc(merged_h1)
    union_with_disk(h4, d1, a.out4h, a.out1d)
    print("[OK] Backfill fusionado y unido con disco.")
