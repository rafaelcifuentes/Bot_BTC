import argparse, time, math, sys
from datetime import datetime, timezone
from pathlib import Path
import csv

try:
    import requests
except Exception as e:
    print("[ERROR] Falta 'requests' (pip install requests)"); sys.exit(1)

def to_ms(dtstr):
    # 'YYYY-MM-DD' → ms UTC
    dt = datetime.strptime(dtstr, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def iso(ms):
    return datetime.utcfromtimestamp(ms/1000).replace(tzinfo=timezone.utc).isoformat()

def fetch_binance(symbol, interval, start_ms, end_ms, sleep=0.15):
    url = "https://api.binance.com/api/v3/klines"
    out = []
    cur = start_ms
    LIMIT = 1000
    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ms, "limit": LIMIT}
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"Binance {r.status_code}: {r.text[:120]}")
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        last_close_ms = batch[-1][6]  # close time
        # avanzar 1 ms después de close time para no pisar
        cur = last_close_ms + 1
        if cur > end_ms: break
        time.sleep(sleep)
    return out

def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","open","high","low","close","volume"])
        for k in rows:
            # kline format:
            # [0 openTime, 1 open, 2 high, 3 low, 4 close, 5 volume, 6 closeTime, ...]
            w.writerow([iso(k[0]), k[1], k[2], k[3], k[4], k[5]])

def summarize(path):
    try:
        with open(path) as f:
            next(f)  # header
            first = next(f).split(",")[0]
            for last in f: pass
        print(f"[OK] {path}  MIN={first}  MAX={last.split(',')[0]}")
    except Exception:
        print(f"[WARN] No puedo resumir {path}")

def try_yf(out4h, out1d, start, end):
    try:
        import yfinance as yf
        import pandas as pd
    except Exception:
        print("[WARN] yfinance/pandas no disponibles"); return False
    h1 = yf.download("BTC-USD", interval="1h", start=start, end=end, auto_adjust=False, progress=False)
    if h1 is None or h1.empty:
        print("[WARN] yfinance (1h) vacío"); return False
    h1 = h1.rename(columns=str.lower); h1.index.name="timestamp"
    h4 = h1.resample("4H").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(how="any")
    d1 = h1.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(how="any")
    for df,path in [(h4,out4h),(d1,out1d)]:
        df.to_csv(path)
        summarize(path)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)  # YYYY-MM-DD
    ap.add_argument("--end", required=True)
    ap.add_argument("--out4h", default="data/ohlc/4h/BTC-USD.csv")
    ap.add_argument("--out1d", default="data/ohlc/1d/BTC-USD.csv")
    a = ap.parse_args()

    s_ms, e_ms = to_ms(a.start), to_ms(a.end)
    try:
        print("[INFO] Binance 4h…")
        k4 = fetch_binance("BTCUSDT", "4h", s_ms, e_ms)
        if not k4: raise RuntimeError("Sin klines 4h")
        write_csv(a.out4h, k4); summarize(a.out4h)

        print("[INFO] Binance 1d…")
        k1 = fetch_binance("BTCUSDT", "1d", s_ms, e_ms)
        if not k1: raise RuntimeError("Sin klines 1d")
        write_csv(a.out1d, k1); summarize(a.out1d)
        print("[OK] Backfill Binance completado.")
    except Exception as be:
        print(f"[WARN] Binance falló: {be}. Probando yfinance…")
        ok = try_yf(a.out4h, a.out1d, a.start, a.end)
        if not ok:
            print("[ERROR] No pude backfillear con ninguna fuente."); sys.exit(2)

if __name__ == "__main__":
    main()
