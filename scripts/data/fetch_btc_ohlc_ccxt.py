import ccxt, pandas as pd, time, os, sys
from datetime import datetime, timezone

def fetch_range(ex, symbol, timeframe, since_ms, until_ms, limit=1000, sleep_s=0.35):
    out, ms = [], since_ms
    while ms < until_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=ms, limit=limit)
        if not batch: break
        out.extend(batch)
        last = batch[-1][0]
        ms = last + 1  # evita bucles por vela repetida
        time.sleep(sleep_s)
    return out

def write_csv(rows, out_csv):
    if not rows: raise SystemExit(f"Sin data para {out_csv}")
    df = pd.DataFrame(rows, columns=["ts_ms","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df[["ts","open","high","low","close","volume"]]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv} rows={len(df)}")

def main(years):
    for year in years:
        for ex_name, sym in [("binance","BTC/USDT"), ("coinbase","BTC/USD")]:
            try:
                ex = getattr(ccxt, ex_name)()
                since = int(datetime(year,1,1,tzinfo=timezone.utc).timestamp()*1000)
                until = int(datetime(year+1,1,1,tzinfo=timezone.utc).timestamp()*1000)
                print(f"[FETCH] {ex_name} {year} 4h…")
                r4 = fetch_range(ex, sym, "4h", since, until)
                write_csv(r4, f"data/tmp_wf/BTC-USD_4h_WF_{year}.csv")
                print(f"[FETCH] {ex_name} {year} 1d…")
                r1 = fetch_range(ex, sym, "1d", since, until)
                write_csv(r1, f"data/tmp_wf/BTC-USD_1d_WF_{year}.csv")
                break
            except Exception as e:
                print(f"[WARN] {ex_name} {year}: {e} -> pruebo otro exchange…")

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Uso: python scripts/data/fetch_btc_ohlc_ccxt.py 2022 [2023 ...]")
        sys.exit(1)
    main([int(x) for x in sys.argv[1:]])
