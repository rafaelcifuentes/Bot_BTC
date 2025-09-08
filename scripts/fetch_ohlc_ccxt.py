# scripts/fetch_ohlc_ccxt.py
import ccxt, sys, pandas as pd, math, time
from datetime import datetime, timezone

ex = ccxt.binanceus({'enableRateLimit': True})
symbol   = sys.argv[1]       # e.g. BTC/USD
tf       = sys.argv[2]       # e.g. 1m
since_s  = sys.argv[3]       # e.g. 2022-05-15
until_s  = sys.argv[4]       # e.g. 2024-07-15
out_path = sys.argv[5]       # e.g. data/ohlc/1m/BTC-USD.csv

ms = lambda s: int(pd.Timestamp(s, tz='UTC').timestamp()*1000)
since = ms(since_s); until = ms(until_s)

all_ = []
cursor = since
limit = 1500  # principio de paginación
while cursor < until:
    batch = ex.fetch_ohlcv(symbol, timeframe=tf, since=cursor, limit=limit)
    if not batch:
        break
    all_.extend(batch)
    cursor = batch[-1][0] + ex.parse_timeframe(tf)*1000
    time.sleep(ex.rateLimit/1000)

df = pd.DataFrame(all_, columns=['ts','open','high','low','close','volume'])
df = df.drop_duplicates('ts').sort_values('ts')
# Relleno de posibles huecos exactos al marco 1m (opcional si necesitas continuidad estricta):
rng = pd.date_range(start=pd.to_datetime(since, unit='ms', utc=True),
                    end=pd.to_datetime(until, unit='ms', utc=True),
                    freq='1min')
df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
df = df.set_index('ts').reindex(rng, method='pad').rename_axis('ts').reset_index()
df.to_csv(out_path, index=False)
print('WROTE', out_path, len(df))