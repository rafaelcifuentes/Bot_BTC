import pandas as pd
from pathlib import Path

h4 = pd.read_csv("data/ohlc/4h/BTC-USD.csv")
d1 = pd.read_csv("data/ohlc/1d/BTC-USD.csv")
ts4 = h4.columns[0]; ts1 = d1.columns[0]
h4[ts4] = pd.to_datetime(h4[ts4], utc=True, errors='coerce')
d1[ts1] = pd.to_datetime(d1[ts1], utc=True, errors='coerce')
h4 = h4.sort_values(ts4)
d1 = d1.sort_values(ts1)

# EMAs 4h
ema = lambda s, span: s.ewm(span=span, adjust=False).mean()
h4['ema21'] = ema(h4['close'].astype(float), 21)
h4['ema55'] = ema(h4['close'].astype(float), 55)
h4['trend_up'] = h4['ema21'] > h4['ema55']

# ADX14 (Wilder)
hi, lo, cl = [h4[c].astype(float) for c in ['high','low','close']]
up = hi.diff(); down = (-lo.diff(-1))
plus_dm = ((up > down) & (up > 0)).astype(float) * up.clip(lower=0)
minus_dm = ((down > up) & (down > 0)).astype(float) * down.clip(lower=0)
tr = pd.concat([(hi-lo).abs(), (hi-cl.shift(1)).abs(), (lo-cl.shift(1)).abs()], axis=1).max(axis=1)
period = 14
atr = tr.ewm(alpha=1/period, adjust=False).mean()
plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0,pd.NA))
minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0,pd.NA))
dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0.0)
h4['adx14'] = dx.ewm(alpha=1/period, adjust=False).mean()

# Macro D1: EMA200 (EWM; también probamos SMA200 por si tu runner la usa)
d1['ema200'] = ema(d1['close'].astype(float), 200)
d1['sma200'] = d1['close'].astype(float).rolling(200, min_periods=200).mean()
d = d1[[ts1,'close','ema200','sma200']].rename(columns={'close':'d_close'})
m = pd.merge_asof(h4[[ts4,'trend_up','adx14','close']].sort_values(ts4),
                  d.sort_values(ts1), left_on=ts4, right_on=ts1, direction='backward')
m['macro_ema'] = m['d_close'] > m['ema200']
m['macro_sma'] = m['d_close'] > m['sma200']

total = len(m)
def pct(x): 
    n = int(x.sum()); 
    return f"{n}/{total} = {100*n/total:.1f}%" if total else "0/0 = 0.0%"

print("Ventana:", h4[ts4].iloc[0], "→", h4[ts4].iloc[-1])
print("trend_up:", pct(m['trend_up']))
print("ADX>=25:", pct(m['adx14'] >= 25))
print("macro EMA200:", pct(m['macro_ema'].fillna(False)))
print("macro SMA200:", pct(m['macro_sma'].fillna(False)))
all_ema = m['trend_up'] & (m['adx14']>=25) & m['macro_ema'].fillna(False)
all_sma = m['trend_up'] & (m['adx14']>=25) & m['macro_sma'].fillna(False)
print("ALL (EMA200):", pct(all_ema))
print("ALL (SMA200):", pct(all_sma))
print(m.loc[all_ema, ts4].head(5).to_string(index=False))
