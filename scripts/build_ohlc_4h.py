import argparse, pandas as pd, numpy as np

TS_CANDIDATES = ["timestamp","ts","date","datetime","time"]
O_CANDS = ["open","o"]; H_CANDS = ["high","h"]; L_CANDS = ["low","l"]
C_CANDS = ["close","c"]; V_CANDS = ["volume","vol","v"]

def pick_col(cols, cands, name):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low: return low[c]
    raise ValueError(f"Falta columna '{name}' en el CSV (tiene {list(cols)})")

def parse_ts(s):
    # numérico → epoch s/ms; texto → parse iso
    if np.issubdtype(s.dtype, np.number):
        arr = s.astype("int64")
        unit = "ms" if arr.max() > 10**12 else "s"
        return pd.to_datetime(arr, unit=unit, utc=True, errors="coerce")
    else:
        return pd.to_datetime(s, utc=True, errors="coerce")

def rma(series, length):
    # Wilder RMA aproximado con ewm alpha=1/length
    return series.ewm(alpha=1/length, adjust=False).mean()

def compute_adx_df(df, length=14, hi="high", lo="low", cl="close"):
    high = df[hi].astype(float); low = df[lo].astype(float); close = df[cl].astype(float)
    up   = high.diff()
    down = -low.diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()))
    atr = rma(pd.Series(tr, index=df.index), length)
    plus_di  = 100.0 * rma(pd.Series(plus_dm, index=df.index), length) / atr
    minus_di = 100.0 * rma(pd.Series(minus_dm, index=df.index), length) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = rma(dx, length)
    return adx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="CSV 1m local (cols: ts/timestamp, open, high, low, close, volume)")
    ap.add_argument("--out_csv", required=True, help="Salida 4h con ADX: reports/ohlc_4h/BTC-USD.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    ts_col = pick_col(df.columns, TS_CANDIDATES, "timestamp")
    o_col  = pick_col(df.columns, O_CANDS, "open")
    h_col  = pick_col(df.columns, H_CANDS, "high")
    l_col  = pick_col(df.columns, L_CANDS, "low")
    c_col  = pick_col(df.columns, C_CANDS, "close")
    v_col  = pick_col(df.columns, V_CANDS, "volume")

    df["timestamp"] = parse_ts(df[ts_col])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    df = df.rename(columns={o_col:"open", h_col:"high", l_col:"low", c_col:"close", v_col:"volume"})

    # 1m → 4h
    ohlc_4h = df.resample("4H").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna()

    # ADX 4h
    adx4 = compute_adx_df(ohlc_4h, length=14)
    # Daily → ADX1d
    ohlc_1d = df.resample("1D").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna()
    adx1 = compute_adx_df(ohlc_1d, length=14)

    # Alinear ADX1d a índices 4h (ffill)
    adx1_on_4h = adx1.reindex(ohlc_4h.index, method="ffill")

    out = ohlc_4h.copy()
    out["adx4h"] = adx4
    out["adx1d"] = adx1_on_4h
    out = out.reset_index().rename(columns={"index":"timestamp"})
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Escrito {args.out_csv} con columnas: {list(out.columns)}")

if __name__ == "__main__":
    main()
