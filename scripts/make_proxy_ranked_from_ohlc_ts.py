import argparse
from pathlib import Path
import pandas as pd
import numpy as np

WINDOWS = {
    "2022H2": ("2022-06-01","2023-01-31"),
    "2023Q4": ("2023-10-01","2023-12-31"),
    "2024H1": ("2024-01-01","2024-06-30"),
}

def pick_price_column(df):
    low = {c.lower(): c for c in df.columns}
    # candidatos típicos (yfinance, ccxt, csv genéricos)
    for key in ["close", "adj close", "adj_close", "adjclose", "c", "last", "price"]:
        if key in low:
            return low[key]
    # fallback: primera no-timestamp
    tcol = pick_time_column(df)
    for c in df.columns:
        if c != tcol:
            return c
    return df.columns[0]

def pick_time_column(df):
    low = {c.lower(): c for c in df.columns}
    for key in ["timestamp", "time", "date", "ts"]:
        if key in low:
            return low[key]
    return df.columns[0]

def zscore(s, win):
    m = s.rolling(win, min_periods=max(5, win//5)).mean()
    v = s.rolling(win, min_periods=max(5, win//5)).std()
    return (s - m) / v.replace(0, np.nan)

def pct_rank(series: pd.Series) -> pd.Series:
    # percentil global (0..1). Para evitar planicies, rompemos empates con un epsilon determinista.
    n = len(series)
    if n == 0:
        return series
    # epsilon determinista: pequeño ramp basado en la posición
    eps = np.linspace(0, 1e-9, n)
    tmp = pd.Series(series.values + eps, index=series.index)
    return tmp.rank(pct=True)

def make_proba_ranked(ohlc_csv: Path, start: str, end: str) -> pd.DataFrame:
    raw = pd.read_csv(ohlc_csv)
    tcol = pick_time_column(raw)
    pcol = pick_price_column(raw)

    ts = pd.to_datetime(raw[tcol], utc=True, errors="coerce")
    px = pd.to_numeric(raw[pcol], errors="coerce")
    df = pd.DataFrame({"timestamp": ts, "close": px}).dropna().set_index("timestamp").sort_index()
    df = df.loc[start:end]
    if df.empty:
        return pd.DataFrame(columns=["ts", "proba"])

    # Resample a 4h (minúscula para evitar FutureWarning)
    c = df["close"].resample("4h").last().dropna()
    if len(c) < 40:
        return pd.DataFrame(columns=["ts", "proba"])

    logc  = np.log(c)
    ret4h = logc.diff(1)
    ret1d = logc.diff(6)                   # ~1 día (6 velas de 4h)
    ema12 = c.ewm(span=12, min_periods=6).mean()
    ema36 = c.ewm(span=36, min_periods=12).mean()
    s12   = np.log(ema12).diff()
    s36   = np.log(ema36).diff()

    # Rango/ATR simple para normalizar retornos (máx-mín de 4h rolling)
    hi4h = df["close"].resample("4h").max()
    lo4h = df["close"].resample("4h").min()
    rng  = (hi4h - lo4h).rolling(12, min_periods=6).mean()  # ~2 días
    nret4h = ret4h / rng.reindex(c.index).replace(0, np.nan)

    # Score multi-componente con z-scores robustos
    sc = (
        1.3 * zscore(ret4h, 48).fillna(0) +
        1.0 * zscore(ret1d, 48).fillna(0) +
        2.2 * zscore(s12,   48).fillna(0) +
        1.4 * zscore(s36,   48).fillna(0) +
        0.8 * zscore(nret4h,48).fillna(0)
    )

    # Si la varianza sigue muy baja, usa mezcla con retorno acumulado 2d
    if np.nanstd(sc.values) < 1e-8:
        alt = ret4h.rolling(12, min_periods=6).sum().fillna(0)
        sc = sc.fillna(0) + 0.1 * alt

    # Percentil rank (0..1) + clip leve para evitar 0/1 exactos
    proba = pct_rank(sc).clip(0.02, 0.98)

    # Si aún quedan <10 valores distintos, usa directamente rank de ret4h
    if proba.nunique() < 10:
        proba = pct_rank(ret4h.fillna(0)).clip(0.02, 0.98)

    out = pd.DataFrame({
        "ts": proba.index.tz_localize(None),
        "proba": pd.to_numeric(proba.values, errors="coerce")
    }).dropna().reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ohlc_csv", default="data/ohlc/1m/BTC-USD.csv")
    ap.add_argument("--out_root", default="reports/windows_fixed")
    ap.add_argument("--asset", default="BTC-USD")
    ap.add_argument("--windows", nargs="+", default=["2023Q4","2024H1"])
    args = ap.parse_args()

    out_root = Path(args.out_root)
    for w in args.windows:
        start, end = WINDOWS[w]
        out = make_proba_ranked(Path(args.ohlc_csv), start, end)
        outp = out_root / w / f"{args.asset}.csv"
        outp.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(outp, index=False)
        if out.empty:
            print(f"[WARN] {w}: salida VACÍA (revisa rango OHLC).")
        else:
            q = out["proba"].quantile([0,.25,.5,.75,1]).round(4).to_dict()
            ge58 = int((out["proba"]>=0.58).sum())
            ge60 = int((out["proba"]>=0.60).sum())
            print(f"[OK] {w} -> {outp} rows={len(out)} q={q}  >=0.58:{ge58}  >=0.60:{ge60}")

if __name__ == "__main__":
    main()