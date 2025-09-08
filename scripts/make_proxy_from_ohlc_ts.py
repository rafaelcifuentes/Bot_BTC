# scripts/make_proxy_from_ohlc_ts.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

WINDOWS = {
    "2022H2": ("2022-06-01","2023-01-31"),
    "2023Q4": ("2023-10-01","2023-12-31"),
    "2024H1": ("2024-01-01","2024-06-30"),
}

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def zscore(s, win):
    m = s.rolling(win, min_periods=max(5, win//5)).mean()
    v = s.rolling(win, min_periods=max(5, win//5)).std()
    return (s - m) / v.replace(0, np.nan)

def make_proba_4h(ohlc_csv: Path, start: str, end: str):
    df = pd.read_csv(ohlc_csv)
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("timestamp", df.columns[0])
    pcol = cols.get("close") or cols.get("price") or [c for c in df.columns if c != tcol][0]

    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    px = pd.to_numeric(df[pcol], errors="coerce")
    df = pd.DataFrame({"timestamp": ts, "close": px}).dropna().set_index("timestamp").sort_index()

    # recorta ventana y resamplea a 4H
    df = df.loc[start:end]
    if df.empty:
        return pd.DataFrame(columns=["ts","proba"])

    c4h = df["close"].resample("4H").last().dropna()
    if len(c4h) < 40:
        return pd.DataFrame(columns=["ts","proba"])

    # features simples y estables
    logc = np.log(c4h)
    ret  = logc.diff()
    zret = zscore(ret, 48)                 # ~8 días de 4H
    ema  = c4h.ewm(span=24, min_periods=10).mean()
    slope= np.log(ema).diff()

    # momentum de 2 días (12 velas 4H) normalizado
    mom  = (c4h / c4h.rolling(12, min_periods=6).mean() - 1.0)

    # score y proba
    score = (1.4 * zret.fillna(0)) + (4.0 * slope.fillna(0)) + (0.8 * mom.fillna(0))
    proba = pd.Series(sigmoid(score), index=c4h.index)

    # fallback anti-plano: si var(proba) es muy baja, usa rank de returns
    if np.nanstd(proba.values) < 1e-4:
        rk = ret.rank(pct=True).fillna(0.5)
        proba = 0.25 + 0.5 * rk  # en [0.25,0.75]

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
        out = make_proba_4h(Path(args.ohlc_csv), start, end)
        outp = out_root / w / f"{args.asset}.csv"
        outp.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(outp, index=False)
        if out.empty:
            print(f"[WARN] {w}: salida vacía (revisa rango OHLC).")
        else:
            q = out["proba"].quantile([0,.25,.5,.75,1]).round(4).to_dict()
            g58 = int((out["proba"]>=0.58).sum())
            print(f"[OK] {w} -> {outp} rows={len(out)} quantiles={q}  >=0.58:{g58}")

if __name__ == "__main__":
    main()