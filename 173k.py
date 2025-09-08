# swing_4h_phase1.py
# Fase 1 – Swing Trading 4 h con ML (RandomForest) – “Diamante negro” ~235 k

import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Parámetros contextuales
FIXED = {
    "ema100_len":    100,  # EMA 4 h de largo plazo
    "adx_daily_len": 14    # ADX diario para contexto
}

# Parámetros de señal
SIGNAL_P = {
    "ema_fast": 12,
    "ema_slow": 26,
    "rsi_len":  14,
    "atr_len":  14,
    "adx_len":  14
}

FEATURES = ["ema_fast", "ema_slow", "rsi", "atr", "adx4h", "adx_daily"]

def download_data():
    fname = "btc_4h.csv"
    if os.path.exists(fname):
        df = pd.read_csv(fname, index_col=0, parse_dates=True)
        df.index = df.index.tz_localize(None)
    else:
        df = yf.download("BTC-USD", period="730d", interval="4h", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(fname)
    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = df.columns.str.lower()
    return df.dropna().sort_index()

def add_features(df):
    df = df.copy()
    df["ema_fast"] = ta.ema(df["close"], length=SIGNAL_P["ema_fast"])
    df["ema_slow"] = ta.ema(df["close"], length=SIGNAL_P["ema_slow"])
    df["rsi"]      = ta.rsi(df["close"], length=SIGNAL_P["rsi_len"])
    df["atr"]      = ta.atr(df["high"], df["low"], df["close"], length=SIGNAL_P["atr_len"])
    adx4h = ta.adx(df["high"], df["low"], df["close"], length=SIGNAL_P["adx_len"])
    col4h = next((c for c in adx4h.columns if "adx" in c.lower()), None)
    df["adx4h"]    = adx4h[col4h] if col4h else np.nan

    dfd = yf.download("BTC-USD", period="400d", interval="1d", progress=False)
    if isinstance(dfd.columns, pd.MultiIndex):
        dfd.columns = dfd.columns.get_level_values(0)
    dfd.columns = dfd.columns.str.lower()
    adx_d = ta.adx(dfd["high"], dfd["low"], dfd["close"], length=FIXED["adx_daily_len"])
    col_d = next((c for c in adx_d.columns if "adx" in c.lower()), None)
    dfd["adx_daily"] = adx_d[col_d] if col_d else np.nan
    df["adx_daily"]  = dfd["adx_daily"].reindex(df.index, method="ffill")

    return df.dropna()

def make_target(df):
    df = df.copy()
    df["target"] = np.where(
        df["close"].shift(-6) > df["close"] * 1.03, 1,
        np.where(df["close"].shift(-6) < df["close"] * 0.97, -1, 0)
    )
    return df.dropna()

def train_and_tune(X, y):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=42, class_weight="balanced"))
    ])
    param_grid = {
        "rf__n_estimators":      [50, 100],
        "rf__max_depth":         [5, 10],
        "rf__min_samples_split": [2, 5]
    }
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(pipe, param_grid, cv=tscv, scoring="accuracy", n_jobs=-1, verbose=1)
    grid.fit(X, y)
    print("Mejores parámetros RF:", grid.best_params_)
    return grid.best_estimator_

def backtest_threshold(df, model, thresholds=(0.55, 0.60, 0.65, 0.70)):
    df = df.copy()
    df["prob_up"] = model.predict_proba(df[FEATURES])[:, 1]
    results = []
    for thr in thresholds:
        equity, trades = 10000.0, []
        for _, r in df.iterrows():
            if r["prob_up"] > thr and r["close"] > r["ema_slow"]:
                pnl = (r["close"] - r["open"]) - r["close"] * 0.0002
                trades.append(pnl)
        arr = np.array(trades)
        results.append({
            "threshold":     thr,
            "trades":        len(arr),
            "win_rate":      len(arr[arr>0]) / len(arr) * 100 if len(arr) else 0,
            "net_profit":    arr.sum() if arr.size else 0,
            "profit_factor": arr[arr>0].sum() / abs(arr[arr<0].sum()) if arr[arr<0].size else np.nan
        })
    return pd.DataFrame(results).sort_values("net_profit", ascending=False)

def main():
    df0 = download_data()
    df1 = add_features(df0)
    df2 = make_target(df1)

    X, y = df2[FEATURES], df2["target"]
    model = train_and_tune(X, y)

    print("\n=== Threshold Tuning Swing 4H (RF) ===")
    df_tune = backtest_threshold(df2, model)
    print(df_tune.to_string(index=False))

if __name__ == "__main__":
    main()