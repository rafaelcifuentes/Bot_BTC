# swing_4h_phase1_long_short_tuning.py
# Versión con señales long y short para tuning del umbral (fine-grained)

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

# --- Parámetros fijos de contexto ---
FIXED = {
    "ema100_len":    100,
    "adx_daily_len": 14
}

# --- Parámetros para la señal (a afinar en Fase 1) ---
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
    adx4h_cols = [c for c in adx4h.columns if "adx" in c.lower()]
    df["adx4h"] = adx4h[adx4h_cols[0]] if adx4h_cols else np.nan

    dfd = yf.download("BTC-USD", period="400d", interval="1d", progress=False)
    if isinstance(dfd.columns, pd.MultiIndex):
        dfd.columns = dfd.columns.get_level_values(0)
    dfd.columns = dfd.columns.str.lower()
    adx_d = ta.adx(dfd["high"], dfd["low"], dfd["close"], length=FIXED["adx_daily_len"])
    adx_d_cols = [c for c in adx_d.columns if "adx" in c.lower()]
    df["adx_daily"] = adx_d[adx_d_cols[0]].reindex(df.index, method="ffill") if adx_d_cols else np.nan

    return df.dropna()

def make_target(df):
    df = df.copy()
    df["target"] = np.where(
        df["close"].shift(-6) > df["close"] * 1.03,  1,
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

def backtest_threshold(df, model, thresholds):
    X = df[FEATURES]
    proba = model.predict_proba(X)
    df2 = df.copy()
    # clase 1 → long; clase 0 → short
    df2["prob_up"]   = proba[:, 1]
    df2["prob_down"] = proba[:, 0]

    results = []
    for thr in thresholds:
        equity = 10000.0
        trades = []
        for _, r in df2.iterrows():
            # long
            if r["prob_up"]   > thr and r["close"] > r["ema_slow"]:
                pnl = (r["close"] - r["open"]) - r["close"] * 0.0002
                trades.append(pnl)
            # short
            elif r["prob_down"] > thr and r["close"] < r["ema_slow"]:
                pnl = (r["open"] - r["close"]) - r["close"] * 0.0002
                trades.append(pnl)
        arr = np.array(trades)
        net = arr.sum() if arr.size else 0.0
        pf  = arr[arr>0].sum() / abs(arr[arr<0].sum()) if arr.size and arr[arr<0].size else np.nan
        wr  = len(arr[arr>0]) / len(arr) * 100 if arr.size else 0.0
        results.append({
            "threshold":     thr,
            "trades":        len(arr),
            "win_rate":      wr,
            "net_profit":    net,
            "profit_factor": pf
        })
    return pd.DataFrame(results).sort_values("net_profit", ascending=False)

def main():
    df0 = download_data()
    df1 = add_features(df0)
    df2 = make_target(df1)

    # Entrena RF
    X, y = df2[FEATURES], df2["target"]
    model = train_and_tune(X, y)

    # Tuning del umbral (0.50 a 0.80 paso 0.01)
    thresholds = np.round(np.linspace(0.50, 0.80, 31), 3)
    df_tune = backtest_threshold(df2, model, thresholds)

    print("\n=== Threshold Tuning Swing 4H Long/Short (RF) ===")
    print(df_tune.to_string(index=False))

    best = df_tune.iloc[0]
    print(f"\n🖤 Diamante Negro @ thr={best.threshold:.3f}: "
          f"Net Profit=${best.net_profit:,.2f}, Win-rate={best.win_rate:.2f}%, "
          f"PF={best.profit_factor:.2f}, Trades={int(best.trades)}")

if __name__ == "__main__":
    main()