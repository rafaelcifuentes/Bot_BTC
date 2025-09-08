# swing_4h_ml_rf.py
# Etapa 3 – Swing Trading 4h con ML (RandomForest) y gestión de riesgos

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# 1) CARGAR DATOS (4h) O DESCARGAR SI NO EXISTE
def load_data():
    DATA_FILE = "btc_4h.csv"
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE, index_col=0, header=[0])
            df.index = pd.to_datetime(df.index).tz_localize(None)
            for c in ["Open","High","Low","Close","Volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.dropna().sort_index()
        except Exception:
            os.remove(DATA_FILE)

    df = yf.download("BTC-USD", interval="4h", period="729d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv(DATA_FILE)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().sort_index()


# 2) AÑADIR FEATURES
def add_features(df):
    df["EMA12"] = ta.ema(df["Close"], length=12)
    df["EMA26"] = ta.ema(df["Close"], length=26)

    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["MACD"]      = macd["MACD_12_26_9"]
    df["MACD_hist"] = macd["MACDh_12_26_9"]

    bb = ta.bbands(df["Close"], length=20, std=2)
    df["BB_mid"]   = bb["BBM_20_2.0"]
    df["BB_upper"] = bb["BBU_20_2.0"]
    df["BB_lower"] = bb["BBL_20_2.0"]

    df["RSI14"]    = ta.rsi(df["Close"], length=14)
    df["ATR14"]    = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["Vol_SMA10"]= df["Volume"].rolling(10).mean()

    # On-chain flows (opcional)
    onchain = "onchain_flows_4h.csv"
    if os.path.exists(onchain):
        flows = pd.read_csv(onchain, index_col=0, parse_dates=True)
        flows.index = flows.index.tz_localize(None)
        df["net_flow"] = flows["net_flow"].resample("4H").ffill().reindex(df.index, method="ffill")
    else:
        df["net_flow"] = 0.0

    # Fear & Greed Index
    try:
        import requests
        resp = requests.get("https://api.alternative.me/fng/?limit=1500")
        ds = pd.DataFrame(resp.json()["data"])
        ds["timestamp"] = pd.to_datetime(ds["timestamp"].astype(int), unit="s")
        ds.set_index("timestamp", inplace=True)
        ds["value"] = pd.to_numeric(ds["value"], errors="coerce")
        ds.index = ds.index.tz_localize(None)
        df["fng"] = ds["value"].resample("4H").ffill().reindex(df.index, method="ffill")
    except Exception:
        df["fng"] = 50.0

    # Días desde último halving
    halvings = [pd.Timestamp("2012-11-28"), pd.Timestamp("2016-07-09"),
                pd.Timestamp("2020-05-11"), pd.Timestamp("2024-04-19")]
    df["days_since_halving"] = df.index.map(
        lambda ts: (ts - max([h for h in halvings if h<=ts])).days if any(h<=ts for h in halvings) else 0
    )

    return df.dropna()


# 3) DEFINIR TARGET (binario: 1 sólo alzas)
def make_target(df):
    df["Target"] = np.where(
        df["Close"].shift(-6) > df["Close"] * 1.03, 1,
        np.where(df["Close"].shift(-6) < df["Close"] * 0.97, -1, 0)
    )
    return df.dropna()


# 4) ENTRENAR Y TUNEAR MODELO
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
    grid = GridSearchCV(pipe, param_grid, cv=tscv, scoring="accuracy", n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_


# 5) BACKTEST long-only con SL/TP porcentual y trailing
def backtest(df, threshold,
             sl_pct=0.025, tp_pct=0.05, trail_pct=0.01,
             cost_perc=0.0002, slip_perc=0.0001):
    equity, trades, eq_curve, op = 10000.0, [], [], None
    for _, r in df.iterrows():
        p_up = r["Prob_up"]
        o, h, l = r["Open"], r["High"], r["Low"]

        # abrir long
        if op is None and p_up > threshold:
            entry = o * (1 + slip_perc)
            stop  = entry * (1 - sl_pct)
            take  = entry * (1 + tp_pct)
            size  = (equity * 0.01) / (entry - stop)
            equity -= entry * size * cost_perc
            op = {"entry":entry, "stop":stop, "tp":take,
                  "size":size, "high_max":entry, "partial":False}

        # gestionar posición abierta
        elif op:
            op["high_max"] = max(op["high_max"], h)
            # salida parcial
            if not op["partial"] and h >= op["tp"]:
                pnl = (op["tp"] - op["entry"]) * (op["size"] * 0.5)
                equity += pnl; trades.append(pnl)
                op["size"] *= 0.5; op["partial"] = True
            # trailing stop
            op["stop"] = max(op["stop"], op["high_max"] * (1 - trail_pct))
            exit_price = None
            if l <= op["stop"]:
                exit_price = op["stop"]
            elif op["partial"] and h >= op["tp"]:
                exit_price = op["tp"]
            if exit_price is not None:
                fee = exit_price * op["size"] * cost_perc
                pnl = (exit_price - op["entry"]) * op["size"] - fee
                equity += pnl; trades.append(pnl); op = None

        eq_curve.append(equity)

    wins   = np.array([t for t in trades if t>0])
    losses = np.array([t for t in trades if t<0])
    pf     = wins.sum() / abs(losses.sum()) if losses.size else np.inf
    max_dd = (np.maximum.accumulate(eq_curve) - eq_curve).max() if eq_curve else 0

    return {"trades":len(trades),"win_rate":len(wins)/len(trades)*100 if trades else 0,
            "net_profit":equity-10000,"profit_factor":pf,"max_drawdown":max_dd,
            "equity_curve":eq_curve}


if __name__=="__main__":
    df = load_data()
    df = add_features(df)
    df = make_target(df)

    # Entrenamiento y tuning
    features = ['EMA12','EMA26','MACD','MACD_hist','BB_mid','BB_upper','BB_lower',
                'RSI14','ATR14','Vol_SMA10','net_flow','fng','days_since_halving']
    X, y = df[features], df["Target"]
    model = train_and_tune(X, y)

    # Predicciones
    pos_idx = np.where(model.classes_==1)[0]
    df['Prob_up'] = model.predict_proba(X)[:, pos_idx[0]] if pos_idx.size else 0.0

    # Threshold Tuning
    print("\n=== Threshold Tuning Swing 4H (RF) ===")
    results = []
    for thr in [0.55, 0.60, 0.65, 0.70]:
        r = backtest(df, threshold=thr)
        r['threshold'] = thr
        results.append(r)
    df_tune = pd.DataFrame(results).sort_values('net_profit', ascending=False)
    print(df_tune[['threshold','trades','win_rate','net_profit','profit_factor','max_drawdown']].to_string(index=False))

    # Equity curve del mejor
    best = df_tune.iloc[0]
    eq = np.array(best['equity_curve'])
    dates = df.index[:len(eq)]
    plt.figure(figsize=(12,5))
    plt.plot(dates, eq, label=f"Equity @thr={best['threshold']:.2f}")
    plt.title("Equity Curve – Swing 4H RF Tuned")
    plt.xlabel("Date"); plt.ylabel("Equity ($)")
    plt.legend(); plt.tight_layout(); plt.show()