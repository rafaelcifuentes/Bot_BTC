# btc_test.py (actualizado con curvas de equity y drawdown)

import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import optuna
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

DATA_FILE = "btc_4h.csv"
SYMBOL = "BTC-USD"
PERIOD = "730d"
INTERVAL = "4h"
FIXED = {"cost": 0.0002, "slip": 0.0001}

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, header=None)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'nan']
        df.drop(columns=["nan"], inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
    else:
        df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL)
        df.to_csv(DATA_FILE)
    return df[['open', 'high', 'low', 'close', 'volume']]

def add_features(df):
    df = df.copy()
    df["ema_fast"] = ta.ema(df["close"], length=12)
    df["ema_slow"] = ta.ema(df["close"], length=26)
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx4h"] = adx["ADX_14"]
    return df.dropna()

def make_target(df):
    df = df.copy()
    shift = 6
    up = df["close"].shift(-shift) > df["close"] * 1.03
    down = df["close"].shift(-shift) < df["close"] * 0.97
    df["target"] = np.where(up, 1, np.where(down, -1, 0))
    return df.dropna()

def train_rf(df, threshold_rf):
    X = df[["ema_fast", "ema_slow", "rsi", "atr", "adx4h"]]
    y = df["target"].map(lambda v: 1 if v == 1 else 0)
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    pipe.fit(X, y)
    df["prob_up"] = pipe.predict_proba(X)[:, 1]
    df["signal"] = df["prob_up"] > threshold_rf
    return df

def backtest(df, sl_atr_mul, tp1_atr_mul, tp2_atr_mul, partial_pct, threshold, risk_perc):
    equity = 10000.0
    trades = []
    eq_curve = []
    op = None
    for _, r in df.iterrows():
        sig_long = r["prob_up"] > threshold and r["close"] > r["ema_slow"]
        sig_short = r["prob_up"] < (1 - threshold) and r["close"] < r["ema_slow"]

        if op is None and (sig_long or sig_short):
            d = 1 if sig_long else -1
            e = r["open"] * (1 + FIXED["slip"] * d)
            atr0 = r["atr"]
            stop = e - d * atr0 * sl_atr_mul
            t1 = e + d * atr0 * tp1_atr_mul
            t2 = e + d * atr0 * tp2_atr_mul
            sz = (equity * risk_perc) / abs(e - stop)
            equity -= e * sz * FIXED["cost"]
            op = {"dir": d, "e": e, "s": stop, "t1": t1, "t2": t2, "sz": sz, "hm": e, "p1": False, "atr0": atr0}
        elif op:
            d, e = op["dir"], op["e"]
            hm = max(op["hm"], r["high"]) if d == 1 else min(op["hm"], r["low"])
            op["hm"] = hm
            exit_p = None
            if not op["p1"] and ((d == 1 and r["high"] >= op["t1"]) or (d == -1 and r["low"] <= op["t1"])):
                pnl = (op["t1"] - e) * d * (op["sz"] * partial_pct)
                equity += pnl
                trades.append(pnl)
                op["sz"] *= (1 - partial_pct)
                op["p1"] = True
            if (d == 1 and r["high"] >= op["t2"]) or (d == -1 and r["low"] <= op["t2"]):
                exit_p = op["t2"]
            new_s = hm - d * op["atr0"] * sl_atr_mul
            if (d == 1 and r["low"] <= new_s) or (d == -1 and r["high"] >= new_s):
                exit_p = new_s
            if exit_p:
                pnl = (exit_p - e) * d * op["sz"] - exit_p * op["sz"] * FIXED["cost"]
                equity += pnl
                trades.append(pnl)
                op = None
        eq_curve.append(equity)
    return np.array(trades), np.array(eq_curve)

def plot_equity_drawdown(eq_curve, title):
    eq = eq_curve
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(eq, label='Equity Curve')
    plt.title(f'Equity - {title}')
    plt.subplot(1, 2, 2)
    plt.plot(dd, label='Drawdown', color='red')
    plt.title(f'Drawdown - {title}')
    plt.tight_layout()
    plt.show()

# MAIN
if __name__ == "__main__":
    df0 = load_data()
    df1 = add_features(df0)
    df2 = make_target(df1)
    DATA = df2.copy()

    # Mejores parámetros obtenidos tras la optimización (puedes reemplazarlos si optimizas de nuevo)
    best_params = {
        'sl_atr_mul': 4.232104043224681,
        'tp1_atr_mul': 2.9783740545046458,
        'tp2_atr_mul': 5.250667719151054,
        'partial_pct': 0.5520943737712057,
        'threshold': 0.5744797590768347,
        'risk_perc': 0.04928122154755196
    }

    df_signal = train_rf(DATA, threshold_rf=0.55)

    for days in [30, 60, 90]:
        cutoff = df_signal.index[-1] - pd.Timedelta(days=days)
        df_oos = df_signal[df_signal.index >= cutoff]
        pnl, eq_curve = backtest(
            df_oos,
            best_params['sl_atr_mul'],
            best_params['tp1_atr_mul'],
            best_params['tp2_atr_mul'],
            best_params['partial_pct'],
            best_params['threshold'],
            best_params['risk_perc']
        )
        net = pnl.sum()
        pf = pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum()) if pnl[pnl < 0].size else np.nan
        win_rate = (pnl > 0).sum() / len(pnl) * 100 if len(pnl) else 0

        print(f"\n📈 OOS {days}d → Net Profit: {net:.2f}, PF: {pf:.2f}, Trades: {len(pnl)}, Win Rate: {win_rate:.2f}%")
        plot_equity_drawdown(eq_curve, f"{days}d OOS")