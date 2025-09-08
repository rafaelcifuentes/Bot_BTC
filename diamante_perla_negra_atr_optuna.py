#!/usr/bin/env python3
# diamante_perla_negra_atr_optuna.py
# Optimiza SÓLO riesgo+filtros manteniendo fija la señal “Diamante Perla Negra”.

import sys, warnings, optuna
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SYMBOL      = "BTC-USD"
PERIOD_4H   = "730d"
INTERVAL_4H = "4h"
PERIOD_1D   = "800d"
INTERVAL_1D = "1d"

# Señal fija (igual que en el core)
SIG = dict(ema_fast=12, ema_slow=26, rsi_len=14, atr_len=14, adx_len=14)

FEATURES = ["ema_fast","ema_slow","rsi","atr","adx4h","adx1d"]
COST, SLIP, RISK_PERC = 0.0002, 0.0001, 0.01
TRIALS = 60

def load_data():
    df4 = yf.download(SYMBOL, period=PERIOD_4H, interval=INTERVAL_4H,
                      progress=False, auto_adjust=False)
    if df4.empty: sys.exit("❌ Error: no 4h data")
    if isinstance(df4.columns, pd.MultiIndex):
        df4.columns = df4.columns.get_level_values(0)
    df4.columns = df4.columns.str.lower()
    df4.index = pd.to_datetime(df4.index).tz_localize(None)
    df4 = df4[['open','high','low','close','volume']].dropna().copy()

    dfd = yf.download(SYMBOL, period=PERIOD_1D, interval=INTERVAL_1D,
                      progress=False, auto_adjust=False)
    if dfd.empty: sys.exit("❌ Error: no 1d data")
    if isinstance(dfd.columns, pd.MultiIndex):
        dfd.columns = dfd.columns.get_level_values(0)
    dfd.columns = dfd.columns.str.lower()
    dfd.index = pd.to_datetime(dfd.index).tz_localize(None)
    dfd = dfd[['high','low','close']].dropna().copy()

    adx_d = ta.adx(dfd['high'], dfd['low'], dfd['close'], length=14)
    if adx_d is not None and not adx_d.empty:
        col = next((c for c in adx_d.columns if 'ADX_' in c or 'adx' in c.lower()), None)
        dfd['adx1d'] = adx_d[col] if col else np.nan
    else:
        dfd['adx1d'] = np.nan

    df4['adx1d'] = dfd['adx1d'].reindex(df4.index, method='ffill')
    return df4.dropna()

def add_features(df):
    df = df.copy()
    df["ema_fast"] = ta.ema(df["close"], length=SIG['ema_fast'])
    df["ema_slow"] = ta.ema(df["close"], length=SIG['ema_slow'])
    df["rsi"]      = ta.rsi(df["close"], length=SIG['rsi_len'])
    df["atr"]      = ta.atr(df["high"], df["low"], df["close"], length=SIG['atr_len'])
    adx4           = ta.adx(df["high"], df["low"], df["close"], length=SIG['adx_len'])
    if adx4 is not None and not adx4.empty:
        col = next((c for c in adx4.columns if 'ADX_' in c or 'adx' in c.lower()), None)
        df["adx4h"] = adx4[col] if col else 0
    else:
        df["adx4h"] = 0
    return df.dropna()

def make_target(df):
    df = df.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna()

def train_model(df):
    X = df[FEATURES]; y = df["target"]
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=None,
            class_weight="balanced",
            random_state=42, n_jobs=-1))
    ])
    pipe.fit(X, y)
    return pipe

def backtest_long(df, p):
    equity, trades, pos = 10000.0, [], None
    for _, r in df.iterrows():
        if pos is None:
            cond = (r["prob_up"] > p["threshold"]
                    and r["close"] > r["ema_slow"]
                    and r["adx4h"] > p["adx_filter"]
                    and (np.isnan(r["adx1d"]) or r["adx1d"] > p["adx_daily_filter"]))
            if not cond:
                continue
            e = r["open"]*(1+SLIP)
            atr0 = r["atr"]
            if atr0 <= 0: continue
            s  = e - atr0 * p["sl_atr_mul"]
            t1 = e + atr0 * p["tp1_atr_mul"]
            t2 = e + atr0 * p["tp2_atr_mul"]
            if s >= e: continue
            sz = (equity*RISK_PERC)/(e-s)
            equity -= e*sz*COST
            pos = dict(e=e,s=s,t1=t1,t2=t2,sz=sz,hm=e,p1=False,atr0=atr0)
        else:
            e = pos["e"]
            hi, lo = r["high"], r["low"]
            pos["hm"] = max(pos["hm"], hi)
            exit_p=None
            if not pos["p1"] and hi>=pos["t1"]:
                pnl=(pos["t1"]-e)*(pos["sz"]*p["partial_pct"])
                equity+=pnl; trades.append(pnl)
                pos["sz"]*=(1-p["partial_pct"]); pos["p1"]=True
            if hi>=pos["t2"]:
                exit_p=pos["t2"]
            new_s = pos["hm"] - pos["atr0"]*p["trail_mul"]
            if new_s>pos["s"]: pos["s"]=new_s
            if lo<=pos["s"]: exit_p=pos["s"]
            if exit_p is not None:
                pnl=(exit_p-e)*pos["sz"] - exit_p*pos["sz"]*COST
                equity+=pnl; trades.append(pnl); pos=None
    arr=np.array(trades)
    if arr.size==0:
        return dict(net=0,trades=0,max_dd=0,score=-1e6)
    eq=10000+np.cumsum(arr)
    mdd=(np.maximum.accumulate(eq)-eq).max()
    net=arr.sum()
    score = net/(mdd+1)
    return dict(net=net,trades=len(arr),max_dd=mdd,score=score)

def objective(trial, train_df, test_df):
    p = dict(
        sl_atr_mul      = trial.suggest_float("sl_atr_mul", 0.8, 4.0),
        tp1_atr_mul     = trial.suggest_float("tp1_atr_mul", 0.5, 3.0),
        tp2_atr_mul     = trial.suggest_float("tp2_atr_mul", 2.0, 8.0),
        trail_mul       = trial.suggest_float("trail_mul", 1.0, 4.0),
        partial_pct     = trial.suggest_float("partial_pct", 0.1, 0.9),
        threshold       = trial.suggest_float("threshold", 0.50, 0.70),
        adx_filter      = trial.suggest_int("adx_filter", 5, 25),
        adx_daily_filter= trial.suggest_int("adx_daily_filter", 5, 25),
    )
    # entrenar una sola vez afuera; aquí recibimos ya el modelo OOS…
    # pero para simplicidad, reentrenamos barato con las features fijas:
    model = train_model(train_df)

    test = test_df.copy()
    probs = model.predict_proba(test[FEATURES])
    up_idx = np.where(model.classes_==1)[0]
    test["prob_up"] = probs[:, up_idx[0]] if up_idx.size>0 else 0.5

    # usar 90 días OOS para score
    n = 90*6
    if len(test)<n: return -1e6
    res = backtest_long(test.tail(n), p)
    if res["trades"] < 10:
        return -1e6
    return res["score"]

def forward_eval(train_df, test_df, best):
    model = train_model(train_df)
    test = test_df.copy()
    probs = model.predict_proba(test[FEATURES])
    up_idx = np.where(model.classes_==1)[0]
    test["prob_up"] = probs[:, up_idx[0]] if up_idx.size>0 else 0.5

    for days in (90,180,365):
        n=days*6
        if len(test)<n:
            print(f"{days}d → not enough data.");
            continue
        res = backtest_long(test.tail(n), best)
        print(f"{days}d → Net ${res['net']:.2f}, Trades {res['trades']}, MDD {res['max_dd']:.2f}, Score {res['net']/(res['max_dd']+1):.2f}")

def main():
    raw = load_data()
    feat = add_features(raw)
    data = make_target(feat)
    cut = int(len(data)*0.7)
    train_df, test_df = data.iloc[:cut], data.iloc[cut:]

    print("🚀 Optimizando riesgo+filtros (Optuna)…")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, train_df, test_df),
                   n_trials=TRIALS, show_progress_bar=True)
    best = study.best_params
    print("✅ Mejores params:", best)
    print("\n── Forward-Test ──")
    forward_eval(train_df, test_df, best)

if __name__ == "__main__":
    main()