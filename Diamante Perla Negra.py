# swing_4h_atr_optuna_with_shorts_fixed.py
import sys
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# — Parámetros definitivos de riesgo y señal —
BEST_P = {
    "sl_atr_mul":  1.0002671010070825,
    "tp1_atr_mul": 1.182767437207439,
    "tp2_atr_mul": 4.001379532965078,
    "threshold":   0.6034552744914382,
    "partial_pct": 0.5,     # Ajusta si quieres salida parcial
}
SYMBOL    = "BTC-USD"
PERIOD    = "730d"
INTERVAL  = "4h"
FEATURES  = ["ema_fast","ema_slow","rsi","atr","adx4h"]
COST      = 0.0002
SLIP      = 0.0001
RISK_PERC = 0.01

def load_and_feat():
    df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL,
                     progress=False, auto_adjust=False)
    if df.empty:
        sys.exit("Error descargando datos.")
    # Columnas en minúscula
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    # Indicadores
    df["ema_fast"] = ta.ema(df["close"], length=12)
    df["ema_slow"] = ta.ema(df["close"], length=26)
    df["rsi"]      = ta.rsi(df["close"], length=14)
    df["atr"]      = ta.atr(df["high"], df["low"], df["close"], length=14)
    adx4 = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx4 is not None and not adx4.empty:
        col = next((c for c in adx4.columns if "adx" in c.lower()), None)
        df["adx4h"] = adx4[col] if col else 0
    else:
        df["adx4h"] = 0
    return df.dropna()

def train_model(df):
    df_ = df.copy()
    df_["target"] = np.sign(df_["close"].shift(-1) - df_["close"]).fillna(0).astype(int)
    df_.dropna(inplace=True)
    X = df_[FEATURES]
    y = df_["target"]
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            class_weight="balanced", random_state=42, n_estimators=100))
    ])
    pipe.fit(X, y)
    return pipe

def backtest_period(df, days, model, p):
    cut = int(len(df) * 0.7)
    test = df.iloc[cut:].copy()
    # probabilidades
    probs = model.predict_proba(test[FEATURES])
    up_idx = np.where(model.classes_ == 1)[0][0]
    dn_idx = np.where(model.classes_ == -1)[0][0]
    test["p_up"] = probs[:, up_idx]
    test["p_dn"] = probs[:, dn_idx]

    n      = days * 6  # velas de 4h
    period = test.tail(n)
    equity, trades, pos = 10000.0, [], None

    for _, r in period.iterrows():
        if pos is None:
            long_sig  = (r["p_up"] > p["threshold"]) and (r["close"] > r["ema_slow"])
            short_sig = (r["p_dn"] > p["threshold"]) and (r["close"] < r["ema_slow"])
            if not (long_sig or short_sig):
                continue
            d     = 1 if long_sig else -1
            entry = r["open"] * (1 + SLIP * d)
            atr0  = r["atr"]
            stop  = entry - d * p["sl_atr_mul"] * atr0
            tp1   = entry + d * p["tp1_atr_mul"] * atr0
            tp2   = entry + d * p["tp2_atr_mul"] * atr0
            if d * (stop - entry) >= 0:
                continue
            size = (equity * RISK_PERC) / abs(entry - stop)
            pos = {"d":d,"e":entry,"s":stop,"t1":tp1,"t2":tp2,
                   "sz":size,"hm":entry,"p1":False,"atr0":atr0}
        else:
            d,e = pos["d"], pos["e"]
            pos["hm"] = max(pos["hm"], r["high"]) if d==1 else min(pos["hm"], r["low"])
            exit_p = None

            # Salida parcial TP1
            if not pos["p1"]:
                if (d==1 and r["high"]>=pos["t1"]) or (d==-1 and r["low"]<=pos["t1"]):
                    pnl = (pos["t1"]-e)*d*(pos["sz"]*p.get("partial_pct",0.5))
                    equity += pnl
                    trades.append(pnl)
                    pos["sz"] *= (1-p.get("partial_pct",0.5))
                    pos["p1"] = True

            # TP2
            if (d==1 and r["high"]>=pos["t2"]) or (d==-1 and r["low"]<=pos["t2"]):
                exit_p = pos["t2"]

            # Trailing stop
            new_stop = pos["hm"] - d * p["sl_atr_mul"] * pos["atr0"]
            if (d==1 and new_stop > pos["s"]) or (d==-1 and new_stop < pos["s"]):
                pos["s"] = new_stop
            if (d==1 and r["low"]<=pos["s"]) or (d==-1 and r["high"]>=pos["s"]):
                exit_p = pos["s"]

            if exit_p is not None:
                pnl = (exit_p-e)*d*pos["sz"] - exit_p*pos["sz"]*COST
                equity += pnl
                trades.append(pnl)
                pos = None

    arr = np.array(trades)
    return {
        "days": days,
        "net": arr.sum() if arr.size else 0,
        "pf":  (arr[arr>0].sum() / abs(arr[arr<0].sum())) if arr.size and arr[arr<0].size else np.nan,
        "win_rate": len(arr[arr>0]) / len(arr) * 100 if arr.size else 0,
        "trades": len(arr)
    }

def main():
    df = load_and_feat()
    model = train_model(df)
    results = [ backtest_period(df, d, model, BEST_P) for d in (30,60,90) ]
    print(pd.DataFrame(results).set_index("days"))

if __name__=="__main__":
    main()