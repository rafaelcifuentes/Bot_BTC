# forward_test_apply_best_by_horizon.py
# Trains on "all before horizon", tests on the last {90,180,365} days.
# Preserves: ATR SL/TP+trail, partial TP, ADX 4h + ADX 1D filters, costs/slippage, score.

import json, sys
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ---------------- Config ----------------
SYMBOL       = "BTC-USD"
PERIOD_4H    = "730d"    # Yahoo limit for 4h data
INTERVAL_4H  = "4h"
PERIOD_1D    = "1900d"
INTERVAL_1D  = "1d"

COST         = 0.0002
SLIP         = 0.0001
FEATURES     = ["ema_fast","ema_slow","rsi","atr","adx4h","adx1d"]
HORIZONS     = [90, 180, 365]
MIN_TRAIN_DAYS = 90       # require at least this many days for training

PICK_PATH    = "minigrid_fast_pick.json"  # from your mini-grid sweep

# -------------- Data & Features --------------
def download_data():
    print("🔄 Downloading 4h + 1D data…")
    df4 = yf.download(SYMBOL, period=PERIOD_4H, interval=INTERVAL_4H,
                      progress=False, auto_adjust=False)
    if df4.empty:
        sys.exit("Failed to download 4h data (Yahoo 4h limit is ~730d).")
    if isinstance(df4.columns, pd.MultiIndex):
        df4.columns = df4.columns.get_level_values(0)
    df4.columns = df4.columns.str.lower()
    df4.index = pd.to_datetime(df4.index).tz_localize(None)

    dfd = yf.download(SYMBOL, period=PERIOD_1D, interval=INTERVAL_1D,
                      progress=False, auto_adjust=False)
    if dfd.empty:
        sys.exit("Failed to download 1D data.")
    if isinstance(dfd.columns, pd.MultiIndex):
        dfd.columns = dfd.columns.get_level_values(0)
    dfd.columns = dfd.columns.str.lower()
    dfd.index = pd.to_datetime(dfd.index).tz_localize(None)

    # Daily ADX
    adx1 = ta.adx(dfd["high"], dfd["low"], dfd["close"], length=14)
    if adx1 is not None and not adx1.empty:
        col = next((c for c in adx1.columns if "adx" in c.lower()), None)
        dfd["adx1d"] = adx1[col] if col else np.nan
    else:
        dfd["adx1d"] = np.nan

    # Align daily ADX to 4h index
    df4["adx1d"] = dfd["adx1d"].reindex(df4.index, method="ffill")
    return df4.dropna()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ta.ema(df["close"], length=12)
    df["ema_slow"] = ta.ema(df["close"], length=26)
    df["rsi"]      = ta.rsi(df["close"], length=14)
    df["atr"]      = ta.atr(df["high"], df["low"], df["close"], length=14)
    adx4 = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx4 is not None and not adx4.empty:
        col4 = next((c for c in adx4.columns if "adx" in c.lower()), None)
        df["adx4h"] = adx4[col4] if col4 else 0.0
    else:
        df["adx4h"] = 0.0
    return df.dropna()

def make_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    nxt = df["close"].shift(-1)
    diff = nxt - df["close"]
    df["target"] = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    return df.dropna()

# -------------- Model & Probabilities --------------
def train_model(df: pd.DataFrame):
    X, y = df[FEATURES], df["target"]
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))
    ])
    pipe.fit(X, y)
    return pipe

def add_prob_up(model, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    probs = model.predict_proba(df[FEATURES])
    classes = list(model.classes_)
    df["prob_up"] = probs[:, classes.index(1)] if 1 in classes else 0.5
    return df

# -------------- Backtest (long-only) --------------
def backtest_long(df: pd.DataFrame, p: dict, days: int):
    slice_ = df.tail(days * 6).copy()  # 6 bars/day at 4h
    equity_closed = 10000.0
    trades, pos, eq_curve = [], None, []

    for _, r in slice_.iterrows():
        if pos is None:
            if (r["prob_up"] > p["threshold"]
                and r["close"] > r["ema_slow"]
                and r["adx4h"] >= p["adx4_min"]
                and (pd.notna(r["adx1d"]) and r["adx1d"] >= p["adx1d_min"])):
                atr0 = r["atr"]
                if atr0 <= 0:
                    eq_curve.append(equity_closed); continue
                entry = r["open"] * (1 + SLIP)
                stop  = entry - p["sl_atr_mul"] * atr0
                tp1   = entry + p["tp1_atr_mul"] * atr0
                tp2   = entry + p["tp2_atr_mul"] * atr0
                if stop >= entry:
                    eq_curve.append(equity_closed); continue
                sz = (equity_closed * p["risk_perc"]) / (entry - stop)
                # entry cost
                cost = -entry * sz * COST
                trades.append(cost); equity_closed += cost
                pos = {"e": entry, "s": stop, "t1": tp1, "t2": tp2,
                       "sz": sz, "hm": entry, "p1": False, "atr0": atr0}
        else:
            pos["hm"] = max(pos["hm"], r["high"])
            exit_price = None

            # Partial take-profit
            if (not pos["p1"]) and (r["high"] >= pos["t1"]):
                part_sz = pos["sz"] * p["partial_pct"]
                pnl  = (pos["t1"] - pos["e"]) * part_sz
                cost = pos["t1"] * part_sz * COST
                trades.append(pnl - cost); equity_closed += (pnl - cost)
                pos["sz"] *= (1 - p["partial_pct"]); pos["p1"] = True

            # Full TP2
            if r["high"] >= pos["t2"]:
                exit_price = pos["t2"]

            # Trailing stop
            new_stop = pos["hm"] - p["trail_mul"] * pos["atr0"]
            if new_stop > pos["s"]:
                pos["s"] = new_stop
            if r["low"] <= pos["s"]:
                exit_price = pos["s"]

            # Exit
            if exit_price is not None:
                pnl  = (exit_price - pos["e"]) * pos["sz"]
                cost = exit_price * pos["sz"] * COST
                trades.append(pnl - cost); equity_closed += (pnl - cost)
                pos = None

        # Mark-to-market
        unreal = 0.0 if pos is None else (r["close"] - pos["e"]) * pos["sz"]
        eq_curve.append(equity_closed + unreal)

    if not trades:
        return {"net":0.0,"pf":0.0,"win_rate":0.0,"trades":0,"mdd":0.0,"score":0.0}
    arr = np.array(trades, float)
    net = float(arr.sum())
    eq  = np.array(eq_curve, float)
    mdd = float((np.maximum.accumulate(eq) - eq).max()) if eq.size else 0.0
    gains, losses = arr[arr>0], arr[arr<0]
    pf = float(gains.sum()/abs(losses.sum())) if losses.size else np.inf
    wr = float(len(gains)/len(arr)*100)
    score = float(net / (mdd + 1))
    return {"net":net,"pf":pf,"win_rate":wr,"trades":int(len(arr)),"mdd":mdd,"score":score}

# -------------- Horizon split (no leakage) --------------
def eval_horizon(df2: pd.DataFrame, p_best: dict, days: int):
    bars_needed = days * 6
    min_train_bars = MIN_TRAIN_DAYS * 6
    if len(df2) < bars_needed + min_train_bars:
        return f"{days}d → not enough bars. Need ≥ {bars_needed + min_train_bars}, have {len(df2)}"

    cut = len(df2) - bars_needed
    train, test = df2.iloc[:cut].copy(), df2.iloc[cut:].copy()

    model = train_model(train)
    oos   = add_prob_up(model, test)

    m = backtest_long(oos, p_best, days)
    return (f"{days}d → Net: ${m['net']:.2f}, PF: {m['pf']:.2f}, "
            f"Win%: {m['win_rate']:.2f}, Trades: {m['trades']}, "
            f"MDD: {m['mdd']:.2f}, Score: {m['score']:.2f}")

# -------------- Main --------------
def main():
    # Load picked set
    try:
        with open(PICK_PATH, "r") as f:
            p_best = json.load(f)
    except FileNotFoundError:
        sys.exit("minigrid_fast_pick.json not found. Run your mini-grid first.")

    df0 = download_data()
    df1 = add_features(df0)
    df2 = make_target(df1)

    for d in HORIZONS:
        print(eval_horizon(df2, p_best, d))

if __name__ == "__main__":
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    main()