# diamante_profileA_live.py
# Long-only 4h "Diamante Perla Negra" with Profile A (locked), ATR risk, partial TP, trailing,
# 4h+1D ADX filters, ML threshold ~0.50, cost/slippage, proper split, simple walk-forward.

import sys, json, os, math, time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)

# -----------------------------
# Locked Profile A (best risk-adjusted 180d score)
# -----------------------------
PROFILE_A = {
    "threshold": 0.50,
    "adx4_min": 6,
    "adx1d_min": 0,         # effectively off unless >0
    "sl_atr_mul": 1.4,
    "tp1_atr_mul": 1.6,
    "tp2_atr_mul": 6.0,
    "trail_mul": 1.0,
    "partial_pct": 0.50,
    "risk_perc": 0.005,
}

# Market/engine settings
SYMBOL       = "BTC-USD"
PERIOD_4H    = "730d"   # Yahoo 4h limit ~ last 730 days
INTERVAL_4H  = "4h"
PERIOD_1D    = "800d"
INTERVAL_1D  = "1d"

COST         = 0.0002
SLIP         = 0.0001

# Feature set for ML
FEATURES = ["ema_fast", "ema_slow", "rsi", "atr", "adx4h"]

# -----------------------------
# Data & features
# -----------------------------
def _first_adx_col(df_adx: pd.DataFrame) -> str | None:
    if df_adx is None or df_adx.empty:
        return None
    # Try typical pandas_ta columns: ADX_14 (or similar)
    for c in df_adx.columns:
        lc = str(c).lower()
        if "adx" in lc and "di" not in lc:
            return c
    return None

def download_data() -> pd.DataFrame:
    print("🔄 Downloading 4h + 1D data…")
    df4 = yf.download(SYMBOL, period=PERIOD_4H, interval=INTERVAL_4H, progress=False, auto_adjust=False)
    if df4 is None or df4.empty:
        sys.exit("Failed to download 4h data.")
    if isinstance(df4.columns, pd.MultiIndex):
        df4.columns = df4.columns.get_level_values(0)
    df4.columns = df4.columns.str.lower()
    df4.index = pd.to_datetime(df4.index).tz_localize(None)

    dfd = yf.download(SYMBOL, period=PERIOD_1D, interval=INTERVAL_1D, progress=False, auto_adjust=False)
    if dfd is None or dfd.empty:
        sys.exit("Failed to download 1D data.")
    if isinstance(dfd.columns, pd.MultiIndex):
        dfd.columns = dfd.columns.get_level_values(0)
    dfd.columns = dfd.columns.str.lower()
    dfd.index = pd.to_datetime(dfd.index).tz_localize(None)

    # Daily ADX
    adx1 = ta.adx(dfd["high"], dfd["low"], dfd["close"], length=14)
    if adx1 is not None and not adx1.empty:
        col = _first_adx_col(adx1)
        dfd["adx1d"] = adx1[col] if col else np.nan
    else:
        dfd["adx1d"] = np.nan

    # 4h indicators
    df4["ema_fast"] = ta.ema(df4["close"], length=12)
    df4["ema_slow"] = ta.ema(df4["close"], length=26)
    df4["rsi"]      = ta.rsi(df4["close"], length=14)
    df4["atr"]      = ta.atr(df4["high"], df4["low"], df4["close"], length=14)

    adx4 = ta.adx(df4["high"], df4["low"], df4["close"], length=14)
    if adx4 is not None and not adx4.empty:
        col4 = _first_adx_col(adx4)
        df4["adx4h"] = adx4[col4] if col4 else 0.0
    else:
        df4["adx4h"] = 0.0

    # Bring daily ADX to 4h index
    df4["adx1d"] = dfd["adx1d"].reindex(df4.index, method="ffill")

    df4 = df4.dropna(subset=["open","high","low","close","ema_fast","ema_slow","rsi","atr","adx4h"])
    return df4

def make_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target"] = np.sign(out["close"].shift(-1) - out["close"])
    out["target"] = out["target"].fillna(0).astype(int)
    out = out.dropna()
    return out

# -----------------------------
# Model
# -----------------------------
def train_model(df: pd.DataFrame) -> Pipeline:
    X = df[FEATURES]
    y = df["target"]
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )),
    ])
    pipe.fit(X, y)
    return pipe

def add_probs(model: Pipeline, df_test: pd.DataFrame) -> pd.DataFrame:
    df = df_test.copy()
    probs = model.predict_proba(df[FEATURES])
    classes = list(model.classes_)
    try:
        up_idx = classes.index(1)
        df["prob_up"] = probs[:, up_idx]
    except ValueError:
        # class 1 not present (rare), fallback to 0.5
        df["prob_up"] = 0.5
    return df

# -----------------------------
# Backtest (long-only, ATR)
# -----------------------------
def backtest_long(df: pd.DataFrame, p: dict, days: int) -> dict:
    """
    df must contain: open, high, low, close, atr, adx4h, adx1d, ema_slow, prob_up
    Returns dict: net, pf, win_rate, trades, mdd, score
    """
    if days * 6 > len(df):
        # not enough bars
        return {"net": 0.0, "pf": 0.0, "win_rate": 0.0, "trades": 0, "mdd": 0.0, "score": 0.0}
    slice_ = df.tail(days * 6).copy()

    equity_closed = 10000.0
    trades = []
    pos = None
    equity_curve = []

    for _, r in slice_.iterrows():
        # open
        if pos is None:
            cond = (
                (r["prob_up"] > p["threshold"]) and
                (r["close"] > r["ema_slow"]) and
                (r["adx4h"] >= p["adx4_min"]) and
                (pd.isna(r["adx1d"]) or r["adx1d"] >= p["adx1d_min"])
            )
            if cond:
                atr0 = r["atr"]
                if atr0 <= 0:
                    equity_curve.append(equity_closed)
                    continue
                entry = r["open"] * (1 + SLIP)
                stop  = entry - p["sl_atr_mul"] * atr0
                tp1   = entry + p["tp1_atr_mul"] * atr0
                tp2   = entry + p["tp2_atr_mul"] * atr0
                if stop >= entry:
                    equity_curve.append(equity_closed)
                    continue
                sz = (equity_closed * p["risk_perc"]) / (entry - stop)

                # entry cost (book as a trade for PF/WR accounting)
                entry_cost = -entry * sz * COST
                trades.append(entry_cost)
                equity_closed += entry_cost

                pos = {"e": entry, "s": stop, "t1": tp1, "t2": tp2,
                       "sz": sz, "hm": entry, "p1": False, "atr0": atr0}
        else:
            # manage
            pos["hm"] = max(pos["hm"], r["high"])
            exit_price = None

            # partial at TP1
            if (not pos["p1"]) and (r["high"] >= pos["t1"]):
                part_sz = pos["sz"] * p["partial_pct"]
                pnl = (pos["t1"] - pos["e"]) * part_sz
                cost = pos["t1"] * part_sz * COST
                trades.append(pnl - cost)
                equity_closed += (pnl - cost)
                pos["sz"] *= (1 - p["partial_pct"])
                pos["p1"] = True

            # TP2
            if r["high"] >= pos["t2"]:
                exit_price = pos["t2"]

            # trailing stop
            new_stop = pos["hm"] - p["trail_mul"] * pos["atr0"]
            if new_stop > pos["s"]:
                pos["s"] = new_stop
            if r["low"] <= pos["s"]:
                exit_price = pos["s"]

            if exit_price is not None:
                pnl = (exit_price - pos["e"]) * pos["sz"]
                cost = exit_price * pos["sz"] * COST
                trades.append(pnl - cost)
                equity_closed += (pnl - cost)
                pos = None

        # mark-to-market
        unreal = 0.0
        if pos is not None:
            unreal = (r["close"] - pos["e"]) * pos["sz"]
        equity_curve.append(equity_closed + unreal)

    # metrics
    if len(trades) == 0:
        return {"net": 0.0, "pf": 0.0, "win_rate": 0.0, "trades": 0, "mdd": 0.0, "score": 0.0}

    arr = np.array(trades, dtype=float)
    net = float(arr.sum())

    eq = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq) if eq.size else np.array([0.0])
    mdd = float((peak - eq).max()) if eq.size else 0.0

    gains = arr[arr > 0]
    losses = arr[arr < 0]
    pf = float(gains.sum() / abs(losses.sum())) if losses.size else (np.inf if gains.size else 0.0)
    win_rate = float(len(gains) / len(arr) * 100)
    score = float(net / (mdd + 1.0))
    return {"net": net, "pf": pf, "win_rate": win_rate, "trades": int(len(arr)), "mdd": mdd, "score": score}

# -----------------------------
# Evaluation helpers
# -----------------------------
def print_row(prefix: str, r: dict):
    print(f"{prefix} → Net {r['net']:.2f}, PF {r['pf']:.2f}, Win% {r['win_rate']:.2f}, "
          f"Trades {r['trades']}, MDD {r['mdd']:.2f}, Score {r['score']:.2f}")

def walk_forward(df_all: pd.DataFrame, p: dict, horizon_days: int, folds: int = 3) -> pd.DataFrame:
    """
    Simple non-overlapping WF on test segment.
    """
    bars_per_day = 6
    horizon_bars = horizon_days * bars_per_day

    # Use last 30% as test pool, then split into non-overlapping folds
    split = int(len(df_all) * 0.7)
    test = df_all.iloc[split:].copy()
    n = len(test)
    k = min(folds, max(1, n // horizon_bars))
    if k < 1:
        return pd.DataFrame()

    rows = []
    for i in range(k):
        seg = test.iloc[max(0, n - (k - i) * horizon_bars): n - (k - 1 - i) * horizon_bars]
        if seg.empty or len(seg) < horizon_bars // 2:
            continue
        # Evaluate using the locked params (no re-train inside fold; model already trained)
        res = backtest_long(seg, p, days=min(horizon_days, len(seg)//bars_per_day))
        rows.append({
            "fold_start": seg.index[0],
            **res
        })
    return pd.DataFrame(rows)

# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Data + features
    df0 = download_data()
    df0 = make_target(df0)

    # 2) Train/test split (time-based)
    split = int(len(df0) * 0.7)
    train, test = df0.iloc[:split], df0.iloc[split:]
    if train.empty or test.empty:
        sys.exit("Not enough data after split.")

    # 3) Train once on train
    model = train_model(train)

    # 4) Add probs to test
    test_probs = add_probs(model, test)

    # 5) Baseline OOS with Profile A
    print("\n— Baseline (Profile A locked) —\n")
    for d in (90, 180, 365):
        res = backtest_long(test_probs, PROFILE_A, d)
        print_row(f"{d}d", res)

    # 6) Walk-forward (use same test probs; split into folds)
    print("\n— Walk-forward (non-overlapping) —")
    wf90 = walk_forward(test_probs, PROFILE_A, horizon_days=90, folds=3)
    if not wf90.empty:
        print("\n90d folds:")
        print(wf90)

    wf180 = walk_forward(test_probs, PROFILE_A, horizon_days=180, folds=1)
    if not wf180.empty:
        print("\n180d folds:")
        print(wf180)

    # 7) Save locked params for reference
    os.makedirs("reports", exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = f"reports/locked_profileA_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(PROFILE_A, f, indent=2)
    print(f"\nSaved locked params → {out_path}")

if __name__ == "__main__":
    main()