# forward_test_diamante_perla_negra_365d.py
# Long-only, 4h, ATR-based risk. Trains once on 70% (in-sample),
# forward-tests on the last 30% (out-of-sample) for 90/180/365 days.

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

# =========================
# Settings
# =========================
SYMBOL       = "BTC-USD"
PERIOD_4H    = "730d"   # yfinance limit for 4h history
INTERVAL_4H  = "4h"
PERIOD_1D    = "800d"
INTERVAL_1D  = "1d"

# Core features used by the model
FEATURES     = ["ema_fast","ema_slow","rsi","atr","adx4h","adx_daily"]

# Trading frictions & sizing
COST         = 0.0002    # taker fee approx
SLIP         = 0.0001    # slippage factor
RISK_PERC    = 0.01      # 1% risk per trade (position sized by stop distance)

# Baseline “Diamante” + ATR risk params (tweak as needed)
PARAMS = {
    # entry quality filters
    "threshold": 0.51,   # ML prob_up threshold
    "adx4_min":  6,      # 4h ADX min
    "adx1d_min": 3,      # 1D ADX min

    # ATR-based risk
    "sl_atr_mul": 3.0,
    "tp1_atr_mul": 1.8,
    "tp2_atr_mul": 6.0,
    "trail_mul":  3.0,
    "partial_pct": 0.45,

    # indicator lengths (classic Diamante base)
    "ema_fast_len": 12,
    "ema_slow_len": 26,
    "rsi_len":      14,
    "atr_len":      14,
    "adx_len":      14,
    "adx_daily_len":14
}

# =========================
# Data
# =========================
def load_data():
    print("🔄 Downloading 4h + 1D data…")
    df4 = yf.download(SYMBOL, period=PERIOD_4H, interval=INTERVAL_4H,
                      progress=False, auto_adjust=False)
    if df4.empty:
        sys.exit("❌ Error downloading 4h data")

    if isinstance(df4.columns, pd.MultiIndex):
        df4.columns = df4.columns.get_level_values(0)
    df4.columns = df4.columns.str.lower()
    df4.index = pd.to_datetime(df4.index).tz_localize(None)
    df4 = df4[['open','high','low','close','volume']].dropna()

    dfd = yf.download(SYMBOL, period=PERIOD_1D, interval=INTERVAL_1D,
                      progress=False, auto_adjust=False)
    if dfd.empty:
        sys.exit("❌ Error downloading 1d data")

    if isinstance(dfd.columns, pd.MultiIndex):
        dfd.columns = dfd.columns.get_level_values(0)
    dfd.columns = dfd.columns.str.lower()
    dfd.index = pd.to_datetime(dfd.index).tz_localize(None)
    dfd = dfd[['open','high','low','close','volume']].dropna()

    # Daily ADX
    adx_daily = ta.adx(dfd['high'], dfd['low'], dfd['close'], length=PARAMS['adx_daily_len'])
    if isinstance(adx_daily, pd.DataFrame) and not adx_daily.empty:
        # pick the ADX series among ['ADX_xx','DMP_xx','DMN_xx']
        adx_col = next((c for c in adx_daily.columns if 'adx' in c.lower()), None)
        dfd['adx_daily'] = adx_daily[adx_col] if adx_col else np.nan
    else:
        dfd['adx_daily'] = np.nan

    # align daily ADX to 4h index
    df4['adx_daily'] = dfd['adx_daily'].reindex(df4.index, method='ffill')
    return df4.dropna()

# =========================
# Features & Target
# =========================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ta.ema(df["close"], length=PARAMS['ema_fast_len'])
    df["ema_slow"] = ta.ema(df["close"], length=PARAMS['ema_slow_len'])
    df["rsi"]      = ta.rsi(df["close"], length=PARAMS['rsi_len'])
    df["atr"]      = ta.atr(df["high"], df["low"], df["close"], length=PARAMS['atr_len'])

    adx4 = ta.adx(df["high"], df["low"], df["close"], length=PARAMS['adx_len'])
    if isinstance(adx4, pd.DataFrame) and not adx4.empty:
        adx4_col = next((c for c in adx4.columns if 'adx' in c.lower()), None)
        df["adx4h"] = adx4[adx4_col] if adx4_col else 0.0
    else:
        df["adx4h"] = 0.0

    return df.dropna()

def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    NaN-safe target: sign of next-bar close change.
    Drop NaNs before casting to int to avoid IntCastingNaNError.
    """
    df = df.copy()
    diff = df["close"].shift(-1) - df["close"]
    df["target"] = np.sign(diff)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)
    # Also ensure no leftover NaNs from indicators
    return df.dropna()

# =========================
# Model
# =========================
def train_model(train_df: pd.DataFrame):
    X = train_df[FEATURES]
    y = train_df["target"]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
    pipe.fit(X, y)
    return pipe

def add_predictions(model, test_df: pd.DataFrame) -> pd.DataFrame:
    df = test_df.copy()
    probs = model.predict_proba(df[FEATURES])
    classes = list(model.classes_)
    # Probability of "up" class (1). If class 1 not present, fallback 0.5.
    if 1 in classes:
        up_idx = classes.index(1)
        df["prob_up"] = probs[:, up_idx]
    else:
        df["prob_up"] = 0.5
    return df

# =========================
# Backtest (long-only, ATR risk)
# =========================
def backtest_long_atr(df: pd.DataFrame, days: int = 365) -> dict:
    """
    Run long-only ATR backtest on the last `days` (6 * days bars @ 4h).
    Entry if:
      - prob_up > threshold
      - close > ema_slow
      - adx4h >= adx4_min AND adx_daily >= adx1d_min
    Risk:
      - SL = entry - sl_atr_mul * ATR
      - TP1 = entry + tp1_atr_mul * ATR (partial take)
      - TP2 = entry + tp2_atr_mul * ATR (final take)
      - Trailing = highest_high - trail_mul * ATR_at_entry
    """
    n = days * 6
    data = df.tail(n)

    equity = 10000.0
    trades = []
    pos = None  # {'e','s','t1','t2','sz','hm','p1','atr0'}

    for _, r in data.iterrows():
        if pos is None:
            if (
                (r["prob_up"] > PARAMS["threshold"]) and
                (r["close"] > r["ema_slow"]) and
                (r["adx4h"] >= PARAMS["adx4_min"]) and
                (r["adx_daily"] >= PARAMS["adx1d_min"])
            ):
                entry = r["open"] * (1 + SLIP)
                atr0  = r["atr"]
                if atr0 <= 0:
                    continue
                stop  = entry - PARAMS["sl_atr_mul"] * atr0
                tp1   = entry + PARAMS["tp1_atr_mul"] * atr0
                tp2   = entry + PARAMS["tp2_atr_mul"] * atr0
                if stop >= entry:
                    continue
                size  = (equity * RISK_PERC) / (entry - stop)
                equity -= entry * size * COST  # entry fee
                pos = {"e":entry, "s":stop, "t1":tp1, "t2":tp2,
                       "sz":size, "hm":entry, "p1":False, "atr0":atr0}
        else:
            # update highest mark for trailing
            pos["hm"] = max(pos["hm"], r["high"])

            exit_p = None
            # partial take at TP1
            if (not pos["p1"]) and (r["high"] >= pos["t1"]):
                pnl_partial = (pos["t1"] - pos["e"]) * (pos["sz"] * PARAMS["partial_pct"])
                equity += pnl_partial
                trades.append(pnl_partial)
                pos["sz"] *= (1 - PARAMS["partial_pct"])
                pos["p1"] = True

            # full TP2
            if r["high"] >= pos["t2"]:
                exit_p = pos["t2"]

            # trailing stop (using ATR at entry)
            new_stop = pos["hm"] - PARAMS["trail_mul"] * pos["atr0"]
            if new_stop > pos["s"]:
                pos["s"] = new_stop

            if r["low"] <= pos["s"]:
                exit_p = pos["s"]

            if exit_p is not None:
                pnl = (exit_p - pos["e"]) * pos["sz"] - exit_p * pos["sz"] * COST
                equity += pnl
                trades.append(pnl)
                pos = None

    # Metrics
    if not trades:
        return {"net": 0.0, "pf": 0.0, "win_rate": 0.0, "trades": 0, "max_dd": 0.0}

    arr = np.array(trades, dtype=float)
    net = arr.sum()
    wins, losses = arr[arr > 0], arr[arr < 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.size else np.inf
    wr = (len(wins) / len(arr)) * 100.0

    eq_curve = 10000.0 + np.cumsum(arr)
    max_dd = float((np.maximum.accumulate(eq_curve) - eq_curve).max())
    return {"net": net, "pf": pf, "win_rate": wr, "trades": len(arr), "max_dd": max_dd}

# =========================
# Main
# =========================
def main():
    df0 = load_data()
    df1 = add_features(df0)

    # Split 70/30 (IS/OOS)
    split = int(len(df1) * 0.7)
    train_raw = df1.iloc[:split].copy()
    test_raw  = df1.iloc[split:].copy()

    # Build target on train only (avoid leakage)
    train = make_target(train_raw)

    # Train model once on IS
    model = train_model(train)

    # OOS predictions on test set
    test = add_predictions(model, test_raw)

    # Forward tests
    for d in (90, 180, 365):
        res = backtest_long_atr(test, days=d)
        print(f"{d}d → Net: ${res['net']:.2f}, PF: {res['pf']:.2f}, "
              f"Win%: {res['win_rate']:.2f}, Trades: {res['trades']}, MDD: {res['max_dd']:.2f}")

if __name__ == "__main__":
    main()