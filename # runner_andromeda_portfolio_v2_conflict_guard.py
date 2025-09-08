# runner_andromeda_portfolio_v2.py — con conflict-guard adaptativo y Net Profit visible

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import joblib
from sklearn.ensemble import RandomForestClassifier

# ==============================
# Parámetros base Andromeda
# ==============================
MODEL_FILE_LONG = "andromeda_long.pkl"
MODEL_FILE_SHORT = "andromeda_short.pkl"

PARAMS_LONG = {
    "THRESHOLD": 0.55,
    "COST": 0.0002,
    "SLIP": 0.0001,
}
PARAMS_SHORT = {
    "THRESHOLD": 0.55,
    "COST": 0.0002,
    "SLIP": 0.0001,
}

FEATURES = [
    "vola_10", "adx4", "ema_200", "ema_50",
    "obv_z", "atr_norm", "ema_ratio",
    "bb_width", "atr", "rsi"
]

# ==============================
# Funciones comunes
# ==============================
def load_data():
    print("🔄 Loading 4h data (ccxt + fallbacks + yfinance)…")
    df = yf.download("BTC-USD", interval="4h", period="730d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df.dropna()

def add_features(df):
    print("✨ Adding features…")
    df["ema_50"] = ta.ema(df["close"], 50)
    df["ema_200"] = ta.ema(df["close"], 200)
    df["ema_ratio"] = df["ema_50"] / df["ema_200"] - 1
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], 14)
    df["atr_norm"] = df["atr"] / df["close"]
    adx_df = ta.adx(df["high"], df["low"], df["close"], 14)
    df["adx4"] = adx_df[adx_df.columns[0]] if adx_df is not None else np.nan
    df["rsi"] = ta.rsi(df["close"], 14)
    bb = ta.bbands(df["close"], 20)
    if bb is not None:
        df["bb_width"] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / df["close"]
    df["vola_10"] = df["close"].pct_change().rolling(10).std()
    obv = ta.obv(df["close"], df["volume"])
    df["obv_z"] = (obv - obv.rolling(20).mean()) / obv.rolling(20).std()
    return df.dropna()

def train_model(df, target_col, model_file):
    split = int(len(df) * 0.7)
    train, test = df.iloc[:split], df.iloc[split:]
    X_train, y_train = train[FEATURES], train[target_col]
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)
    joblib.dump((model, FEATURES), model_file)
    return model, test

def backtest_combo(df, pL, pS, wL=1.0, wS=1.0):
    bal, trades, wins, losses = 0.0, [], [], []
    eq_curve = []

    for _, r in df.iterrows():
        pnl = 0.0
        if r.get("long_sig", False):
            pnl += wL * ((r["close"] - r["open"]) - r["open"] * (pL["COST"] + pL["SLIP"]))
        if r.get("short_sig", False):
            pnl += wS * ((r["open"] - r["close"]) - r["open"] * (pS["COST"] + pS["SLIP"]))
        if pnl != 0:
            trades.append(pnl)
            bal += pnl
            eq_curve.append(bal)
            if pnl > 0:
                wins.append(pnl)
            elif pnl < 0:
                losses.append(pnl)

    pf = sum(wins) / abs(sum(losses)) if losses else np.inf
    mdd = 0
    if eq_curve:
        peak = eq_curve[0]
        for x in eq_curve:
            peak = max(peak, x)
            mdd = min(mdd, x - peak)

    score = (bal / abs(mdd)) * (pf if pf != np.inf else 1) if mdd != 0 else bal
    return {
        "net": bal,
        "trades": len(trades),
        "wr": len(wins) / len(trades) * 100 if trades else 0,
        "pf": pf,
        "mdd": abs(mdd),
        "score": score
    }

# ==============================
# Main runner con conflict-guard adaptativo
# ==============================
def main():
    raw = load_data()
    feats = add_features(raw)

    feats_long = feats.copy()
    feats_long["target_long"] = (feats_long["close"].shift(-1) > feats_long["close"]).astype(int)
    model_long, test_long = train_model(feats_long.dropna(), "target_long", MODEL_FILE_LONG)
    proba_long = model_long.predict_proba(test_long[FEATURES])[:, 1]
    test_long["long_sig"] = proba_long > PARAMS_LONG["THRESHOLD"]

    feats_short = feats.copy()
    feats_short["target_short"] = (feats_short["close"].shift(-1) < feats_short["close"]).astype(int)
    model_short, test_short = train_model(feats_short.dropna(), "target_short", MODEL_FILE_SHORT)
    proba_short = model_short.predict_proba(test_short[FEATURES])[:, 1]
    test_short["short_sig"] = proba_short > PARAMS_SHORT["THRESHOLD"]

    # unificar test para combo
    test = test_long.copy()
    test["short_sig"] = test_short["short_sig"]

    # medir piernas individuales
    soloL = test.copy(); soloL["short_sig"] = False
    rL = backtest_combo(soloL, PARAMS_LONG, PARAMS_SHORT, wL=1.0, wS=0.0)

    soloS = test.copy(); soloS["long_sig"] = False
    rS = backtest_combo(soloS, PARAMS_LONG, PARAMS_SHORT, wL=0.0, wS=1.0)

    mdd_cap = min(rL["mdd"], rS["mdd"]) * 1.05 if (rL["trades"] > 0 and rS["trades"] > 0) else max(rL["mdd"], rS["mdd"])

    # baseline (wL=1, wS=1)
    base_r = backtest_combo(test, PARAMS_LONG, PARAMS_SHORT, wL=1.0, wS=1.0)

    # búsqueda adaptativa
    def search_weights():
        best = None
        wL_vals = [1.0, 0.85, 0.75, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
        wS_vals = [1.0, 0.9, 0.85, 0.75, 0.6, 0.5]
        for wL in wL_vals:
            for wS in wS_vals:
                r = backtest_combo(test, PARAMS_LONG, PARAMS_SHORT, wL=wL, wS=wS)
                feasible = (r["mdd"] <= mdd_cap)
                if feasible and (best is None or r["score"] > best["score"]):
                    best = {"wL": wL, "wS": wS, **r, "feasible": feasible}
        return best

    best = search_weights() or {"wL": 1.0, "wS": 1.0, **base_r, "feasible": False}

    # Mostrar resultados
    print("\n— Portfolio with conflict-guard (OOS zone) —")
    print(f"Solo LONG  → Net {rL['net']:.2f}, PF {rL['pf']:.2f}, WR {rL['wr']:.2f}%, Trades {rL['trades']}, MDD {rL['mdd']:.2f}, Score {rL['score']:.2f}")
    print(f"Solo SHORT → Net {rS['net']:.2f}, PF {rS['pf']:.2f}, WR {rS['wr']:.2f}%, Trades {rS['trades']}, MDD {rS['mdd']:.2f}, Score {rS['score']:.2f}")
    print(f"Baseline   → Net {base_r['net']:.2f}, PF {base_r['pf']:.2f}, WR {base_r['wr']:.2f}%, Trades {base_r['trades']}, MDD {base_r['mdd']:.2f}, Score {base_r['score']:.2f}")
    print(f"Selected   → wL {best['wL']:.2f}, wS {best['wS']:.2f} | Net {best['net']:.2f}, PF {best['pf']:.2f}, WR {best['wr']:.2f}%, Trades {best['trades']}, MDD {best['mdd']:.2f}, Score {best['score']:.2f} | feasible={best['feasible']} | cap={mdd_cap:.2f}")

if __name__ == "__main__":
    main()