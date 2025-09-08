# swing_4h_ml_rf_with_shorts_refactored_perla_negra.py
# CORRECCIONES: Solucionado AttributeError, implementado split train/test correcto para evitar
# data leakage, y estandarizada la carga de datos.

import os
import sys
import json
import argparse
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

# --------------------------
# Parámetros FIJOS
# --------------------------
FEATURES = ["ema_fast", "ema_slow", "rsi", "atr", "adx4h", "adx_daily"]
REPORT_DIR = "./reports"


# --------------------------
# Carga de datos
# --------------------------
def load_data(symbol_4h="BTC-USD", period_4h="1500d", symbol_1d="BTC-USD", period_1d="1500d") -> pd.DataFrame:
    print("🔄 Descargando datos (yfinance)…")
    # Carga de datos de 4h
    df4h = yf.download(symbol_4h, period=period_4h, interval="4h", progress=False, auto_adjust=False)
    if df4h.empty:
        raise RuntimeError("No se pudo descargar 4h.")
    if isinstance(df4h.columns, pd.MultiIndex):
        df4h.columns = df4h.columns.get_level_values(0)
    df4h.columns = df4h.columns.str.lower()
    df4h.index = pd.to_datetime(df4h.index, utc=True).tz_convert(None)

    # Carga de datos diarios
    dfd = yf.download(symbol_1d, period=period_1d, interval="1d", progress=False, auto_adjust=False)
    if dfd.empty:
        raise RuntimeError("No se pudo descargar 1d.")
    if isinstance(dfd.columns, pd.MultiIndex):
        dfd.columns = dfd.columns.get_level_values(0)
    # <<< CORRECCIÓN: Estandarizar columnas del DataFrame diario >>>
    dfd.columns = dfd.columns.str.lower()
    dfd.index = pd.to_datetime(dfd.index, utc=True).tz_convert(None)

    return df4h, dfd


# --------------------------
# Features + modelo
# --------------------------
def add_features(df4h: pd.DataFrame, dfd: pd.DataFrame) -> pd.DataFrame:
    df = df4h.copy()
    # Indicadores 4h
    df["ema_fast"] = ta.ema(df["close"], length=12)
    df["ema_slow"] = ta.ema(df["close"], length=26)
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    adx4 = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx4 is not None and not adx4.empty:
        adx_col_4h = next((c for c in adx4.columns if "adx" in c.lower()), None)
        df["adx4h"] = adx4[adx_col_4h] if adx_col_4h else 0.0
    else:
        df["adx4h"] = 0.0

    # ADX diario mapeado a 4h
    adx1d = ta.adx(dfd["high"], dfd["low"], dfd["close"], length=14)
    if adx1d is not None and not adx1d.empty:
        adx_col_1d = next((c for c in adx1d.columns if "adx" in c.lower()), None)
        dfd["adx_daily"] = adx1d[adx_col_1d] if adx_col_1d else np.nan
    else:
        dfd["adx_daily"] = np.nan
    df["adx_daily"] = dfd["adx_daily"].reindex(df.index, method="ffill")
    return df.dropna()


def make_target_for_rf(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target"] = np.sign(out["close"].shift(-1) - out["close"]).astype(int)
    return out.dropna()


def train_model(train_df: pd.DataFrame) -> Pipeline:
    X = train_df[FEATURES]
    y = train_df["target"]
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=None,
                                      class_weight="balanced_subsample", random_state=42, n_jobs=-1))
    ])
    pipe.fit(X, y)
    return pipe


def predict_probs(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES]
    probs = model.predict_proba(X)
    classes = model.classes_
    up_idx = np.where(classes == 1)[0]
    down_idx = np.where(classes == -1)[0]
    out = df.copy()
    out["prob_up"] = probs[:, up_idx[0]] if up_idx.size > 0 else 0.5
    out["prob_down"] = probs[:, down_idx[0]] if down_idx.size > 0 else 0.5
    return out


# --------------------------
# Backtest
# --------------------------
def backtest_percent(df: pd.DataFrame, p: Dict) -> Dict:
    equity, trades, pos = 10_000.0, [], None
    for _, r in df.iterrows():
        long_sig = r.get("prob_up", 0.5) > p['threshold']
        short_sig = r.get("prob_down", 0.5) > p['threshold']

        if pos is None and (long_sig or short_sig):
            d = 1 if long_sig else -1
            e = float(r["open"]) * (1 + p['slip'] * d)
            sl = e * (1 - p['sl_pct'] * d)
            tp = e * (1 + p['tp_pct'] * d)
            risk_per_unit = abs(e - sl)
            if risk_per_unit <= 1e-9: continue
            sz = (equity * p['risk_perc']) / risk_per_unit
            equity -= e * sz * p['cost']
            pos = {"d": d, "e": e, "sl": sl, "tp": tp, "tr": e, "sz": sz}
        elif pos is not None:
            d, e = pos["d"], pos["e"]
            if d == 1:
                pos["tr"] = max(pos["tr"], r["high"])
                sl = max(pos["sl"], pos["tr"] * (1 - p['trail_pct']))
            else:
                pos["tr"] = min(pos["tr"], r["low"])
                sl = min(pos["sl"], pos["tr"] * (1 + p['trail_pct']))

            exit_price = None
            if d == 1 and (r["low"] <= sl or r["high"] >= pos["tp"]):
                exit_price = sl if r["low"] <= sl else pos["tp"]
            elif d == -1 and (r["high"] >= sl or r["low"] <= pos["tp"]):
                exit_price = sl if r["high"] >= sl else pos["tp"]

            if exit_price is not None:
                pnl = (exit_price - e) * d * pos["sz"] - exit_price * pos["sz"] * p['cost']
                equity += pnl
                trades.append(pnl)
                pos = None

    arr = np.array(trades, dtype=float)
    return {"trades": arr}


# --------------------------
# Main
# --------------------------
def run_pipeline(p: Dict) -> Dict:
    df4h, dfd = load_data()
    df = add_features(df4h, dfd)
    df = make_target_for_rf(df)

    # --- DIVISIÓN DE DATOS CORRECTA (TRAIN/TEST) ---
    cut = int(len(df) * p['train_frac'])
    train, test = df.iloc[:cut], df.iloc[cut:].copy()

    model = train_model(train)
    test_with_preds = predict_probs(model, test)

    trades_array = backtest_percent(test_with_preds, p)["trades"]

    # Calcular métricas
    if trades_array.size == 0:
        return {"net_profit": 0, "profit_factor": 0, "win_rate_pct": 0, "trades": 0, "max_dd": 0}

    eq_curve = 10_000 + trades_array.cumsum()
    net = eq_curve[-1] - 10_000
    mdd = (np.maximum.accumulate(eq_curve) - eq_curve).max()

    return {
        "net_profit": net,
        "profit_factor": profit_factor(trades_array),
        "win_rate_pct": (trades_array > 0).mean() * 100,
        "trades": len(trades_array),
        "max_dd": mdd
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perla Negra 4h (RF) — backtest OOS.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--sl_pct", type=float, default=SL_PCT)
    parser.add_argument("--tp_pct", type=float, default=TP_PCT)
    parser.add_argument("--trail_pct", type=float, default=TRAIL_PCT)
    parser.add_argument("--risk_perc", type=float, default=RISK_PERC)
    parser.add_argument("--cost", type=float, default=COST)
    parser.add_argument("--slip", type=float, default=SLIP)
    parser.add_argument("--train_frac", type=float, default=0.7)
    args = parser.parse_args()

    print("🚀 Perla Negra — parámetros fijos")
    print({k: v for k, v in vars(args).items()})

    results = run_pipeline(vars(args))

    print("\n— RESULTADOS OOS (Out-of-Sample) —")
    for k, v in results.items():
        print(f"{k.replace('_', ' ').title():<15}: {v:,.2f}")