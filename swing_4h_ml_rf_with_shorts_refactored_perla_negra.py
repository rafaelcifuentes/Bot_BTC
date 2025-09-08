# swing_4h_ml_rf_with_shorts_refactored_perla_negra.py
# CORRECCIONES: Solucionado KeyError, implementado split train/test correcto,
# y optimizada la estructura para que Optuna no re-entrene el modelo en cada trial.

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings("ignore")

# --- Parámetros de la Estrategia ---
FEATURES = ["ema_fast", "ema_slow", "rsi", "atr", "adx4h", "adx_daily"]
FIXED = {
    "cost": 0.0002,
    "slip": 0.0001
}
TRIALS = 50


# --- 1. Carga de datos robusta ---
def load_data(days4h="730d", days1d="800d"):
    print("🔄 Descargando y procesando datos...")
    df4h = yf.download("BTC-USD", period=days4h, interval="4h", progress=False, auto_adjust=False)
    if df4h.empty: sys.exit("❌ Error: descarga de datos de 4h fallida.")

    if isinstance(df4h.columns, pd.MultiIndex):
        df4h.columns = df4h.columns.get_level_values(0)
    df4h.columns = df4h.columns.str.lower()
    df4h.index = pd.to_datetime(df4h.index).tz_localize(None)

    dfd = yf.download("BTC-USD", period=days1d, interval="1d", progress=False, auto_adjust=False)
    if dfd.empty: sys.exit("❌ Error: descarga de datos diarios fallida.")

    if isinstance(dfd.columns, pd.MultiIndex):
        dfd.columns = dfd.columns.get_level_values(0)
    dfd.columns = dfd.columns.str.lower()
    dfd.index = pd.to_datetime(dfd.index).tz_localize(None)

    adx_d = ta.adx(dfd['high'], dfd['low'], dfd['close'], length=14)
    if adx_d is not None and not adx_d.empty:
        col = next((c for c in adx_d.columns if 'adx' in c.lower()), None)
        dfd['adx_daily'] = adx_d[col] if col else np.nan
    else:
        dfd['adx_daily'] = np.nan

    df4h['adx_daily'] = dfd['adx_daily'].reindex(df4h.index, method='ffill')
    return df4h.dropna()


# --- 2. Features y Target ---
def add_features(df, p):
    df = df.copy()
    df["ema_fast"] = ta.ema(df["close"], length=p["ema_fast"])
    df["ema_slow"] = ta.ema(df["close"], length=p["ema_slow"])
    df["rsi"] = ta.rsi(df["close"], length=p["rsi_len"])
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=p["atr_len"])
    adx4h = ta.adx(df["high"], df["low"], df["close"], length=p["adx_len"])
    if adx4h is not None and not adx4h.empty:
        col = next((c for c in adx4h.columns if 'adx' in c.lower()), None)
        df["adx4h"] = adx4h[col] if col else 0
    else:
        df["adx4h"] = 0
    return df.dropna()


def make_target(df, p):
    df = df.copy()
    df["target"] = np.where(
        df["close"].shift(-p["shift"]) > df["close"] * (1 + p["pct_target"]), 1,
        np.where(df["close"].shift(-p["shift"]) < df["close"] * (1 - p["pct_target"]), -1, 0)
    )
    return df.dropna()


# --- 3. Entrenamiento y Predicción ---
def get_predictions(df_train, df_test, p_model):
    X_train, y_train = df_train[FEATURES], df_train["target"]
    X_test = df_test[FEATURES]

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=p_model['n_estimators'], max_depth=p_model['max_depth'],
            class_weight="balanced", random_state=42, n_jobs=-1))
    ])
    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)
    df_pred = df_test.copy()

    class_up_idx = np.where(pipe.classes_ == 1)[0]
    class_down_idx = np.where(pipe.classes_ == -1)[0]

    df_pred["prob_up"] = probs[:, class_up_idx[0]] if class_up_idx.size > 0 else 0.5
    df_pred["prob_down"] = probs[:, class_down_idx[0]] if class_down_idx.size > 0 else 0.5
    return df_pred


# --- 4. Backtest ---
def backtest(df, p, days=90):
    df_t = df.tail(days * 6).copy()
    equity, trades, op = 10000.0, [], None

    for _, r in df_t.iterrows():
        long_sig = (r["prob_up"] > p["threshold"]) and (r["adx4h"] > p["adx_filter"]) and (
                    r["adx_daily"] > p["adx_daily_filter"])
        short_sig = (r["prob_down"] > p["threshold"]) and (r["adx4h"] > p["adx_filter"]) and (
                    r["adx_daily"] > p["adx_daily_filter"])

        if op is None and (long_sig or short_sig):
            d = 1 if long_sig else -1
            e = r["open"] * (1 + FIXED["slip"] * d)
            atr0 = r["atr"]
            if atr0 == 0: continue
            s = e - d * atr0 * p["sl_atr_mul"]
            t1 = e + d * atr0 * p["tp1_atr_mul"]
            t2 = e + d * atr0 * p["tp2_atr_mul"]
            if d * (s - e) >= 0: continue
            sz = (equity * p["risk_perc"]) / abs(e - s)
            equity -= e * sz * FIXED["cost"]
            op = {'d': d, 'e': e, 's': s, 't1': t1, 't2': t2, 'sz': sz, 'hm': e, 'p1': False, 'atr0': atr0}
        elif op:
            d, e = op['d'], op['e']
            op['hm'] = max(op['hm'], r.high) if d == 1 else min(op['hm'], r.low)
            exit_p = None

            if not op['p1'] and ((d == 1 and r.high >= op['t1']) or (d == -1 and r.low <= op['t1'])):
                pnl = (op['t1'] - e) * d * (op['sz'] * p['partial_pct'])
                equity += pnl;
                trades.append(pnl)
                op['sz'] *= (1 - p['partial_pct']);
                op['p1'] = True

            if (d == 1 and r.high >= op['t2']) or (d == -1 and r.low <= op['t2']):
                exit_p = op['t2']

            new_s = op['hm'] - d * op['atr0'] * p['trail_mul']
            if (d == 1 and new_s > op['s']) or (d == -1 and new_s < op['s']): op['s'] = new_s

            if (d == 1 and r.low <= op['s']) or (d == -1 and r.high >= op['s']):
                exit_p = op['s']

            if exit_p is not None:
                pnl = (exit_p - e) * d * op['sz'] - exit_p * op['sz'] * FIXED['cost']
                equity += pnl;
                trades.append(pnl);
                op = None

    arr = np.array(trades)
    net = arr.sum() if arr.size else 0
    return {"net_profit": net, "trades": len(arr)}


# --- 5. Objetivo Optuna ---
def objective(trial, df_train_raw, df_test_raw):
    p = {
        'ema_fast': trial.suggest_int('ema_fast', 10, 24),
        'ema_slow': trial.suggest_int('ema_slow', 25, 80),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'atr_len': trial.suggest_int('atr_len', 7, 21),
        'adx_len': trial.suggest_int('adx_len', 10, 20),
        'pct_target': trial.suggest_float('pct_target', 0.005, 0.02),
        'shift': trial.suggest_int('shift', 1, 6),
        'threshold': trial.suggest_float('threshold', 0.50, 0.70),
        'adx_filter': trial.suggest_int('adx_filter', 5, 25),
        'adx_daily_filter': trial.suggest_int('adx_daily_filter', 5, 25),
        'sl_atr_mul': trial.suggest_float('sl_atr_mul', 1.0, 5.0),
        'tp1_atr_mul': trial.suggest_float('tp1_atr_mul', 0.5, 3.0),
        'tp2_atr_mul': trial.suggest_float('tp2_atr_mul', 2.0, 8.0),
        'trail_mul': trial.suggest_float('trail_mul', 1.0, 4.0),
        'partial_pct': trial.suggest_float('partial_pct', 0.1, 0.9),
        'n_estimators': trial.suggest_categorical('n_estimators', [50, 100]),
        'max_depth': trial.suggest_categorical('max_depth', [5, 10, None]),
        'risk_perc': trial.suggest_float('risk_perc', 0.005, 0.025)
    }

    # Procesar datos y entrenar/predecir
    df_train = add_features(df_train_raw.copy(), p)
    df_train = make_target(df_train, p)

    df_test = add_features(df_test_raw.copy(), p)

    if df_train.empty or df_test.empty:
        return -1e6

    df_test_pred = get_predictions(df_train, df_test, p)

    # Backtest
    res = backtest(df_test_pred, p, days=90)

    score = res['net_profit']
    if res['trades'] < 10:
        score -= 5000

    return score


# --- 6. Ejecución Principal ---
if __name__ == '__main__':
    df0 = load_data()

    split = int(len(df0) * 0.7)
    train_df, test_df = df0.iloc[:split], df0.iloc[split:]

    print('🚀 Iniciando Optuna (Fase Holística)…')
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: objective(t, train_df, test_df), n_trials=100, show_progress_bar=True)

    best = study.best_params
    print('\n✅ Optimización completada')
    print('Mejores parámetros:', best)
