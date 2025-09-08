# swing_4h_phase1_long_short.py
# Fase 1 – Optimización de la Señal con soporte para largo y corto
# CORRECCIONES: Se añade lógica short, se implementa un split train/test para evitar data leakage,
# y se estandariza el manejo de nombres de columnas y la carga de datos.

import os
import sys
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# --- Parámetros fijos y de la estrategia ---
FIXED = {
    "adx_daily_len": 14
}
SIGNAL_P = {
    "ema_fast": 12,
    "ema_slow": 26,
    "rsi_len": 14,
    "atr_len": 14,
    "adx_len": 14
}
FEATURES = ["ema_fast", "ema_slow", "rsi", "atr", "adx4h", "adx_daily"]


# --- 1. CARGA DE DATOS ROBUSTA ---
def download_data(days4h="730d", days1d="800d"):
    """Descarga y procesa datos de 4h y diarios de forma robusta."""
    print("🔄 Descargando y procesando datos...")

    # Carga de datos de 4h
    df_4h = yf.download("BTC-USD", period=days4h, interval="4h", progress=False, auto_adjust=False)
    if df_4h.empty:
        sys.exit("❌ Error: descarga de datos de 4h fallida.")

    if isinstance(df_4h.columns, pd.MultiIndex):
        df_4h.columns = df_4h.columns.get_level_values(0)
    df_4h.columns = df_4h.columns.str.lower()
    df_4h = df_4h.loc[:, ~df_4h.columns.duplicated()]
    df_4h.index = pd.to_datetime(df_4h.index).tz_localize(None)

    # Carga de datos diarios para contexto
    df_d = yf.download("BTC-USD", period=days1d, interval="1d", progress=False, auto_adjust=False)
    if df_d.empty:
        sys.exit("❌ Error: descarga de datos diarios fallida.")

    if isinstance(df_d.columns, pd.MultiIndex):
        df_d.columns = df_d.columns.get_level_values(0)
    df_d.columns = df_d.columns.str.lower()  # Estandarizar columnas
    df_d = df_d.loc[:, ~df_d.columns.duplicated()]
    df_d.index = pd.to_datetime(df_d.index).tz_localize(None)

    # Añadir ADX diario a los datos de 4h
    adx_d = ta.adx(df_d['high'], df_d['low'], df_d['close'], length=FIXED["adx_daily_len"])
    if adx_d is not None and not adx_d.empty:
        adx_col = next((c for c in adx_d.columns if 'adx' in c.lower()), None)
        df_d['adx_daily'] = adx_d[adx_col] if adx_col else np.nan
    else:
        df_d['adx_daily'] = np.nan

    df_4h['adx_daily'] = df_d['adx_daily'].reindex(df_4h.index, method='ffill')

    return df_4h.dropna()


# --- 2. AÑADIR FEATURES ---
def add_features(df):
    df = df.copy()
    p = SIGNAL_P
    df["ema_fast"] = ta.ema(df["close"], length=p["ema_fast"])
    df["ema_slow"] = ta.ema(df["close"], length=p["ema_slow"])
    df["rsi"] = ta.rsi(df["close"], length=p["rsi_len"])
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=p["atr_len"])
    adx4h = ta.adx(df["high"], df["low"], df["close"], length=p["adx_len"])
    if adx4h is not None and not adx4h.empty:
        adx_col = next((c for c in adx4h.columns if 'adx' in c.lower()), None)
        df["adx4h"] = adx4h[adx_col] if adx_col else 0
    else:
        df["adx4h"] = 0
    return df.dropna()


# --- 3. CREAR TARGET Y ENTRENAR MODELO ---
def make_target(df):
    df = df.copy()
    df["target"] = np.where(
        df["close"].shift(-6) > df["close"] * 1.03, 1,
        np.where(df["close"].shift(-6) < df["close"] * 0.97, -1, 0)
    )
    return df.dropna()


def train_and_tune(X, y):
    """Ajusta RandomForest con GridSearchCV sobre datos de entrenamiento."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=42, class_weight="balanced"))
    ])
    param_grid = {
        "rf__n_estimators": [50, 100],
        "rf__max_depth": [5, 10],
        "rf__min_samples_split": [2, 5]
    }
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(pipe, param_grid, cv=tscv, scoring="accuracy", n_jobs=-1, verbose=1)
    grid.fit(X, y)
    print("Mejores parámetros RF:", grid.best_params_)
    return grid.best_estimator_


# --- 4. BACKTEST LONG/SHORT ---
def backtest_threshold(df, thresholds=(0.55, 0.60, 0.65, 0.70)):
    """Testea distintos umbrales de probabilidad para largos y cortos."""
    results = []

    for thr in thresholds:
        trades = []
        for _, r in df.iterrows():
            p_long = r.get("prob_up", 0.5)
            p_short = r.get("prob_down", 0.5)

            pnl = 0
            if p_long > thr and r["close"] > r["ema_slow"]:
                # Backtest simplificado de 1 vela para largos
                pnl = (r["close"] - r["open"]) - r["close"] * 0.0002
            elif p_short > thr and r["close"] < r["ema_slow"]:
                # Backtest simplificado de 1 vela para cortos
                pnl = (r["open"] - r["close"]) - r["close"] * 0.0002

            if pnl != 0:
                trades.append(pnl)

        arr = np.array(trades)
        if arr.size == 0:
            results.append({"threshold": thr, "trades": 0, "win_rate": 0, "net_profit": 0, "profit_factor": 0})
            continue

        net = arr.sum()
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
        wr = len(wins) / len(arr) * 100

        results.append({
            "threshold": thr, "trades": len(arr), "win_rate": wr,
            "net_profit": net, "profit_factor": pf
        })

    return pd.DataFrame(results).sort_values("net_profit", ascending=False)


# --- 5. EJECUCIÓN PRINCIPAL ---
def main():
    df0 = download_data()
    df1 = add_features(df0)
    df2 = make_target(df1)

    # --- DIVISIÓN DE DATOS CORRECTA PARA EVITAR DATA LEAKAGE ---
    split_idx = int(len(df2) * 0.7)
    df_train = df2.iloc[:split_idx]
    df_test = df2.iloc[split_idx:].copy()

    X_train, y_train = df_train[FEATURES], df_train['target']
    X_test = df_test[FEATURES]

    print("🚀 Entrenando y tuneando el modelo en el conjunto de entrenamiento...")
    model = train_and_tune(X_train, y_train)

    print("\n=== Backtest de Umbrales en Datos Fuera de Muestra (Out-of-Sample) ===")

    # Predecir probabilidades solo en el conjunto de prueba
    if model:
        probs = model.predict_proba(X_test)

        # Encontrar el índice de las columnas para las clases 1 (long) y -1 (short)
        class_1_idx = np.where(model.classes_ == 1)[0]
        class_neg1_idx = np.where(model.classes_ == -1)[0]

        # Asignar probabilidades a nuevas columnas en el DataFrame de prueba
        if class_1_idx.size > 0:
            df_test["prob_up"] = probs[:, class_1_idx[0]]
        if class_neg1_idx.size > 0:
            df_test["prob_down"] = probs[:, class_neg1_idx[0]]

    # Ejecutar el backtest de umbrales en el conjunto de prueba
    df_tune = backtest_threshold(df_test)
    print(df_tune.to_string(index=False))


if __name__ == "__main__":
    main()