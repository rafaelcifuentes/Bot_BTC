# swing_4h_ml_rf_with_shorts_refactored.py
# CORRECCIONES: Se soluciona el RuntimeError de yfinance, se implementa un split
# train/test correcto para evitar data leakage, y se refactoriza la lógica.

import sys
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")

# --- Parámetros de la Estrategia ---
THRESHOLD = 0.55
SL_PCT = 0.025
TP_PCT = 0.05
TRAIL_PCT = 0.01
RISK_PERC = 0.01
COST = 0.0002
SLIP = 0.0001
SEED = 42
np.random.seed(SEED)
FEATURES = ["ema_fast", "ema_slow", "rsi", "atr"]


# --- 1. Carga de datos robusta ---
def load_data(symbol: str = "BTC-USD", interval: str = "4h", period: str = "730d") -> pd.DataFrame:
    """Carga datos de 4h de forma robusta, asegurando nombres de columna en minúsculas."""
    print(f"🔄 Cargando {period} de datos para {symbol}...")
    df = yf.download(symbol, interval=interval, period=period, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("No se pudieron descargar datos con yfinance.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# --- 2. Añadir indicadores ---
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print("✨ Añadiendo indicadores...")
    df = df.copy()
    df["ema_fast"] = ta.ema(df["close"], length=12)
    df["ema_slow"] = ta.ema(df["close"], length=26)
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    return df.dropna()


# --- 3. Crear target ---
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    future_price = df["close"].shift(-6)
    df["target"] = np.where(future_price > df["close"] * 1.02, 1,
                            np.where(future_price < df["close"] * 0.98, -1, 0))
    return df.dropna()


# --- 4. Entrenar modelo ---
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Entrena el modelo de clasificación solo con los datos de entrenamiento."""
    print("🚀 Entrenando modelo...")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# --- 5. Backtest long + short ---
def backtest(df: pd.DataFrame) -> pd.Series:
    """Ejecuta el backtest en un DataFrame que ya tiene las predicciones."""
    equity = 10000.0
    trades = []
    position = None

    for _, row in df.iterrows():
        signal_long = row.get("prob_up", 0.5) > THRESHOLD and row["close"] > row["ema_slow"]
        signal_short = row.get("prob_down", 0.5) > THRESHOLD and row["close"] < row["ema_slow"]

        if position is None:
            if signal_long or signal_short:
                direction = 1 if signal_long else -1
                price = row["open"]
                sl = price * (1 - SL_PCT * direction)
                tp = price * (1 + TP_PCT * direction)
                if abs(price - sl) == 0: continue
                qty = (equity * RISK_PERC) / abs(price - sl)
                entry = price * (1 + SLIP * direction)
                cost = entry * qty * COST
                equity -= cost
                position = {"dir": direction, "sl": sl, "tp": tp, "entry": entry,
                            "trail": entry, "qty": qty}
        elif position:
            high, low = row["high"], row["low"]
            exit_price = None
            direction = position["dir"]

            # Trailing stop
            if direction == 1:
                position["trail"] = max(position["trail"], high)
                trail_sl = position["trail"] * (1 - TRAIL_PCT)
                if low <= trail_sl: exit_price = trail_sl
            else:  # Short
                position["trail"] = min(position["trail"], low)
                trail_sl = position["trail"] * (1 + TRAIL_PCT)
                if high >= trail_sl: exit_price = trail_sl

            # TP o SL
            if direction == 1 and high >= position["tp"]:
                exit_price = position["tp"]
            elif direction == 1 and low <= position["sl"]:
                exit_price = position["sl"]
            elif direction == -1 and low <= position["tp"]:
                exit_price = position["tp"]
            elif direction == -1 and high >= position["sl"]:
                exit_price = position["sl"]

            if exit_price:
                pnl = (exit_price - position["entry"]) * position["qty"] * direction
                cost = exit_price * position["qty"] * COST
                equity += pnl - cost
                trades.append(pnl - cost)
                position = None

    return pd.Series(trades)


# --- 6. Ejecución Principal ---
def main():
    df = load_data()
    df = add_indicators(df)
    df = create_target(df)

    # --- DIVISIÓN DE DATOS CORRECTA PARA EVITAR DATA LEAKAGE ---
    split_idx = int(len(df) * 0.75)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:].copy()  # Usar .copy() para evitar warnings

    X_train, y_train = df_train[FEATURES], df_train['target']
    X_test = df_test[FEATURES]

    model = train_model(X_train, y_train)

    # Predecir probabilidades solo en el conjunto de prueba (out-of-sample)
    print("📊 Generando predicciones en datos fuera de muestra...")
    probs = model.predict_proba(X_test)

    # Asignar probabilidades de subida y bajada de forma segura
    class_up_idx = np.where(model.classes_ == 1)[0]
    class_down_idx = np.where(model.classes_ == -1)[0]

    if class_up_idx.size > 0:
        df_test["prob_up"] = probs[:, class_up_idx[0]]
    if class_down_idx.size > 0:
        df_test["prob_down"] = probs[:, class_down_idx[0]]

    # Ejecutar backtest solo en el conjunto de prueba
    print("📈 Ejecutando backtest...")
    pnl = backtest(df_test)

    # Métricas
    if not pnl.empty:
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        net = pnl.sum()
        pf = wins.sum() / abs(losses.sum()) if not losses.empty and wins.sum() > 0 else float('inf')
        wr = len(wins) / len(pnl) * 100 if not pnl.empty else 0.0
        print(f"\n--- Resultados del Backtest (Out-of-Sample) ---")
        print(f"Net Profit: {net:,.2f}, PF: {pf:.2f}, Trades: {len(pnl)}, Win%: {wr:.2f}")
    else:
        print("\nNo se realizaron trades en el período de prueba.")


if __name__ == "__main__":
    main()