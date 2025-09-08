# Manus_v1.2.py
# CORRECCIÓN: Implementado un sistema de renombrado de columnas más robusto en add_indicators
# para encontrar y estandarizar los nombres de los indicadores de forma fiable, solucionando el KeyError.

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import pytz
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Configuración global
# -----------------------------
TIMEZONE_LOCAL = "America/Toronto"
TZ = pytz.timezone(TIMEZONE_LOCAL)

DEFAULT_PARAMS = {
    "THRESHOLD": 0.55,
    "SL_PCT": 0.025,
    "TP_PCT": 0.05,
    "TRAIL_PCT": 0.01,
    "RISK_PERC": 0.01,
    "COST": 0.0002,
    "SLIPPAGE": 0.0001,
}

MODEL_PATH = Path("rf_swing4h.pkl")

# --- Lista de Features Unificada ---
# Nombres estandarizados que usaremos en todo el script
FEATURE_NAMES = ["atr", "adx", "rsi", "ema_50", "ema_200"]


# -----------------------------
# Utilidades de datos
# -----------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    return df


def load_data(symbol: str, csv_path: str | Path | None = None) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True, header=0)
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df = yf.download(symbol, interval="4h", period="720d", progress=False, auto_adjust=False)
        if df.empty:
            raise RuntimeError("yfinance devolvió un DataFrame vacío")
        df.index = pd.to_datetime(df.index, utc=True)
        df = _flatten_columns(df)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Columnas faltantes: {missing}")
    return df[required]


# -----------------------------
# Indicadores
# -----------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Calcular indicadores con pandas_ta
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)

    # --- SOLUCIÓN AL KeyError ---
    # Renombrado robusto: buscar la columna que contiene el nombre del indicador
    rename_map = {
        "atr": "atr",
        "adx": "adx",
        "rsi": "rsi",
        "ema_50": "ema_50",
        "ema_200": "ema_200"
    }

    # Crear un diccionario para el renombrado final
    final_rename_map = {}
    current_cols = df.columns.str.lower()

    # Buscar cada indicador en las columnas actuales y mapearlo a su nombre estándar
    for key in rename_map:
        # Caso especial para EMA, ya que hay dos
        if "ema" in key:
            length = key.split('_')[1]
            # Buscar una columna como 'ema_50'
            found_col = next((col for col in current_cols if f"ema_{length}" in col), None)
        else:
            # Buscar una columna como 'atr_14' o 'adx_14'
            found_col = next((col for col in current_cols if key in col), None)

        if found_col:
            # Mapear el nombre original encontrado al nombre estándar
            original_col_name = df.columns[current_cols.tolist().index(found_col)]
            final_rename_map[original_col_name] = rename_map[key]

    df.rename(columns=final_rename_map, inplace=True)

    # Eliminar filas con NaNs DESPUÉS de haber calculado y renombrado todo
    df.dropna(inplace=True)
    return df


# -----------------------------
# Modelado & señales
# -----------------------------

def load_or_train_model(df: pd.DataFrame, force_retrain: bool = False) -> RandomForestClassifier:
    if MODEL_PATH.exists() and not force_retrain:
        return joblib.load(MODEL_PATH)

    df = df.copy()
    df["target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)
    df.dropna(inplace=True)

    X = df[FEATURE_NAMES]
    y = df["target"]

    model = RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


def generate_signals(df: pd.DataFrame, model: RandomForestClassifier, threshold: float) -> pd.DataFrame:
    prob_long = model.predict_proba(df[FEATURE_NAMES])[:, 1]

    df["signal_long"] = ((prob_long >= threshold) & (df["ema_50"] > df["ema_200"])).astype(int)
    df["signal_short"] = ((prob_long <= 1 - threshold) & (df["ema_50"] < df["ema_200"])).astype(int)
    return df


# -----------------------------
# Backtest (sin cambios, ya era correcto)
# -----------------------------
def backtest_realistic(df: pd.DataFrame, *, sl_pct, tp_pct, trail_pct, risk_perc, cost, slippage):
    init_equity = 100_000.0
    equity = init_equity
    peak = equity
    equity_curve: list[float] = []
    drawdown_curve: list[float] = []
    pnl_list: list[float] = []
    open_pos = None
    for _, row in df.iterrows():
        if open_pos:
            d = open_pos["direction"]
            if d == 1:
                open_pos["high_water_mark"] = max(open_pos["high_water_mark"], row["high"])
                open_pos["trail_stop"] = max(open_pos["trail_stop"], open_pos["high_water_mark"] * (1 - trail_pct))
            else:
                open_pos["high_water_mark"] = min(open_pos["high_water_mark"], row["low"])
                open_pos["trail_stop"] = min(open_pos["trail_stop"], open_pos["high_water_mark"] * (1 + trail_pct))
            exit_price = None
            if d == 1:
                if row["low"] <= open_pos["stop"]:
                    exit_price = open_pos["stop"]
                elif row["low"] <= open_pos["trail_stop"]:
                    exit_price = open_pos["trail_stop"]
                elif row["high"] >= open_pos["target"]:
                    exit_price = open_pos["target"]
            else:
                if row["high"] >= open_pos["stop"]:
                    exit_price = open_pos["stop"]
                elif row["high"] >= open_pos["trail_stop"]:
                    exit_price = open_pos["trail_stop"]
                elif row["low"] <= open_pos["target"]:
                    exit_price = open_pos["target"]
            if exit_price:
                exit_price_slippage = exit_price * (1 - slippage * d)
                pnl = (exit_price_slippage - open_pos["entry_price"]) * d * open_pos["units"] - (
                            open_pos["entry_price"] + exit_price_slippage) * cost * open_pos["units"]
                pnl_list.append(pnl)
                equity += pnl
                open_pos = None
        elif open_pos is None:
            direction = 1 if row["signal_long"] else -1 if row["signal_short"] else 0
            if direction != 0:
                entry_price = row["close"] * (1 + slippage * direction)
                risk_amount = equity * risk_perc
                units = risk_amount / (sl_pct * entry_price)
                open_pos = {
                    "direction": direction, "entry_price": entry_price, "units": units,
                    "stop": entry_price * (1 - sl_pct * direction),
                    "target": entry_price * (1 + tp_pct * direction),
                    "trail_stop": entry_price * (1 - trail_pct * direction),
                    "high_water_mark": entry_price,
                }
        peak = max(peak, equity)
        equity_curve.append(equity)
        drawdown_curve.append((equity - peak) / peak if peak > 0 else 0)
    trades = len(pnl_list)
    wins = sum(1 for pnl in pnl_list if pnl > 0)
    gross_profit = sum(p for p in pnl_list if p > 0)
    gross_loss = -sum(p for p in pnl_list if p < 0)
    profit_factor = gross_profit / gross_loss if gross_loss else np.inf
    return {
        "equity_curve": np.array(equity_curve), "drawdown_curve": np.array(drawdown_curve),
        "net_profit": equity - init_equity, "win_rate": wins / trades * 100 if trades else 0,
        "profit_factor": profit_factor, "max_drawdown": min(drawdown_curve) if drawdown_curve else 0,
        "trades": trades,
    }


# -----------------------------
# Gráficos y CLI (sin cambios)
# -----------------------------
def plot_equity_drawdown(equity: np.ndarray, drawdown: np.ndarray, df_index):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(df_index, equity, label="Equity ($)", color="dodgerblue")
    ax1.set_title("Curva de Equity y Drawdown")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    ax2.plot(df_index, drawdown * 100, color="red")
    ax2.fill_between(df_index, drawdown * 100, 0, alpha=0.3, color="red")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(min(drawdown) * 100 * 1.1 - 1, 1)
    ax2.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel("Fecha")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Backtest Swing 4h ML RF long/short")
    parser.add_argument("--symbol", default="BTC-USD", help="Símbolo yfinance")
    parser.add_argument("--csv", help="Ruta CSV local opcional")
    parser.add_argument("--retrain", action="store_true", help="Fuerza reentrenar el modelo")
    args = parser.parse_args()

    print("1. Cargando datos...")
    df = load_data(args.symbol, args.csv)
    print("2. Añadiendo indicadores...")
    df = add_indicators(df)
    print("3. Cargando o entrenando modelo...")
    model = load_or_train_model(df, force_retrain=args.retrain)
    print("4. Generando señales...")
    df = generate_signals(df, model, DEFAULT_PARAMS["THRESHOLD"])
    print("5. Ejecutando backtest realista...")
    results = backtest_realistic(df, **DEFAULT_PARAMS)

    print("\n=== RESULTADOS REALISTAS DEL BACKTEST ===")
    for k, v in results.items():
        if k.endswith("curve"): continue
        label = k.replace("_", " ").title()
        if isinstance(v, float):
            print(f"{label:<15}: {v:,.2f}")
        else:
            print(f"{label:<15}: {v}")
    plot_equity_drawdown(results["equity_curve"], results["drawdown_curve"], df.index)


if __name__ == "__main__":
    main()
