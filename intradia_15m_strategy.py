"""Intradía 15m – Estrategia técnica simple (EMA+RSI), con SL/TP tight
- Descarga en tramos de 7 días para intradía (limitación Yahoo Finance)
- Fallback a CSV local (admite CSV sin cabecera: ts,open,high,low,close,volume)
- Exporta equity y metrics.json si se indica

Requiere: pandas, numpy, yfinance, pandas_ta
"""
import argparse
import json
import math
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    # Asegurarse de que se usa pandas_ta
    import pandas_ta as ta
except ImportError:
    ta = None

# --- Parámetros de la Estrategia ---
SL_PCT = 0.003
TP_PCT = 0.006
TRAIL_PCT = 0.002
RISK_PERC = 0.01
COST = 0.0002
SLIP = 0.0001
EMA_FAST = 12
EMA_SLOW = 48
RSI_LEN = 14
RSI_OB = 70
RSI_OS = 30


# --- Funciones de Datos ---
def daterange_chunks(start: datetime, end: datetime, days=7):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def download_in_chunks(symbol: str, start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance no disponible. Usa --csv-path para respaldo.")

    s = pd.to_datetime(start, utc=True)
    # <<< CORRECCIÓN: Se elimina el .tz_localize('UTC') redundante >>>
    e = pd.to_datetime(end, utc=True) if end else pd.Timestamp.utcnow()

    frames = []
    print(f"Descargando datos para {symbol} desde {s.date()} hasta {e.date()}...")
    for a, b in daterange_chunks(s, e, days=7):
        df_chunk = yf.download(symbol, interval='15m', start=a, end=b, auto_adjust=False, progress=False)
        if df_chunk is not None and not df_chunk.empty:
            frames.append(df_chunk)

    if not frames:
        raise RuntimeError("No se pudo descargar ningún dato intradía en los tramos solicitados.")

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep='last')]

    # Estandarizar nombres de columnas
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [c.lower() for c in out.columns]
    out.index = pd.to_datetime(out.index, utc=True).tz_localize(None)

    return out[['open', 'high', 'low', 'close', 'volume']]


def load_data(symbol: str, start: str, end: str = None, csv_path: str = None) -> pd.DataFrame:
    if csv_path and os.path.exists(csv_path):
        print(f"Cargando datos desde archivo local: {csv_path}")
        df = pd.read_csv(csv_path, header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()
        return df.astype(float)

    return download_in_chunks(symbol, start, end)


# --- Funciones de Estrategia ---
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print("✨ Añadiendo indicadores técnicos...")
    df = df.copy()
    if ta is None:
        raise RuntimeError("La librería 'pandas_ta' es necesaria. Por favor, instálala.")

    # <<< CORRECCIÓN: Usar las funciones correctas de pandas_ta >>>
    df['ema_fast'] = ta.ema(df['close'], length=EMA_FAST)
    df['ema_slow'] = ta.ema(df['close'], length=EMA_SLOW)
    df['rsi'] = ta.rsi(df['close'], length=RSI_LEN)

    return df.dropna()


def generate_signals(df: pd.DataFrame) -> pd.Series:
    long_sig = (df['ema_fast'] > df['ema_slow']) & (df['rsi'] < RSI_OS)
    short_sig = (df['ema_fast'] < df['ema_slow']) & (df['rsi'] > RSI_OB)
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[long_sig] = 1
    signal[short_sig] = -1
    return signal


def backtest(df: pd.DataFrame, signal: pd.Series, initial_capital=100_000.0):
    print("📈 Ejecutando backtest...")
    equity = initial_capital
    trades, position = [], None
    equity_curve = []

    for ts, row in df.iterrows():
        current_signal = signal.loc[ts]

        # --- Gestión de Posición Abierta ---
        if position:
            exit_price = None
            direction = position['dir']

            # Actualizar trailing stop
            if direction == 1:
                position['trail_stop'] = max(position['trail_stop'], row['high'] * (1 - TRAIL_PCT))
                if row['low'] <= position['sl'] or row['low'] <= position['trail_stop']:
                    exit_price = min(position['sl'], position['trail_stop'])
                elif row['high'] >= position['tp']:
                    exit_price = position['tp']
            else:  # Short
                position['trail_stop'] = min(position['trail_stop'], row['low'] * (1 + TRAIL_PCT))
                if row['high'] >= position['sl'] or row['high'] >= position['trail_stop']:
                    exit_price = max(position['sl'], position['trail_stop'])
                elif row['low'] <= position['tp']:
                    exit_price = position['tp']

            if exit_price:
                pnl = (exit_price - position['entry']) * direction * position['size']
                equity += pnl - (exit_price * position['size'] * COST)
                trades.append(pnl - (exit_price * position['size'] * COST))
                position = None

        # --- Abrir Nueva Posición ---
        if position is None and current_signal != 0:
            direction = current_signal
            entry_price = row['open'] * (1 + (SLIP * direction))
            stop_price = entry_price * (1 - (SL_PCT * direction))
            take_price = entry_price * (1 + (TP_PCT * direction))

            if abs(entry_price - stop_price) > 0:
                risk_amount = equity * RISK_PERC
                size = risk_amount / abs(entry_price - stop_price)
                equity -= entry_price * size * COST
                position = {
                    'dir': direction, 'entry': entry_price, 'sl': stop_price,
                    'tp': take_price, 'trail_stop': entry_price * (1 - (TRAIL_PCT * direction)),
                    'size': size
                }

        equity_curve.append(equity)

    # --- Cálculo de Métricas ---
    pnl_array = np.array(trades)
    if pnl_array.size == 0:
        return pd.DataFrame({'equity': [initial_capital]}), {}

    equity_df = pd.DataFrame({'equity': equity_curve}, index=df.index)
    wins = pnl_array[pnl_array > 0]
    losses = pnl_array[pnl_array < 0]

    net_profit = equity_curve[-1] - initial_capital
    peak = equity_df['equity'].expanding(min_periods=1).max()
    drawdown = (equity_df['equity'] - peak) / peak

    result = {
        'net_profit_usd': float(net_profit),
        'profit_factor': float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf,
        'win_rate': float(len(wins) / len(pnl_array) * 100),
        'trades': int(len(pnl_array)),
        'max_drawdown_pct': float(abs(drawdown.min()) * 100),
    }

    return equity_df, result


def save_outputs(eq: pd.DataFrame, res: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    eq.to_csv(os.path.join(out_dir, 'equity_curve.csv'))
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nArchivos guardados en: {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Backtest de estrategia simple de 15m")
    p.add_argument('--symbol', default='BTC-USD')
    # Argumento para calcular dinámicamente la fecha de inicio
    p.add_argument('--days-back', type=int, default=59, help="Días de datos a descargar (máx 60 para 15m)")
    p.add_argument('--csv-path', default=None, help='Ruta CSV local (respaldo)')
    p.add_argument('--initial-capital', type=float, default=100_000.0)
    p.add_argument('--save-json', action='store_true', help='Exportar equity_curve.csv y metrics.json')
    p.add_argument('--out-dir', default='./profiles/intradia_15m')
    args = p.parse_args()

    # Calcular fecha de inicio para cumplir con el límite de yfinance
    start_date = (datetime.utcnow() - timedelta(days=args.days_back)).strftime('%Y-%m-%dT%H:%M:%SZ')

    df = load_data(args.symbol, start=start_date, end=None, csv_path=args.csv_path)
    df_feat = add_features(df)
    signal = generate_signals(df_feat)

    eq, res = backtest(df_feat, signal, initial_capital=args.initial_capital)

    print("\n==== RESULTADOS INTRADÍA 15M ====")
    print(f"Periodo          : {df.index[0].date()} → {df.index[-1].date()}")
    for k, v in res.items():
        label = k.replace('_', ' ').title()
        print(f"{label:<18}: {v:,.2f}")

    if args.save_json:
        save_outputs(eq, res, args.out_dir)


if __name__ == '__main__':
    main()