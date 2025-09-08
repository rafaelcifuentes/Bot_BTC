from __future__ import annotations
import pandas as pd
from .indicators import ema

def load_ohlc(csv_path: str, ts_col: str, tz_input: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if ts_col not in df.columns:
        ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
    if df[ts_col].dt.tz is None:
        if tz_input and tz_input.upper() != 'UTC':
            df[ts_col] = df[ts_col].dt.tz_localize(tz_input).dt.tz_convert('UTC')
        else:
            df[ts_col] = df[ts_col].dt.tz_localize('UTC')
    df = df.sort_values(ts_col).dropna(subset=[ts_col]).reset_index(drop=True)
    df = df.rename(columns={ts_col: 'ts'})
    required = {'open', 'high', 'low', 'close'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {csv_path}: {sorted(missing)}")
    return df


def merge_daily_into_4h(df4: pd.DataFrame, df1d: pd.DataFrame) -> pd.DataFrame:
    d = df1d[['ts', 'close']].copy()
    d['d_ema200'] = ema(d['close'], 200)
    d = d.rename(columns={'close': 'd_close'})
    merged = pd.merge_asof(
        df4.sort_values('ts'),
        d.sort_values('ts'),
        left_on='ts', right_on='ts',
        direction='backward'
    )
    merged['macro_green'] = merged['d_close'] > merged['d_ema200']
    return merged
