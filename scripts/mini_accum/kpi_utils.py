# scripts/mini_accum/kpi_utils.py
from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd

__all__ = [
    "normalize_ohlc_columns",
    "to_ts",
    "compute_fpy",
    "sanity_kpis",
    "safe_read_csv",
]

def safe_read_csv(path: str, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows)
    return df

def normalize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    aliases = {
        'timestamp': ['time','ts','date','datetime'],
        'open':      ['o','open_price'],
        'high':      ['h','max','high_price'],
        'low':       ['l','min','low_price'],
        'close':     ['c','close_price','price'],
        'volume':    ['v','vol']
    }
    for target, alts in aliases.items():
        if target not in df.columns:
            for a in alts:
                if a in df.columns:
                    df.rename(columns={a: target}, inplace=True)
                    break

    if 'high' not in df.columns and {'open','close'}.issubset(df.columns):
        df['high'] = df[['open','close']].max(axis=1)
    if 'low' not in df.columns and {'open','close'}.issubset(df.columns):
        df['low'] = df[['open','close']].min(axis=1)

    return df

def to_ts(x) -> datetime:
    if hasattr(x, 'tzinfo'):
        dt = x.to_pydatetime() if hasattr(x, 'to_pydatetime') else x
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    s = str(x)
    fmts = [
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            if "%z" not in fmt:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            continue
    return pd.to_datetime(s, utc=True).to_pydatetime()

def compute_fpy(flips_df: pd.DataFrame, start_ts=None, end_ts=None) -> float:
    if flips_df is None or len(flips_df) == 0:
        return 0.0

    time_col = None
    for col in ['ts','timestamp','time','executed','date','datetime']:
        if col in flips_df.columns:
            time_col = col
            break

    if start_ts is None or end_ts is None:
        if time_col is None:
            return 0.0
        ts_sorted = sorted(to_ts(t) for t in flips_df[time_col])
        start, end = ts_sorted[0], ts_sorted[-1]
    else:
        start, end = to_ts(start_ts), to_ts(end_ts)

    days = (end - start).total_seconds() / 86400.0
    if days <= 0:
        return 0.0
    flips_total = len(flips_df)
    return flips_total * 365.25 / days

def sanity_kpis(fpy: float, flips_total: int) -> None:
    if flips_total > 0 and (fpy == 0 or fpy is None):
        raise ValueError("FPY=0 con flips>0 → revisar annualización o período seleccionado.")
