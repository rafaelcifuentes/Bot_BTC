# scripts/_timekeys.py
import pandas as pd

_TS_CANDS = ("timestamp", "ts", "time", "datetime", "date", "dt")

def find_ts(df, preferred=None):
    if preferred and preferred in df.columns:
        return preferred
    for c in df.columns:
        if c.lower() in _TS_CANDS:
            return c
    raise KeyError("No timestamp-like column found")

def norm_ts(df, col=None, freq=None):
    """Parsea a UTC, quita tz (naive) y opcionalmente hace floor a freq."""
    col = find_ts(df, col)
    s = pd.to_datetime(df[col], utc=True, errors="coerce")
    s = s.dt.tz_convert(None)
    if freq:
        s = s.dt.floor(freq)
    df[col] = s
    return col

def safe_asof(left, right, left_on=None, right_on=None, freq=None, **kwargs):
    """merge_asof con normalización de timestamps en ambos lados."""
    left_on = norm_ts(left, left_on, freq=freq)
    right_on = norm_ts(right, right_on, freq=freq)
    return pd.merge_asof(
        left.sort_values(left_on),
        right.sort_values(right_on),
        left_on=left_on,
        right_on=right_on,
        **kwargs,
    )