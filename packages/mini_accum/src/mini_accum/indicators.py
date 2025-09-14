import pandas as pd

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()
# --- KISS exit-guard: Wilder ATR14 -----------------------------------------

def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR de Wilder con EWM(alpha=1/period). Alineado con close.index. Anti-lookahead:
    el consumidor debe usar shift(1) si requiere usar ATR 'ya conocido' al cierre de la barra previa."""
    hi = high.astype(float)
    lo = low.astype(float)
    cl = close.astype(float)
    prev_close = cl.shift(1)
    tr = pd.concat(
        [(hi - lo).abs(),
         (hi - prev_close).abs(),
         (lo - prev_close).abs()],
        axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr
