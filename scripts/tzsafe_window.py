# -*- coding: utf-8 -*-
"""
Utilidades tz-safe para ventanas de backtest:
- Normaliza indices y límites a una misma zona (UTC por defecto).
- Corta por ventana sin errores tz-naive/aware.
- Heurística para elegir la columna de señal.
- Estadísticas rápidas y conteo de crosses / bars>=threshold con histéresis.

Uso típico en tus scripts:
    from tzsafe_window import (
        parse_windows_arg, ensure_index_tzaware, slice_df_by_window_tzsafe,
        pick_signal_column, quick_quantiles_and_counts, count_crosses
    )
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ===== Config por defecto (puedes cambiar por env vars) =====
TZ_DEFAULT = os.environ.get("TZSAFE_TZ", "UTC")
HYS_PP = float(os.environ.get("HYSTERESIS_PP", "0.04"))  # p.ej. 0.04 = 4 pp en escala [0,1]
TREND_FILTER = bool(int(os.environ.get("TREND_FILTER", "0")))
TREND_MODE = os.environ.get("TREND_MODE", "strict")
REARM_MIN = int(os.environ.get("REARM_MIN", "2"))

# ===== Dataclass para ventanas =====
@dataclass
class Window:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp

# ===== Helpers tz-aware / parsing =====
def _to_tzaware(ts_like, tz: str = TZ_DEFAULT) -> pd.Timestamp:
    """Convierte cualquier str/Timestamp a Timestamp tz-aware en tz (sin convertir si ya está en tz)."""
    ts = pd.Timestamp(ts_like)
    if ts.tz is None:
        # naïve -> localizar en tz
        return ts.tz_localize(tz)
    # aware -> convertir a tz destino si hace falta
    return ts.tz_convert(tz)

def ensure_index_tzaware(df: pd.DataFrame, tz: str = TZ_DEFAULT, copy: bool = True) -> pd.DataFrame:
    """Asegura que el DatetimeIndex del DF sea tz-aware y en 'tz'. Devuelve un nuevo DF si copy=True."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("El DataFrame debe tener un DatetimeIndex para usar ensure_index_tzaware.")
    out = df.copy() if copy else df
    idx = out.index
    if idx.tz is None:
        # naïve -> localizar
        out.index = idx.tz_localize(tz)
    else:
        # aware -> convertir
        out.index = idx.tz_convert(tz)
    return out

def slice_df_by_window_tzsafe(df: pd.DataFrame,
                              start,
                              end,
                              tz: str = TZ_DEFAULT,
                              closed: str = "both") -> pd.DataFrame:
    """
    Corta df entre [start, end] de forma tz-safe.
    Acepta start/end como str o Timestamp (naïve o aware).
    """
    df2 = ensure_index_tzaware(df, tz=tz, copy=True)
    s = _to_tzaware(start, tz)
    e = _to_tzaware(end, tz)
    # pandas respeta tz al hacer loc; closed controla inclusión de extremos
    if closed == "both":
        return df2.loc[s:e]
    elif closed == "left":
        return df2.loc[:e].loc[df2.index >= s]
    elif closed == "right":
        return df2.loc[s:].loc[df2.index <= e]
    elif closed == "neither":
        return df2.loc[(df2.index > s) & (df2.index < e)]
    else:
        raise ValueError(f"closed inválido: {closed}")

WINDOW_RE = re.compile(r"^(?:(?P<label>[^:]+):)?(?P<start>\d{4}-\d{2}-\d{2})(?::(?P<end>\d{4}-\d{2}-\d{2}))$")

def parse_window_spec(spec: str, tz: str = TZ_DEFAULT) -> Window:
    """
    Acepta formatos:
      - LABEL:YYYY-MM-DD:YYYY-MM-DD
      - YYYY-MM-DD:YYYY-MM-DD  (label=YYYYQx/Hx si quieres lo generas aparte)
    """
    m = WINDOW_RE.match(spec.strip())
    if not m:
        raise ValueError(f"Spec de ventana inválida: {spec}")
    label = m.group("label") or spec.split(":")[0]
    start = _to_tzaware(m.group("start"), tz)
    # end al final del día: pandas corta inclusivo con .loc; hacemos 23:59:59.999999999
    end_raw = pd.Timestamp(m.group("end"))
    end = pd.Timestamp(end_raw.date()) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    end = _to_tzaware(end, tz)
    return Window(label=label, start=start, end=end)

def parse_windows_arg(windows: Sequence[str], tz: str = TZ_DEFAULT) -> List[Window]:
    return [parse_window_spec(w, tz) for w in windows]

# ===== Señales / KPIs rápidos =====
_NUMERIC_EXCLUDE = {"open", "high", "low", "close", "volume", "oi", "openinterest"}

def pick_signal_column(df: pd.DataFrame,
                       preferred: Optional[str] = None) -> str:
    """
    Elige la columna de señal (score/prob/signal) de forma robusta.
    - Si preferred existe, la usa.
    - Si hay ['score','prob','signal','s','p'] en df, toma la primera.
    - Si no, toma la primera numérica que no sea OHLC/volume.
    """
    if preferred and preferred in df.columns:
        return preferred
    candidates = [c for c in ["score", "prob", "signal", "s", "p", "value"] if c in df.columns]
    if candidates:
        return candidates[0]
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in _NUMERIC_EXCLUDE]
    if not numeric:
        raise ValueError("No se encontró columna de señal numérica en el DataFrame.")
    return numeric[0]

def quick_quantiles_and_counts(s: pd.Series,
                               thresholds: Iterable[float] = (0.55, 0.58, 0.60, 0.62, 0.66, 0.70)) -> Tuple[dict, dict]:
    """Devuelve (quantiles, ge_counts) para logging."""
    q = s.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).round(3).to_dict()
    ge = {float(th): int((s >= th).sum()) for th in thresholds}
    return q, ge

# ===== Crosses y bars>=threshold con histéresis =====
def count_crosses(s: pd.Series,
                  threshold: float,
                  hysteresis_pp: Optional[float] = None) -> Tuple[int, int]:
    """
    Cuenta cuántas veces la señal cruza el umbral (modo on/off) y cuántas barras >= umbral.
    Histéresis: ON si s >= th; OFF sólo si s <= th - hys_pp (para evitar dither).
    Devuelve (crosses, bars_ge_th).
    """
    hys = HYS_PP if hysteresis_pp is None else float(hysteresis_pp)
    th_on = threshold
    th_off = max(threshold - hys, 0.0)

    on = False
    crosses = 0
    ge = 0

    vals = s.astype(float).values
    for v in vals:
        if v >= th_on:
            ge += 1
            if not on:
                on = True
                crosses += 1
        elif on and v <= th_off:
            on = False
    return crosses, ge

# ===== Logging estilo [tzsafe] para depurar =====
def log_pre(path: str, win: Window, rows: int, s: pd.Series, tf: bool, mode: str, rm: int, hys: float):
    q, ge = quick_quantiles_and_counts(s)
    print(f"[tzsafe:pre] start={win.start} end={win.end} rows={rows} q={{{0.0: {q[0.0]}, 0.25: {q[0.25]}, 0.5: {q[0.5]}, 0.75: {q[0.75]}, 1.0: {q[1.0]}}}} ge={ge} TF={tf} MODE={mode} RM={rm} HYS={hys}")

def log_stats(win: Window, th: float, crosses: int, bars_ge: int, horizon: Optional[int] = None):
    hor = "None" if horizon is None else int(horizon)
    print(f"[tzsafe] start={win.start} end={win.end} th={th:.2f} hor={hor} crosses={crosses} bars>=th={bars_ge}")

# ===== Carga de señales (tus CSVs en reports/windows_fixed/<label>/<asset>.csv) =====
def load_signals(signals_root: str, window_label: str, asset: str) -> pd.DataFrame:
    """
    Carga CSV con índice temporal (columna 0 o 'timestamp') y normaliza a tz=TZ_DEFAULT.
    """
    path = os.path.join(signals_root, window_label, f"{asset}.csv")
    if not os.path.exists(path):
        # fallback por compatibilidad: algunos dumps están en .../<label>-<asset>.csv
        alt = os.path.join(signals_root, f"{window_label}-{asset}.csv")
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"No existe {path} ni {alt}")

    df = pd.read_csv(path)
    # Índice temporal robusto
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
        df = df.drop(columns=["timestamp"])
    else:
        # asumir primera columna es tiempo
        ts = pd.to_datetime(df.iloc[:, 0], utc=False, errors="coerce")
        df = df.iloc[:, 1:]

    if ts.isna().any():
        raise ValueError(f"Timestamps inválidos/NaN al cargar {path}")

    # Si vienen naïve -> localizamos; si vienen aware -> convertimos a TZ_DEFAULT
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(TZ_DEFAULT)
    else:
        ts = ts.dt.tz_convert(TZ_DEFAULT)

    df.index = ts
    df = ensure_index_tzaware(df, tz=TZ_DEFAULT, copy=False)
    return df

# ===== Mini-runner opcional para debug rápido =====
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Debug tz-safe windows and thresholds.")
    p.add_argument("--windows", nargs="+", required=True, help='Ej: 2023Q4:2023-10-01:2023-12-31  2024H1:2024-01-01:2024-06-30')
    p.add_argument("--asset", required=True, help="BTC-USD, etc.")
    p.add_argument("--thresholds", nargs="+", type=float, required=True, help="Lista de umbrales, ej: 0.62 0.64 0.66 ...")
    p.add_argument("--horizon", type=int, default=None, help="Sólo para logging.")
    p.add_argument("--signals_root", default="reports/windows_fixed")
    p.add_argument("--signal_col", default=None, help="Nombre explícito de la columna de señal (opcional).")
    args = p.parse_args()

    wins = parse_windows_arg(args.windows, tz=TZ_DEFAULT)
    for w in wins:
        df = load_signals(args.signals_root, w.label, args.asset)
        col = pick_signal_column(df, preferred=args.signal_col)
        win_df = slice_df_by_window_tzsafe(df, w.start, w.end, tz=TZ_DEFAULT, closed="both")
        s = win_df[col].astype(float)
        log_pre(args.signals_root, w, len(win_df), s, TREND_FILTER, TREND_MODE, REARM_MIN, HYS_PP)
        for th in args.thresholds:
            cr, ge = count_crosses(s, th, hysteresis_pp=HYS_PP)
            log_stats(w, th, cr, ge, horizon=args.horizon)