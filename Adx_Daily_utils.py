# adx_daily_utils.py
# -*- coding: utf-8 -*-
"""
Utilidades para calcular ADX Diario y aplicarlo como "gate" sobre velas 4h.

- Por defecto NO usa yfinance. El origen diario por defecto es "resample":
  a partir de tus velas 4h se hace OHLC diario y se calcula ADX(14) daily.
- Opcionalmente puedes pedir "yfinance" o "ccxt" como fuente diaria (si lo usas).
- Devuelve df4h["ADX1D"] y una máscara booleana para filtrar señales.

CLI rápido de prueba:
    python adx_daily_utils.py --adx_daily_source resample --adx1d_len 14 --adx1d_min 20

Si no pasas datos externos, genera una serie 4h sintética para validar el gate.
"""
from __future__ import annotations
import argparse
import math
import sys
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except Exception as e:
    ta = None
    print("[WARN] pandas_ta no disponible. ADX usará una aproximación simple.")

# ---------------------------------------------------------------------------
# Helpers básicos
# ---------------------------------------------------------------------------

def _ensure_dt_index_utc(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Se espera un DataFrame indexado por fecha-hora (DatetimeIndex)")
    if df.index.tz is None:
        # asume UTC si viene naive
        df = df.tz_localize("UTC")
    else:
        df = df.tz_convert("UTC")
    return df


def _resample_4h_to_1d_ohlc(df4h: pd.DataFrame, daily_tz: str = "UTC") -> pd.DataFrame:
    df4h = _ensure_dt_index_utc(df4h)
    # OHLC diario a partir de 4h
    daily = pd.DataFrame({
        "open": df4h["open"].resample("1D").first(),
        "high": df4h["high"].resample("1D").max(),
        "low":  df4h["low" ].resample("1D").min(),
        "close":df4h["close"].resample("1D").last(),
        "volume": df4h.get("volume", pd.Series(index=df4h.index, dtype=float)).resample("1D").sum()
    })
    daily = daily.dropna(how="any")
    # Ajuste de tz si hace falta (para consistencia visual)
    if daily_tz.upper() != "UTC":
        try:
            daily = daily.tz_convert(daily_tz)
        except Exception:
            pass  # si falla, mantenemos UTC
    return daily


def _adx_from_daily_ohlc(daily: pd.DataFrame, length: int = 14) -> pd.Series:
    if len(daily) < max(20, length + 5):
        # muy pocos datos: devolvemos NaN
        return pd.Series(index=daily.index, dtype=float, name="ADX")

    if ta is not None:
        adx_df = ta.adx(daily["high"], daily["low"], daily["close"], length=length)
        # pandas_ta típicamente nombra la columna como ADX_{length}
        adx_col = f"ADX_{length}"
        if adx_df is None or adx_col not in adx_df.columns:
            return pd.Series(index=daily.index, dtype=float, name="ADX")
        adx = adx_df[adx_col].rename("ADX")
    else:
        # Fallback MUY simple si no hay pandas_ta (no recomendado para producción)
        tr = (daily["high"] - daily["low"]).abs()
        tr = tr.rolling(length, min_periods=1).mean()
        up = (daily["high"].diff()).clip(lower=0.0).rolling(length, min_periods=1).mean()
        dn = (-daily["low" ].diff()).clip(lower=0.0).rolling(length, min_periods=1).mean()
        with np.errstate(divide='ignore', invalid='ignore'):
            plus_di  = 100 * (up / tr).replace([np.inf, -np.inf], np.nan).fillna(0)
            minus_di = 100 * (dn / tr).replace([np.inf, -np.inf], np.nan).fillna(0)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(length, min_periods=1).mean().rename("ADX")
    return adx

# ---------------------------------------------------------------------------
# API pública: argumentos, cálculo de columna ADX1D y máscara booleana
# ---------------------------------------------------------------------------

def add_adx_daily_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--adx_daily_source",
        choices=["resample", "yfinance", "ccxt"],
        default="resample",
        help="Fuente para OHLC diario: resample (por defecto), yfinance o ccxt"
    )
    parser.add_argument("--adx1d_len", type=int, default=14, help="Longitud ADX diario (default: 14)")
    parser.add_argument("--adx1d_min", type=float, default=0.0, help="Umbral mínimo ADX1D para habilitar señales")
    parser.add_argument("--adx1d_col", type=str, default="ADX1D", help="Nombre de columna ADX 1D en df4h")
    parser.add_argument("--daily_tz", type=str, default="UTC", help="Zona horaria visual del diario (solo presentación)")


def add_adx1d_column(df4h: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Añade df4h[args.adx1d_col] con ADX 1D proyectado a 4h.
    Requiere columnas: open, high, low, close en df4h.
    """
    required = {"open", "high", "low", "close"}
    missing = required.difference(df4h.columns)
    if missing:
        raise ValueError(f"Faltan columnas OHLC en df4h: {sorted(missing)}")

    # Por ahora implementamos 'resample' (recomendado). Soportes yfinance/ccxt
    # se pueden añadir aquí si en tu entorno los usas para diario.
    if args.adx_daily_source == "resample":
        daily = _resample_4h_to_1d_ohlc(df4h, daily_tz=args.daily_tz)
    else:
        # Placeholders para futuras fuentes; por ahora, usa resample siempre
        daily = _resample_4h_to_1d_ohlc(df4h, daily_tz=args.daily_tz)

    adx = _adx_from_daily_ohlc(daily, length=args.adx1d_len)
    adx.name = args.adx1d_col

    # Proyecta a 4h con forward-fill por fecha
    df4h = _ensure_dt_index_utc(df4h)
    adx_4h = adx.tz_convert("UTC") if adx.index.tz is not None else adx.tz_localize("UTC")
    adx_4h = adx_4h.reindex(df4h.resample("1D").last().index).ffill()  # daily grid
    adx_4h = adx_4h.reindex(df4h.index, method="ffill")                 # pinta en 4h

    out = df4h.copy()
    out[args.adx1d_col] = adx_4h
    return out


def apply_adx1d_filter(df4h: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    col = args.adx1d_col
    if col not in df4h.columns:
        raise ValueError(f"No existe columna '{col}'. Llama primero a add_adx1d_column().")
    mask = (df4h[col] >= float(args.adx1d_min)).fillna(False)
    return mask


def summarize_filters(mask: pd.Series, name: str = "ADX1D gate") -> dict:
    total = int(mask.shape[0])
    passed = int(mask.sum())
    pct = 100.0 * passed / max(1, total)
    summary = {"name": name, "passed": passed, "total": total, "pct": pct}
    print(f"{name} → {passed}/{total} bars ({pct:.2f}%) pass")
    return summary

# ---------------------------------------------------------------------------
# CLI de prueba (si ejecutas directamente este archivo)
# ---------------------------------------------------------------------------

def _make_synthetic_4h(periods: int = 6*30, seed: int = 42) -> pd.DataFrame:
    """Serie 4h sintética con OHLC para validar rápidamente el gate."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=periods, freq="4h", tz="UTC")
    # random walk de cierres
    ret = rng.normal(0, 0.002, size=len(idx))
    close = 40000 * np.exp(np.cumsum(ret))
    # genera OHLC simples alrededor de close
    high = close * (1 + rng.uniform(0.0, 0.003, len(idx)))
    low  = close * (1 - rng.uniform(0.0, 0.003, len(idx)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol   = rng.uniform(10, 100, len(idx))
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": vol
    }, index=idx)
    return df


def main_cli(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Demo de gate ADX 1D sobre 4h")
    add_adx_daily_args(p)
    p.add_argument("--csv4h", type=str, default=None,
                   help="Ruta CSV con columnas: timestamp,open,high,low,close[,volume]")
    args = p.parse_args(argv)

    if args.csv4h:
        df4h = pd.read_csv(args.csv4h)
        # intenta parsear timestamp
        if "timestamp" in df4h.columns:
            df4h["timestamp"] = pd.to_datetime(df4h["timestamp"], utc=True)
            df4h = df4h.set_index("timestamp")
        df4h = df4h.sort_index()
        df4h = _ensure_dt_index_utc(df4h)
    else:
        df4h = _make_synthetic_4h(periods=180)  # ~30 días de 4h

    df4h = add_adx1d_column(df4h, args)
    mask = apply_adx1d_filter(df4h, args)
    summary = summarize_filters(mask, name="ADX1D gate")

    # Muestra una cabecera de control visual
    head = df4h[["close", args.adx1d_col]].head(5)
    print(f"Source: {args.adx_daily_source} len: {args.adx1d_len}")
    print(head.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())