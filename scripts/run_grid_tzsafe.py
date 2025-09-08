#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_grid_tzsafe.py
Pre-check rápido de señales por ventanas/horizontes/umbrales:
- Cuenta cruces y barras>=umbral por combinación
- Integra iterador robusto de ventanas (dict / lista / Window)
- Normaliza todo a UTC tz-aware para evitar tz-naive vs tz-aware
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Iterator, Tuple, List, Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Imports del módulo del proyecto (con fallback al mismo dir "scripts/")
# ---------------------------------------------------------------------
try:
    from tzsafe_window import (  # type: ignore
        parse_windows_arg,
        ensure_index_tzaware as _ensure_index_tzaware_upstream,
        slice_df_by_window_tzsafe,
        load_signals as _load_signals_upstream,
        pick_signal_column as _pick_signal_column_upstream,
        count_crosses as _count_crosses_upstream,
        quick_quantiles_and_counts as _qqc_upstream,
    )
except Exception:
    HERE = os.path.dirname(__file__)
    ROOT = os.path.abspath(os.path.join(HERE, ".."))
    for p in (HERE, ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    from tzsafe_window import (  # type: ignore
        parse_windows_arg,
        ensure_index_tzaware as _ensure_index_tzaware_upstream,
        slice_df_by_window_tzsafe,
        load_signals as _load_signals_upstream,
        pick_signal_column as _pick_signal_column_upstream,
        count_crosses as _count_crosses_upstream,
        quick_quantiles_and_counts as _qqc_upstream,
    )

# ------------------------------
# Utiles tz / ventanas robustos
# ------------------------------

def _to_utc_ts(x: Any) -> pd.Timestamp:
    """
    Convierte a pd.Timestamp tz-aware en UTC.
    - naive -> tz_localize('UTC')
    - aware -> tz_convert('UTC')
    """
    ts = x if isinstance(x, pd.Timestamp) else pd.Timestamp(x)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")  # naive -> aware (UTC)
    else:
        ts = ts.tz_convert("UTC")   # aware -> UTC
    return ts


def iter_windows_robust(windows: Any) -> Iterator[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """
    Normaliza múltiples formatos de 'windows' a tuplas (label, start_utc, end_utc).
    Soporta:
      - dict: {label: (start, end)}
      - list/tuple de: (label, start, end)
      - list/tuple de objetos con .label/.start/.end (p.ej. dataclass Window)
      - un solo objeto con .label/.start/.end
    """
    if isinstance(windows, dict):
        for label, se in windows.items():
            if not isinstance(se, (tuple, list)) or len(se) != 2:
                raise ValueError(f"Valor inválido para '{label}': {se!r} (esperado (start, end))")
            start, end = se
            yield str(label), _to_utc_ts(start), _to_utc_ts(end)
        return

    if isinstance(windows, (list, tuple)):
        for item in windows:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                label, start, end = item
                yield str(label), _to_utc_ts(start), _to_utc_ts(end)
                continue

            if all(hasattr(item, a) for a in ("label", "start", "end")):
                yield str(getattr(item, "label")), _to_utc_ts(getattr(item, "start")), _to_utc_ts(getattr(item, "end"))
                continue

            if isinstance(item, dict) and {"label", "start", "end"} <= set(item.keys()):
                yield str(item["label"]), _to_utc_ts(item["start"]), _to_utc_ts(item["end"])
                continue

            raise ValueError(f"Item inesperado en lista: {type(item)} - {item!r}")
        return

    if all(hasattr(windows, a) for a in ("label", "start", "end")):
        yield str(getattr(windows, "label")), _to_utc_ts(getattr(windows, "start")), _to_utc_ts(getattr(windows, "end"))
        return

    raise TypeError(f"Estructura de 'windows' no soportada: {type(windows)} - {windows!r}")


# ----------------------------
# Wrappers de compatibilidad
# ----------------------------

def ensure_index_tzaware(df: pd.DataFrame) -> pd.DataFrame:
    """Simple wrapper al upstream (por si cambia de nombre/ubicación)."""
    return _ensure_index_tzaware_upstream(df)


def count_crosses_and_bars(
    series: pd.Series,
    threshold: float,
    rearm_min: int | None = None,
    hysteresis_pp: float | None = None,
) -> Tuple[int, int]:
    """
    Llama al upstream count_crosses con compatibilidad:
      - Si el upstream acepta (series, th, rearm_min, hysteresis), lo usa
      - Si sólo acepta (series, th), lo usa
      - Si el upstream devuelve (cruces, bars>=th) lo respeta
      - Si devuelve solo cruces, computa bars>=th localmente
    """
    try:
        res = _count_crosses_upstream(series, threshold, rearm_min, hysteresis_pp)
    except TypeError:
        res = _count_crosses_upstream(series, threshold)

    if isinstance(res, tuple) and len(res) >= 2:
        crosses, bars_ge = int(res[0]), int(res[1])
    else:
        crosses = int(res)
        bars_ge = int((series >= float(threshold)).sum())
    return crosses, bars_ge


# ----------------------------
# Carga robusta de señales
# ----------------------------

def _maybe_make_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Si el índice no es DatetimeIndex, intenta promover columnas comunes a índice temporal UTC."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    # 1) columnas habituales
    for col in ("timestamp", "time", "datetime", "date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            df = df.set_index(col)
            break
    else:
        # 2) primer columna
        if len(df.columns) > 0:
            first = df.columns[0]
            try:
                df[first] = pd.to_datetime(df[first], utc=True, errors="coerce")
                df = df.set_index(first)
            except Exception:
                # 3) último recurso: intentar parsear el índice actual
                df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    return df


def load_signals_from_root(signals_root: str, window_label: str, asset: str) -> Tuple[pd.Series, str]:
    """
    Intenta usar el loader upstream; si falla, lee CSV:
      {signals_root}/{window_label}/{asset}.csv
    Devuelve la Serie de señal numérica principal y el path usado (para logging).
    """
    # 1) loader oficial
    try:
        df = _load_signals_upstream(signals_root, window_label, asset)
        src = f"(upstream loader) {signals_root}/{window_label}/{asset}.csv"
    except Exception:
        # 2) CSV directo
        csv_path = os.path.join(signals_root, window_label, f"{asset}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No existe el archivo de señales: {csv_path}")
        df = pd.read_csv(csv_path)
        src = csv_path

    # índice temporal robusto + tz-aware
    df = _maybe_make_dt_index(df)
    df = ensure_index_tzaware(df)

    # elegir columna de señal
    try:
        col = _pick_signal_column_upstream(df)
    except Exception:
        # Fallback: primera numérica
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("No encontré columna numérica de señal en el DataFrame.")
        col = num_cols[0]

    sig = df[col].astype(float)
    return sig, src


# ----------------------------
# CLI principal
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-check tzsafe (cruces y barras>=umbral) por ventana/horizonte/umbral")
    ap.add_argument("--windows", nargs="+", required=True,
                    help='Ej.: "2023Q4:2023-10-01:2023-12-31"  "2024H1:2024-01-01:2024-06-30"')
    ap.add_argument("--assets", nargs="+", required=True)
    ap.add_argument("--horizons", nargs="+", required=True, type=int)
    ap.add_argument("--thresholds", nargs="+", required=True, type=float)

    ap.add_argument("--signals_root", required=True)
    ap.add_argument("--ohlc_root", required=False, default="")  # no se usa aquí, pero lo dejamos para compat

    ap.add_argument("--fee_bps", type=int, default=6)
    ap.add_argument("--slip_bps", type=int, default=6)
    ap.add_argument("--partial", choices=["none", "50_50"], default="none")
    ap.add_argument("--breakeven_after_tp1", action="store_true")
    ap.add_argument("--risk_total_pct", type=float, default=0.75)
    ap.add_argument("--weights", nargs="+", default=[])

    # gates (se aceptan pero NO se aplican aquí; esto es pre-check)
    ap.add_argument("--gate_pf", type=float, default=1.6)
    ap.add_argument("--gate_wr", type=float, default=0.60)
    ap.add_argument("--gate_trades", type=int, default=30)

    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_top", required=True)

    # ENV opcional para rearm/hysteresis/TF-mode
    rearm_env = os.getenv("REARM_MIN")
    hyst_env = os.getenv("HYSTERESIS_PP")
    trend_filter_env = os.getenv("TREND_FILTER")
    trend_mode_env = os.getenv("TREND_MODE", "soft")

    args = ap.parse_args()

    # Parseo y normalización robusta de ventanas
    windows_parsed = parse_windows_arg(args.windows)
    windows_iterable = list(iter_windows_robust(windows_parsed))

    rows: List[Dict[str, Any]] = []

    for (wlabel, wstart, wend) in windows_iterable:
        # Carga señales por asset
        for asset in args.assets:
            sig, src = load_signals_from_root(args.signals_root, wlabel, asset)

            # Recorte por ventana (usa util upstream si está)
            try:
                sig_w = slice_df_by_window_tzsafe(sig.to_frame("sig"), wstart, wend)["sig"]
            except Exception:
                sig_w = sig[(sig.index >= wstart) & (sig.index <= wend)]

            # Debug/quantiles opcional
            try:
                q, ge = _qqc_upstream(sig_w.values, [0.55, 0.58, 0.60, 0.62, 0.66, 0.70])
                print(f"[tzsafe:pre] start={wstart} end={wend} rows={len(sig_w)} q={q} ge={ge} "
                      f"TF={'True' if trend_filter_env=='1' else 'False'} MODE={trend_mode_env} "
                      f"RM={rearm_env if rearm_env else '?'} HYS={hyst_env if hyst_env else '?'}")
            except Exception:
                print(f"[tzsafe:pre] start={wstart} end={wend} rows={len(sig_w)} "
                      f"TF={'True' if trend_filter_env=='1' else 'False'} MODE={trend_mode_env} "
                      f"RM={rearm_env if rearm_env else '?'} HYS={hyst_env if hyst_env else '?'}")

            for hor in args.horizons:
                for th in args.thresholds:
                    crosses, bars_ge = count_crosses_and_bars(
                        sig_w, float(th),
                        int(rearm_env) if rearm_env else None,
                        float(hyst_env) if hyst_env else None
                    )
                    print(f"[tzsafe] start={wstart} end={wend} th={th} hor={hor} "
                          f"crosses={crosses} bars>=th={bars_ge}")

                    # registramos (PF / WR etc. como NaN para compat con lectores)
                    rows.append({
                        "window": wlabel,
                        "asset": asset,
                        "horizon": int(hor),
                        "threshold": float(th),
                        "trades": int(crosses),
                        "bars_ge_th": int(bars_ge),
                        "pf": np.nan,
                        "wr": np.nan,
                        "mdd": np.nan,
                        "sortino": np.nan,
                        "roi": np.nan,
                        "roi_bh": np.nan,
                        "edge_vs_bh": np.nan,
                        "avg_trade": np.nan,
                        "_src": src,
                    })

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    print(f"[OK] Escrito {args.out_csv} rows={len(df_out)}")

    # "top" vacío por ahora (pre-check); deja el archivo por compat
    top_path = args.out_top
    pd.DataFrame(columns=["window", "asset", "horizon", "threshold", "trades", "pf", "wr"]).to_csv(top_path, index=False)
    print(f"[OK] Escrito {top_path} rows=0")


if __name__ == "__main__":
    main()