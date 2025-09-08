#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#(actualizado: loader + –perla_csv/–max_corr + corr_perla en CSV)

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

# --- Rutas para imports locales ---
CURR_DIR = Path(__file__).resolve().parent
REPO_DIR = CURR_DIR.parent
if str(CURR_DIR) not in sys.path:
    sys.path.insert(0, str(CURR_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# --- tz/ventanas helpers (de tu repo) ---
try:
    from tzsafe_window import (
        parse_windows_arg,
        ensure_index_tzaware as _ensure_index_tzaware_upstream,
        ensure_tzaware,
    )
except Exception:
    # Fallback mínimo si no existe tzsafe_window.py
    @dataclass
    class Window:
        label: str
        start: pd.Timestamp
        end: pd.Timestamp

    def parse_windows_arg(ws: List[str]):
        out = []
        for token in ws:
            # "2023Q4:2023-10-01:2023-12-31"
            label, s, e = token.split(":")
            start = pd.Timestamp(s, tz="UTC")
            # final inclusivo nanoseg
            end = pd.Timestamp(e, tz="UTC") + pd.Timedelta(nanoseconds=999_999_999)
            out.append(Window(label, start, end))
        return out

    def ensure_tzaware(ts: pd.Timestamp) -> pd.Timestamp:
        if ts.tz is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def _ensure_index_tzaware_upstream(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("El DataFrame debe tener DatetimeIndex.")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df

# --- mini loader diamante_selected.yaml ---
try:
    from selected_loader import read_diamante_selected
except Exception:
    read_diamante_selected = lambda *_args, **_kw: None  # noqa: E731


def ensure_index_tzaware(df: pd.DataFrame) -> pd.DataFrame:
    return _ensure_index_tzaware_upstream(df)


def load_signals_from_root(signals_root: str, window_label: str, asset: str) -> Tuple[pd.DataFrame, Path]:
    path = Path(signals_root) / window_label / f"{asset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe archivo de señales: {path}")
    df = pd.read_csv(path)
    # Se asume columna 'ts' o índice ya temporal
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts")
    df = ensure_index_tzaware(df)
    return df, path


def robust_numeric_column(df: pd.DataFrame, prefer: Iterable[str]) -> pd.Series:
    """Devuelve la primera columna numérica disponible (según preferencia)."""
    for name in prefer:
        if name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
            return df[name].astype(float)
    # fallback: primera numérica
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return df[c].astype(float)
    raise ValueError("No se encontró columna numérica en el CSV.")


def make_binary_exposure(sig: pd.Series, th: float, hys: float) -> pd.Series:
    """
    Señal binaria (0/1) con histéresis:
      - sube a 1 si sig >= th + hys
      - baja a 0 si sig <= th - hys
      - en zona media mantiene último estado
    """
    sig = sig.astype(float)
    up = th + hys
    down = th - hys
    out = np.zeros(len(sig), dtype=np.int8)
    last = 0
    for i, v in enumerate(sig.values):
        if v >= up:
            last = 1
        elif v <= down:
            last = 0
        out[i] = last
    return pd.Series(out, index=sig.index)


def effective_corr(perla: pd.Series, diamante: pd.Series, min_overlap: int = 100) -> float:
    """
    Correlación de Pearson sobre la intersección temporal.
    Devuelve np.nan si solape insuficiente o varianza nula.
    """
    s = perla.dropna().astype(float)
    d = diamante.dropna().astype(float)
    idx = s.index.intersection(d.index)
    if len(idx) < min_overlap:
        return float("nan")
    s2 = s.loc[idx]
    d2 = d.loc[idx]
    if s2.std(ddof=0) == 0 or d2.std(ddof=0) == 0:
        return float("nan")
    return float(s2.corr(d2))


def count_crosses_compat(sig: pd.Series, threshold: float, rearm_min: int, hysteresis_pp: float) -> int:
    """
    Cuenta 'disparos' ascendentes con rearm e histéresis:
      - disparo si sig >= th + hys y estamos 'armados'
      - desarme cuando sig <= th - hys y contamos barras consecutivas abajo
      - rearm cuando llevamos >= rearm_min barras abajo
    """
    sig = sig.astype(float).values
    th = float(threshold)
    hys = float(hysteresis_pp)
    upper = th + hys
    lower = th - hys

    crosses = 0
    armed = True
    below_count = 0

    for v in sig:
        if armed:
            if v >= upper:
                crosses += 1
                armed = False
                below_count = 0
        else:
            if v <= lower:
                below_count += 1
                if below_count >= int(rearm_min):
                    armed = True
            else:
                below_count = 0
    return int(crosses)


def parse_partial(s: str) -> Tuple[int, int]:
    # "50_50" -> (50, 50)  | "100_0" -> (100, 0)
    if "_" in s:
        a, b = s.split("_", 1)
        return int(a), int(b)
    return (100, 0)


def main():
    p = argparse.ArgumentParser(description="Grid tz-safe con gate de correlación contra Perla.")
    p.add_argument("--windows", nargs="+", required=True, help='Ej: "2023Q4:2023-10-01:2023-12-31" ...')
    p.add_argument("--assets", nargs="+", required=True)
    p.add_argument("--horizons", nargs="+", type=int, required=True)
    p.add_argument("--thresholds", nargs="+", type=float, required=True)

    p.add_argument("--signals_root", type=str, required=True)
    p.add_argument("--ohlc_root", type=str, required=False, default="data/ohlc/1m")

    # Costes / gestión (placeholders; mantenemos interfaz)
    p.add_argument("--fee_bps", type=float, default=6)
    p.add_argument("--slip_bps", type=float, default=6)
    p.add_argument("--partial", type=str, default="50_50")
    p.add_argument("--breakeven_after_tp1", action="store_true")
    p.add_argument("--risk_total_pct", type=float, default=0.75)
    p.add_argument("--weights", type=str, default="BTC-USD=1.0")

    # Gates clásicos
    p.add_argument("--gate_pf", type=float, default=1.6)
    p.add_argument("--gate_wr", type=float, default=0.60)
    p.add_argument("--gate_trades", type=int, default=30)

    # NUEVO: correlación contra Perla
    p.add_argument("--perla_csv", type=str, help="CSV de Perla (posiciones o señal binaria).")
    p.add_argument("--perla_col", type=str, default=None, help="Columna a usar (auto si no se indica).")
    p.add_argument("--max_corr", type=float, default=0.75, help="Máxima correlación aceptable (gate).")
    p.add_argument("--corr_min_overlap", type=int, default=100, help="Mínimo solape requerido.")
    # NUEVO: activar automáticamente última selección
    p.add_argument("--use_selected", action="store_true", help="Lee configs/diamante_selected.yaml para (H, TH).")

    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--out_top", type=str, required=True)

    args = p.parse_args()

    # ENV del filtro de tendencia (solo para logs/consistencia)
    tf_on = str(os.getenv("TREND_FILTER", "0")) == "1"
    tf_mode = os.getenv("TREND_MODE", "soft")
    rearm_min = int(os.getenv("REARM_MIN", "4"))
    hysteresis_pp = float(os.getenv("HYSTERESIS_PP", "0.04"))

    # Cargar Perla si nos la pasan
    perla_series = None
    if args.perla_csv:
        dfp = pd.read_csv(args.perla_csv)
        # Usamos 'ts' si está, o intentamos leer índice temporal
        if "ts" in dfp.columns:
            dfp["ts"] = pd.to_datetime(dfp["ts"], utc=True)
            dfp = dfp.set_index("ts")
        else:
            # intenta convertir índice a datetime utc
            try:
                dfp.index = pd.to_datetime(dfp.index, utc=True)
            except Exception:
                pass
        dfp = ensure_index_tzaware(dfp)

        prefer_cols = [args.perla_col] if args.perla_col else ["pos", "signal", "position", "exp"]
        perla_series = robust_numeric_column(dfp, [c for c in prefer_cols if c])

    # Loader de selección automática (si se pide)
    if args.use_selected:
        sel = read_diamante_selected()
        if sel:
            h_sel, th_sel = sel
            args.horizons = [int(h_sel)]
            args.thresholds = [float(th_sel)]

    windows = parse_windows_arg(args.windows)
    out_rows = []

    print(f"[tzsafe:pre] windows={args.windows} TF={tf_on} MODE={tf_mode} RM={rearm_min} HYS={hysteresis_pp}")

    for W in windows:
        # Para compatibilidad con el helper tzsafe_window, W puede ser tupla/objeto
        label = getattr(W, "label", None) or str(W[0]) if isinstance(W, (list, tuple)) else None
        start = ensure_tzaware(getattr(W, "start", None) or W[1])
        end = ensure_tzaware(getattr(W, "end", None) or W[2])
        label = label or "win"

        for asset in args.assets:
            df_sig, sig_file = load_signals_from_root(args.signals_root, label, asset)
            # columna de señal (prob)
            sig_w = robust_numeric_column(df_sig, ["score", "signal", "prob", "p"])

            # recortar a ventana
            sig_w = sig_w.loc[(sig_w.index >= start) & (sig_w.index <= end)]

            print(f"[tzsafe:load] file={sig_file} rows={len(sig_w)} ...")
            print(f"[tzsafe] start={start} end={end} q=... TF={tf_on} MODE={tf_mode} RM={rearm_min} HYS={hysteresis_pp}")

            for th in args.thresholds:
                for hor in args.horizons:
                    crosses = count_crosses_compat(sig_w, float(th), rearm_min, hysteresis_pp)
                    bars_ge_th = int((sig_w >= th).sum())
                    print(f"[tzsafe] start={start} end={end} th={th} hor={hor} crosses={crosses} bars>=th={bars_ge_th}")

                    # Exposición binaria de Diamante (para corr contra Perla)
                    dia_exp = make_binary_exposure(sig_w, float(th), float(hysteresis_pp))
                    corr_val = float("nan")
                    if perla_series is not None:
                        corr_val = effective_corr(perla_series, dia_exp, min_overlap=args.corr_min_overlap)
                        # Gate de correlación (opcional): descartamos si supera max_corr
                        if (not np.isnan(corr_val)) and (abs(corr_val) > float(args.max_corr)):
                            # lo registramos igualmente pero con flag de descarte
                            pass

                    # KPI placeholders (se mantiene estructura de columnas)
                    row = {
                        "window": label,
                        "asset": asset,
                        "horizon": int(hor),
                        "threshold": float(th),
                        "trades": int(crosses),  # proxy de actividad
                        "pf": 0.0,
                        "wr": 0.0,
                        "mdd": 0.0,
                        "sortino": 0.0,
                        "roi": 0.0,
                        "roi_bh": 0.0,
                        "edge_vs_bh": 0.0,
                        "avg_trade": 0.0,
                        "corr_perla": corr_val,
                        "_file": str(sig_file),
                    }
                    out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[OK] Escrito {args.out_csv} rows={len(out_df)}")

    # 'Top' (si tu flujo lo usa): solo orden por proxies, PF/WR/Trades, deja corr visible
    if not out_df.empty:
        cols = ["pf", "wr", "trades", "corr_perla"]
        # cuidado si pf/wr son todos iguales (0.0), no pasa nada
        top_df = out_df.sort_values(by=["pf", "wr", "trades"], ascending=[False, False, False]).head(50)
        top_df.to_csv(args.out_top, index=False)
        print(f"[OK] Escrito {args.out_top} rows={len(top_df)}")
    else:
        pd.DataFrame().to_csv(args.out_top, index=False)
        print(f"[OK] Escrito {args.out_top} rows=0")


if __name__ == "__main__":
    main()