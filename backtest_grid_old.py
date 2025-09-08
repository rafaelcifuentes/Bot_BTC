#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
backtest_grid.py — Micro-grid de validación por ventanas (tz-aware, robusto)

- Acepta --windows, --assets, --horizons, --thresholds y paths de señales/OHLC.
- Carga señales y OHLC con índice tz-aware UTC.
- Cuenta entradas por "cruce" del score >= threshold con rearmado e histéresis.
- Backtest sencillo de salida por tiempo (horizon en minutos) sobre OHLC.
- Calcula KPIs y escribe CSVs de resultados y de los 'top' que pasan gates.

Limitaciones:
- Estrategia long-only, una posición a la vez (no solapa).
- "partial 50_50" y "breakeven_after_tp1" se aceptan pero se aproximan:
  aquí se modela una única salida a horizon. (Puedes refinar según tu lógica.)
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

# =========================================================
# Dataclass Window + parseo de --windows y normalización tz
# =========================================================

@dataclass
class Window:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp


def _to_utc_ts(x: Any) -> pd.Timestamp:
    """Convierte str/Timestamp/datetime a pd.Timestamp tz-aware en UTC."""
    if x is None:
        return None
    ts = x if isinstance(x, pd.Timestamp) else pd.Timestamp(x)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def parse_windows_arg(windows: List[str]) -> List[Window]:
    """
    Espera items del tipo "LABEL:YYYY-MM-DD:YYYY-MM-DD".
    Devuelve lista de Window con límites tz-aware UTC (end al nanoseg del día).
    """
    out: List[Window] = []
    for item in windows:
        parts = str(item).split(":")
        if len(parts) != 3:
            raise ValueError(f"Formato inválido en --windows: {item!r}")
        label, s_str, e_str = parts
        s = _to_utc_ts(pd.Timestamp(s_str))
        # set end al final del día
        e = _to_utc_ts(pd.Timestamp(e_str)) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        out.append(Window(label=label, start=s, end=e))
    return out


def iter_windows_generic(windows: Any) -> Iterator[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """
    Iterador robusto: dict, list/tuple ((label,start,end) | obj con attrs | dict),
    o un único objeto con .label/.start/.end.
    Devuelve (label, start_utc, end_utc).
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
            if all(hasattr(item, attr) for attr in ("label", "start", "end")):
                yield str(getattr(item, "label")), _to_utc_ts(getattr(item, "start")), _to_utc_ts(getattr(item, "end"))
                continue
            if isinstance(item, dict) and {"label", "start", "end"} <= set(item.keys()):
                yield str(item["label"]), _to_utc_ts(item["start"]), _to_utc_ts(item["end"])
                continue
            raise ValueError(f"Item inesperado en lista: {type(item)} - {item!r}")
        return

    if all(hasattr(windows, attr) for attr in ("label", "start", "end")):
        yield str(getattr(windows, "label")), _to_utc_ts(getattr(windows, "start")), _to_utc_ts(getattr(windows, "end"))
        return

    raise TypeError(f"Estructura de 'windows' no soportada: {type(windows)} - {windows!r}")


# =========================
# Utilidades de carga (tz)
# =========================

def ensure_index_tzaware(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura DatetimeIndex tz-aware UTC."""
    if not isinstance(df.index, pd.DatetimeIndex):
        # intenta convertir el index actual
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception as e:
            raise TypeError("El DataFrame debe tener un DatetimeIndex o un index convertible a datetime.") from e
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


def read_csv_tzaware(path: str) -> pd.DataFrame:
    """
    Lector robusto: intenta detectar columna temporal e índice. Acepta:
    - columna 'timestamp'/'time'/'date'/'datetime'
    - o el primer campo como fecha
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Intento 1: detectar columna de tiempo conocida
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"No se pudo leer CSV: {path}") from e

    # Buscar columna de tiempo
    time_cols = [c for c in df.columns if str(c).lower() in ("timestamp", "time", "date", "datetime")]
    if time_cols:
        tcol = time_cols[0]
        df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        df = df.set_index(tcol)
        df = df[~df.index.isna()]
        return ensure_index_tzaware(df)

    # Intento 2: usar el primer campo como fecha
    try:
        df.index = pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce")
        df = df.drop(columns=[df.columns[0]])
        df = df[~df.index.isna()]
        return ensure_index_tzaware(df)
    except Exception:
        pass

    # Intento 3: quizá ya viene con DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        return ensure_index_tzaware(df)

    raise RuntimeError(f"No se encontró columna temporal en {path}. Columnas: {list(df.columns)}")


def load_signals_from_root(root, wlabel, asset):
    """
    Devuelve (df, path) si existe un CSV de señales en root/wlabel/asset.csv.
    Si `root` es None/'none'/'' o el archivo no existe, devuelve (None, None)
    para que el caller haga fallback a OHLC.
    """
    if not root or str(root).strip().lower() in ("none", "auto"):
        return None, None
    path = os.path.join(root, wlabel, f"{asset}.csv")
    if not os.path.exists(path):
        return None, None
    df = read_csv_tzaware(path)
    return df, path

def load_ohlc_from_root(ohlc_root: str, asset: str) -> Tuple[pd.DataFrame, str]:
    """
    Carga OHLC de data/ohlc/1m/<asset>.csv (o carpeta que pases en --ohlc_root).
    Busca columna 'close' (insensible a may/min).
    """
    path = os.path.join(ohlc_root, f"{asset}.csv")
    df = read_csv_tzaware(path)

    # Detecta columna close
    close_cols = [c for c in df.columns if str(c).lower() == "close"]
    if not close_cols:
        # intenta heurística: última columna numérica
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise RuntimeError(f"No se encontró columna 'close' ni columnas numéricas en {path}")
        close_col = num_cols[-1]
    else:
        close_col = close_cols[0]

    # nos quedamos con close (y opcionalmente open/high/low si existen)
    keep = [c for c in df.columns if str(c).lower() in ("open", "high", "low", "close")]
    if close_col not in keep:
        keep.append(close_col)
    df = df[keep].rename(columns={close_col: "close"})
    return df.sort_index(), path


# ==========================================
# Señales: elegir columna y conteo de cruces
# ==========================================

def pick_signal_column(df: pd.DataFrame) -> str:
    """
    Elige la mejor columna de señal (score) de forma robusta:
    - Preferencia: 'score', 'signal', 'prob', 'p', 's'
    - Si no, última columna numérica.
    """
    prefs = ("score", "signal", "prob", "p", "s")
    cols_lower = {str(c).lower(): c for c in df.columns}
    for p in prefs:
        if p in cols_lower and pd.api.types.is_numeric_dtype(df[cols_lower[p]]):
            return cols_lower[p]
    # fallback: última numérica
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise RuntimeError("No hay columnas numéricas en DF de señales.")
    return num_cols[-1]


def count_crosses(series: pd.Series, threshold: float,
                  rearm_min: int = 0, hysteresis_pp: float = 0.0) -> int:
    """
    Cuenta cruces 'al alza' de series >= threshold, con rearmado e histéresis.
    - rearm_min: nº de barras por debajo de (threshold - hysteresis_pp) para rearmar.
    - Devuelve nº de entradas detectadas.
    """
    s = pd.Series(series).astype(float).copy()
    th_up = float(threshold)
    th_down = th_up - float(hysteresis_pp or 0.0)
    if th_down > th_up:
        th_down = th_up  # no dejes histéresis invertida

    above = s >= th_up
    crosses = 0
    armed = True
    below_count = 0

    prev_above = False
    for val, is_above in zip(s.values, above.values):
        if armed:
            # disparamos en flanco ascendente
            if is_above and not prev_above:
                crosses += 1
                armed = False
                below_count = 0
        else:
            # aún desarmado: esperamos rearmado por debajo de th_down
            if val <= th_down:
                below_count += 1
                if below_count >= int(rearm_min or 0):
                    armed = True
                    below_count = 0
            else:
                # sigue alto -> resetea contador
                below_count = 0
        prev_above = is_above

    return int(crosses)


# ===========================
# Backtest time-exit (simple)
# ===========================

def backtest_time_exit(
    ohlc: pd.DataFrame,
    sig: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon_min: int,
    threshold: float,
    fee_bps: float,
    slip_bps: float,
    partial: str = "none",
    breakeven_after_tp1: bool = False,
    risk_total_pct: float = 1.0,
) -> Dict[str, float]:
    """
    Backtest simple long-only: entra en cruces >= threshold y sale en t+horizon (min).
    - Ejecuta de forma no solapada (una posición a la vez) para simplificar.
    - Fees/slippage aplicados en entrada y salida.
    - partial/breakeven aceptados pero no modelados como TPs múltiples; salida única.

    Devuelve dict con KPIs: trades, wr, pf, mdd, sortino, roi, roi_bh, edge_vs_bh, avg_trade
    """

    # recorta a ventana (y reserva margen para horizon de salida)
    o = ohlc[(ohlc.index >= start) & (ohlc.index <= end + pd.Timedelta(minutes=horizon_min))].copy()
    if o.empty:
        return dict(trades=0, pf=0.0, wr=0.0, mdd=0.0, sortino=0.0, roi=0.0,
                    roi_bh=0.0, edge_vs_bh=0.0, avg_trade=0.0)

    close = o["close"]

    s = sig[(sig.index >= start) & (sig.index <= end)].copy()
    s = pd.Series(s).astype(float)

    # detectar cruces (mismo criterio que count_crosses)
    rm = int(os.getenv("REARM_MIN", "0") or "0")
    hys = float(os.getenv("HYSTERESIS_PP", "0") or "0")
    th = float(threshold)

    # flancos ascendentes:
    th_up = th
    th_down = th_up - hys
    if th_down > th_up:
        th_down = th_up

    above = s >= th_up

    entries: List[pd.Timestamp] = []
    armed = True
    below_count = 0
    prev_above = False
    for ts, (val, is_above) in zip(s.index, zip(s.values, above.values)):
        if armed:
            if is_above and not prev_above:
                entries.append(ts)
                armed = False
                below_count = 0
        else:
            if val <= th_down:
                below_count += 1
                if below_count >= rm:
                    armed = True
                    below_count = 0
            else:
                below_count = 0
        prev_above = is_above

    if not entries:
        # ROI buy&hold igual para consistencia
        bh = (close.loc[end] / close.loc[start] - 1.0) if (start in close.index and end in close.index) else 0.0
        return dict(trades=0, pf=0.0, wr=0.0, mdd=0.0, sortino=0.0, roi=0.0,
                    roi_bh=bh, edge_vs_bh=(0.0 - bh), avg_trade=0.0)

    # no solapar: si un entry cae antes de un exit, saltamos al siguiente tras el exit
    fee = (float(fee_bps) + float(slip_bps)) / 10000.0  # por lado
    capital = 1.0
    risk = float(risk_total_pct)
    rets: List[float] = []

    last_exit_time: Optional[pd.Timestamp] = None
    for t in entries:
        if last_exit_time is not None and t < last_exit_time:
            continue
        if t not in close.index:
            # si no hay precio exacto, usa el último <= t
            where = close.index.searchsorted(t, side="right") - 1
            if where < 0:
                continue
            t = close.index[where]

        exit_t = t + pd.Timedelta(minutes=int(horizon_min))
        # buscar barra de salida
        if exit_t not in close.index:
            where = close.index.searchsorted(exit_t, side="left")
            if where >= len(close.index):
                continue
            exit_t = close.index[where]

        p_in = float(close.loc[t])
        p_out = float(close.loc[exit_t])

        gross = (p_out / p_in) - 1.0
        net = gross - 2.0 * fee  # entrada+salida

        # aplica riesgo sobre capital
        rets.append(net * risk)

        # avanza puntero de solape
        last_exit_time = exit_t

    if not rets:
        bh = (close.loc[end] / close.loc[start] - 1.0) if (start in close.index and end in close.index) else 0.0
        return dict(trades=0, pf=0.0, wr=0.0, mdd=0.0, sortino=0.0, roi=0.0,
                    roi_bh=bh, edge_vs_bh=(0.0 - bh), avg_trade=0.0)

    rets_arr = np.array(rets, dtype=float)
    wins = rets_arr[rets_arr > 0]
    losses = rets_arr[rets_arr < 0]

    wr = float(len(wins)) / float(len(rets_arr)) if len(rets_arr) else 0.0
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else (np.inf if wins.sum() > 0 else 0.0)

    # equity curve y MDD
    eq = np.cumprod(1.0 + rets_arr)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    mdd = float(dd.min()) if len(dd) else 0.0

    # sortino (simple con rf=0)
    neg = rets_arr[rets_arr < 0]
    downside = neg.std(ddof=1) if len(neg) > 1 else (abs(neg).mean() if len(neg) == 1 else 0.0)
    sortino = (rets_arr.mean() / downside) * np.sqrt(252) if downside and downside > 0 else 0.0  # escala diaria aprox

    roi = float(eq[-1] - 1.0)

    # buy & hold en la ventana
    try:
        p0 = float(close.loc[start])
        p1 = float(close.loc[end])
        roi_bh = (p1 / p0) - 1.0
    except Exception:
        roi_bh = 0.0

    edge_vs_bh = roi - roi_bh
    avg_trade = float(rets_arr.mean())

    return dict(
        trades=int(len(rets_arr)),
        pf=float(0.0 if pf == np.inf else pf),
        wr=float(wr),
        mdd=float(mdd),
        sortino=float(sortino),
        roi=float(roi),
        roi_bh=float(roi_bh),
        edge_vs_bh=float(edge_vs_bh),
        avg_trade=float(avg_trade),
    )


# ================
# CLI y ejecución
# ================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Backtest grid por ventanas (tz-aware, robusto)")

    ap.add_argument("--windows", nargs="+", required=True,
                    help='Ej.: "2023Q4:2023-10-01:2023-12-31" "2024H1:2024-01-01:2024-06-30"')
    ap.add_argument("--assets", nargs="+", required=True)
    ap.add_argument("--horizons", nargs="+", type=int, required=True)
    ap.add_argument("--thresholds", nargs="+", type=float, required=True)

    ap.add_argument("--signals_root", default="reports/windows_fixed")
    ap.add_argument("--ohlc_root", default="data/ohlc/1m")

    ap.add_argument("--fee_bps", type=float, default=0.0)
    ap.add_argument("--slip_bps", type=float, default=0.0)

    ap.add_argument("--partial", choices=["none", "50_50"], default="none")
    ap.add_argument("--breakeven_after_tp1", action="store_true")

    ap.add_argument("--risk_total_pct", type=float, default=1.0)
    ap.add_argument("--weights", nargs="*", default=[], help="Formato: ASSET=WEIGHT (p.ej., BTC-USD=1.0)")
    ap.add_argument("--capital", type=float, default=1.0)

    ap.add_argument("--gate_pf", type=float, default=0.0)
    ap.add_argument("--gate_wr", type=float, default=0.0)
    ap.add_argument("--gate_trades", type=int, default=0)

    ap.add_argument("--out_csv", default="reports/val.csv")
    ap.add_argument("--out_top", default="reports/val_top.csv")

    return ap


def parse_weights(weights_args: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for w in weights_args:
        if "=" not in w:
            continue
        k, v = w.split("=", 1)
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def main() -> None:
    args = build_argparser().parse_args()

    # ENV para conteo de cruces
    rm = int(os.getenv("REARM_MIN", "0") or "0")
    hys = float(os.getenv("HYSTERESIS_PP", "0") or "0")
    tf = os.getenv("TREND_FILTER", "0")
    tmode = os.getenv("TREND_MODE", "soft")
    print(f"[tzsafe:pre] windows={args.windows} TF={bool(int(tf))} MODE={tmode} RM={rm} HYS={hys}")

    windows = parse_windows_arg(args.windows)

    rows: List[Dict[str, Any]] = []

    for asset in args.assets:
        # OHLC una sola vez por asset
        ohlc, ohlc_file = load_ohlc_from_root(args.ohlc_root, asset)

        for (wlabel, wstart, wend) in iter_windows_generic(windows):
            # Cargar señales específicas de la ventana+asset
            sig_df, sig_path = load_signals_from_root(args.signals_root, wlabel, asset)
            if sig_df is not None:
                sig_df = ensure_index_tzaware(sig_df)
            col_sig = pick_signal_column(sig_df)

            print(f"[tzsafe] start={wstart} end={wend} q=... TF={bool(int(tf))} MODE={tmode} RM={rm} HYS={hys}")

            for hor in args.horizons:
                for th in args.thresholds:
                    # para logging previo: solo conteo de cruces y barras >= th
                    sig_w = sig_df[(sig_df.index >= wstart) & (sig_df.index <= wend)][col_sig].astype(float)
                    if sig_w.empty:
                        crosses = 0
                        bars_ge = 0
                    else:
                        crosses = count_crosses(sig_w, float(th), rm, hys)
                        bars_ge = int((sig_w >= float(th)).sum())

                    print(f"[tzsafe] start={wstart} end={wend} th={th} hor={hor} crosses={crosses} bars>=th={bars_ge}")

                    # backtest
                    kpis = backtest_time_exit(
                        ohlc=ohlc,
                        sig=sig_df[col_sig],
                        start=wstart,
                        end=wend,
                        horizon_min=int(hor),
                        threshold=float(th),
                        fee_bps=float(args.fee_bps),
                        slip_bps=float(args.slip_bps),
                        partial=args.partial,
                        breakeven_after_tp1=bool(args.breakeven_after_tp1),
                        risk_total_pct=float(args.risk_total_pct),
                    )

                    rows.append(dict(
                        window=wlabel,
                        asset=asset,
                        horizon=int(hor),
                        threshold=float(th),
                        trades=int(kpis["trades"]),
                        pf=float(kpis["pf"]),
                        wr=float(kpis["wr"]),
                        mdd=float(kpis["mdd"]),
                        sortino=float(kpis["sortino"]),
                        roi=float(kpis["roi"]),
                        roi_bh=float(kpis["roi_bh"]),
                        edge_vs_bh=float(kpis["edge_vs_bh"]),
                        avg_trade=float(kpis["avg_trade"]),
                    ))

    # Resultados
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Escrito {args.out_csv} rows={len(out)}")

    # Filtrar 'top' por gates
    if len(out):
        passed = out[
            (out["pf"] >= float(args.gate_pf)) &
            (out["wr"] >= float(args.gate_wr)) &
            (out["trades"] >= int(args.gate_trades))
        ].copy()

        if len(passed):
            # ordenar por PF desc, WR desc, Trades desc
            passed = passed.sort_values(by=["pf", "wr", "trades"], ascending=[False, False, False])
        else:
            # si nadie pasa, deja vacío
            passed = passed.copy()

        out_top_path = args.out_top
        os.makedirs(os.path.dirname(out_top_path), exist_ok=True)
        passed.to_csv(out_top_path, index=False)
        print(f"[OK] Escrito {out_top_path} rows={len(passed)}")
    else:
        print("[WARN] No hay filas para calcular top; no se escribió out_top.")


if __name__ == "__main__":
    main()