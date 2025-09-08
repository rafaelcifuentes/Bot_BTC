#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
runner_profileA_RF_sentiment_EXP.py

Runner "Profile A — RF Sentiment EXP" (freeze/live) con:
- Descarga OHLCV 4h (ccxt/Binance)
- ADX 4h y ADX1D (resample)
- Gate de sentimiento (Fear & Greed + Funding) opcional
- Micro-grid alrededor del threshold (si no hay --repro_lock)
- Métricas 90d/180d (Net, PF, MDD, Win%, Trades, Score)
- Artefactos: summary JSON + walkforward CSV en ./reports/

Requisitos: pandas, numpy, pandas_ta, ccxt
"""


import argparse
import datetime as dt
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# pandas_ta emite un warning por pkg_resources en algunos entornos (OK)
try:
    import pandas_ta as ta
except Exception as e:
    print(f"[warn] pandas_ta no disponible: {e}", file=sys.stderr)
    ta = None

# ccxt para OHLCV
try:
    import ccxt
except Exception as e:
    ccxt = None
    print(f"[warn] ccxt no disponible: {e}", file=sys.stderr)

RUNNER_NAME = os.getenv("RUNNER_NAME", "Corazón")  # antes “Sentiment EXP”

# ----------------------------
# Utilidades
# ----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def utcnow():
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)


def fmt_ts(ts: pd.Timestamp) -> str:
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso_date(s: str) -> pd.Timestamp:
    # Asume fecha (YYYY-MM-DD). Devuelve fin del día en UTC.
    d = pd.to_datetime(s, utc=True)
    if d.hour == 0 and d.minute == 0 and d.second == 0:
        d = d + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return d


def max_drawdown(equity: pd.Series) -> float:
    """Equity en escala relativa (arranca en 1.0). Devuelve MDD en [0,1]."""
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    return float(-dd.min()) if not math.isnan(dd.min()) else 0.0


def profit_factor(trade_returns: np.ndarray) -> float:
    if trade_returns.size == 0:
        return 0.0
    gains = trade_returns[trade_returns > 0].sum()
    losses = -trade_returns[trade_returns < 0].sum()
    if losses <= 1e-12:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def segment_trade_returns(returns: pd.Series, pos: pd.Series) -> np.ndarray:
    """Agrega retornos por tramo continuo con pos==1 (long)."""
    pos = pos.fillna(0).astype(int)
    # Identifica comienzos de trade
    start = (pos.diff().fillna(0) == 1)
    trade_ids = start.cumsum() * (pos > 0).astype(int)
    # Agrupa por trade_id > 0
    tr = []
    for tid, grp in returns.groupby(trade_ids):
        if tid == 0:
            continue
        # retorno compuesto del trade: prod(1+r) - 1
        r = float((1.0 + grp).prod() - 1.0)
        tr.append(r)
    return np.array(tr, dtype=float)


def time_window_mask(idx: pd.DatetimeIndex, days: int) -> pd.Series:
    if idx.size == 0:
        return pd.Series([], dtype=bool)
    end = idx[-1]
    start = end - pd.Timedelta(days=days)
    return (idx >= start) & (idx <= end)


def normalize_series(x: pd.Series) -> pd.Series:
    if x.std(skipna=True) == 0 or x.count() < 2:
        return pd.Series(0.0, index=x.index)
    return (x - x.mean(skipna=True)) / x.std(skipna=True)


def robust_first_numeric_column(df: pd.DataFrame) -> pd.Series:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return df[c]
    # si ninguna es numérica, intenta forzar
    for c in df.columns:
        try:
            return pd.to_numeric(df[c], errors="coerce")
        except Exception:
            continue
    raise ValueError("No se encontró columna numérica usable.")


# ----------------------------
# Carga datos
# ----------------------------
def fetch_ohlcv_4h(symbol="BTC/USDT", limit=5400) -> pd.DataFrame:
    sources = []
    df = None
    if ccxt is not None:
        try:
            ex = ccxt.binance()
            ex.load_markets()
            data = ex.fetch_ohlcv(symbol, timeframe="4h", limit=limit)
            df = pd.DataFrame(
                data, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()
            sources.append("binance: ok")
        except Exception as e:
            sources.append(f"binance: {e.__class__.__name__}")
    else:
        sources.append("binance: ccxt not installed")

    if df is None or df.empty:
        # fallback: intenta archivo local
        local = "./data/ohlcv_btcusdt_4h.csv"
        if os.path.exists(local):
            try:
                tdf = pd.read_csv(local)
                # heurística de columnas
                time_col = None
                for c in tdf.columns:
                    if "time" in c.lower() or "date" in c.lower():
                        time_col = c
                        break
                if time_col is None:
                    time_col = tdf.columns[0]
                tdf[time_col] = pd.to_datetime(tdf[time_col], utc=True)
                tdf = tdf.rename(
                    columns={time_col: "timestamp"}
                ).set_index("timestamp")
                # asegurar columnas
                need = ["open", "high", "low", "close", "volume"]
                rename_map = {}
                for c in need:
                    if c not in tdf.columns:
                        # busca aproximado
                        for col in tdf.columns:
                            if col.lower().startswith(c):
                                rename_map[col] = c
                                break
                if rename_map:
                    tdf = tdf.rename(columns=rename_map)
                df = tdf[["open", "high", "low", "close", "volume"]].sort_index()
                sources.append("local_csv: ok")
            except Exception as e:
                sources.append(f"local_csv: {e.__class__.__name__}")

    if df is None or df.empty:
        raise RuntimeError("No se pudo obtener OHLCV 4h de ninguna fuente.")

    print("🔄 Downloading 4h data (ccxt/Binance + fallbacks)…")
    print(f"Sources: {sources}")
    return df


def load_timeseries_csv(path: str) -> Optional[pd.Series]:
    """Lee CSV genérico y devuelve una Serie (index datetime UTC, values numéricos)."""
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        # detectar columna temporal
        time_col = None
        for c in df.columns:
            cl = c.lower()
            if "time" in cl or "date" in cl:
                time_col = c
                break
        if time_col is None:
            time_col = df.columns[0]
        # parseo de fecha
        ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        # primera numérica
        val = robust_first_numeric_column(df.drop(columns=[time_col], errors="ignore"))
        s = pd.Series(val.values, index=ts)
        s = s.sort_index()
        # normaliza valores tipo porcentaje si vienen en %
        if s.abs().max() > 5 and "fund" in path.lower():
            # funding suele ser < 0.1 en términos absolutos
            s = s / 100.0 if s.abs().max() > 1 else s
        return s
    except Exception as e:
        print(f"[warn] No se pudo leer {path}: {e}")
        return None


# ----------------------------
# ADX y sentimiento
# ----------------------------
def compute_adx4h(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    if ta is None:
        raise RuntimeError("pandas_ta es requerido para ADX.")
    adx = ta.adx(df["high"], df["low"], df["close"], length=length)
    # columnas típicas: DMP_14, DMN_14, ADX_14
    return adx


def compute_adx1d_resample(df: pd.DataFrame, length: int = 14) -> pd.Series:
    if ta is None:
        raise RuntimeError("pandas_ta es requerido para ADX.")
    o = df["open"].resample("1D").first()
    h = df["high"].resample("1D").max()
    l = df["low"].resample("1D").min()
    c = df["close"].resample("1D").last()
    adx_d = ta.adx(h, l, c, length=length)
    adx1d = adx_d["ADX_{}".format(length)].reindex(df.index).ffill()
    return adx1d


@dataclass
class SentimentInputs:
    fg_series: Optional[pd.Series]
    funding_series: Optional[pd.Series]


def sentiment_gate(
    df: pd.DataFrame,
    fg_series: Optional[pd.Series],
    funding_series: Optional[pd.Series],
    fg_long_min: float,
    fg_short_max: float,
    funding_bias: float,
    verbose: bool = True,
) -> Tuple[pd.Series, str]:
    """Devuelve (sent_ok por barra, mensaje diagnóstico)."""
    if fg_series is None or funding_series is None:
        msg = "Sentiment gate (diagnostic) skipped: " + (
            "./data/sentiment/fear_greed.csv" if fg_series is None else ""
        )
        if fg_series is None and funding_series is None:
            msg = "Sentiment gate (diagnostic) skipped: missing FG & Funding CSV"
        elif funding_series is None:
            msg = "Sentiment gate (diagnostic) skipped: ./data/sentiment/funding_rates.csv"
        if verbose:
            print(msg)
        return pd.Series(False, index=df.index), msg

    try:
        fg_aligned = fg_series.reindex(df.index).ffill()
        funding_aligned = funding_series.reindex(df.index).ffill()

        cov_fg = 100.0 * fg_aligned.notna().mean()
        cov_fund = 100.0 * funding_aligned.notna().mean()

        fg_ok = (fg_aligned >= fg_long_min) & (fg_aligned <= fg_short_max)
        funding_ok = funding_aligned.abs() <= funding_bias

        sent_ok = (fg_ok & funding_ok).reindex(df.index).fillna(False)

        total = len(df.index)
        passed = int(sent_ok.sum())
        pct = (100.0 * passed / total) if total else 0.0

        msg = (
            f"Sentiment gate → {passed}/{total} bars ({pct:.2f}%) pass — "
            f"FG∈[{fg_long_min:.2f},{fg_short_max:.2f}], |fund|≤{funding_bias:.3f}; "
            f"coverage FG {cov_fg:.2f}%, funding {cov_fund:.2f}%"
        )
        if verbose:
            print(msg)
        return sent_ok, msg
    except Exception as e:
        msg = f"Sentiment gate (diagnostic) skipped: {e}"
        if verbose:
            print(msg)
        return pd.Series(False, index=df.index), msg


# ----------------------------
# Señal y evaluación
# ----------------------------
def build_model_score(
    df: pd.DataFrame, adx4: pd.DataFrame, adx1d: pd.Series, sent_ok: Optional[pd.Series]
) -> pd.Series:
    """
    Modelo EXP simplificado → score en [0,1].
    Combina momentum (RSI), fuerza de tendencia (ADX) y filtro de sentimiento si está disponible.
    """
    close = df["close"]
    rsi = ta.rsi(close, length=14) if ta is not None else pd.Series(50.0, index=df.index)
    rsi_norm = (rsi - 50.0) / 50.0  # ~[-1,1]
    adx4_norm = (adx4["ADX_14"] / 50.0).clip(0, 1)  # ~[0,1]
    adx1d_norm = (adx1d / 50.0).clip(0, 1)

    base = 0.50 + 0.35 * rsi_norm.fillna(0) + 0.25 * adx4_norm.fillna(0) + 0.15 * adx1d_norm.fillna(0)
    score = 1 / (1 + np.exp(-2.0 * (base - 0.5)))  # squashing

    if sent_ok is not None:
        # bonifica levemente si pasa el gate de sentimiento
        score = (score + 0.05 * sent_ok.astype(float)).clip(0, 1)

    return score.fillna(0.5)


@dataclass
class Metrics:
    net: float
    pf: float
    score: float
    mdd: float
    trades: int
    winrate: float


def evaluate_window_metrics(
    df: pd.DataFrame, pos: pd.Series, days: int
) -> Metrics:
    mask = time_window_mask(df.index, days)
    if not mask.any():
        return Metrics(0.0, 0.0, 0.0, 0.0, 0, 0.0)

    close = df["close"].copy()
    returns = close.pct_change().fillna(0.0)
    # retorno sólo cuando hay posición (usa pos de la barra ANTERIOR)
    r_used = returns.where(pos.shift(1).fillna(0) > 0, 0.0)

    r_win = r_used[mask]

    equity = (1.0 + r_win).cumprod()
    net = float(equity.iloc[-1] - 1.0) * 100.0  # en %
    mdd = max_drawdown(equity)

    # por trades
    tr_array = segment_trade_returns(r_win, pos.reindex(df.index).fillna(0))
    pf = profit_factor(tr_array)
    wins = float((tr_array > 0).sum())
    trades = int(tr_array.size)
    winrate = float(100.0 * wins / trades) if trades > 0 else 0.0

    # score compuesto (acotado para que 0..~130)
    pf_norm = min(pf, 2.0) / 2.0  # 0..1
    score = 100.0 * (0.70 * pf_norm + 0.30 * (1.0 - mdd))  # ~[0..130]

    return Metrics(net=net, pf=pf, score=score, mdd=mdd, trades=trades, winrate=winrate)


# ----------------------------
# Micro-grid de threshold
# ----------------------------
def micro_grid_threshold(
    base_thr: float,
    df: pd.DataFrame,
    adx4: pd.DataFrame,
    adx1d: pd.Series,
    adx1d_min: float,
    adx4_min: float,
    sent_ok: Optional[pd.Series],
    repro_lock: bool,
) -> float:
    print("\n— Micro-grid (around locked THRESHOLD) —")
    if repro_lock:
        print("Repro lock ON → micro-grid skip.")
        print(f"Picked THRESHOLD (locked): {base_thr:.3f}")
        return base_thr

    # candidatos: base±0.02, paso 0.005, acotado 0.50..0.65
    lo = max(0.50, base_thr - 0.02)
    hi = min(0.65, base_thr + 0.02 + 1e-9)
    candidates = np.round(np.arange(lo, hi + 1e-9, 0.005), 3)

    best_thr = base_thr
    best_score = -1e9

    # gates fijos
    adx1d_ok = (adx1d >= adx1d_min).reindex(df.index).fillna(False)
    adx4_ok = (adx4["ADX_14"] >= adx4_min).reindex(df.index).fillna(False)
    if sent_ok is None:
        sent_ok = pd.Series(True, index=df.index)

    # features
    model_score = build_model_score(df, adx4, adx1d, sent_ok)

    for t in candidates:
        signal = (model_score >= t) & adx1d_ok & adx4_ok & sent_ok
        # FIX: pos con mismo índice/longitud
        pos = pd.Series(
            np.where(signal.shift(1).fillna(False).values, 1, 0),
            index=df.index,
            dtype=np.int8,
        )
        m180 = evaluate_window_metrics(df, pos, 180)
        print(f"  t={t:.3f} → score180={m180.score:.3f}")
        if m180.score > best_score:
            best_score = m180.score
            best_thr = float(t)

    print(f"Picked THRESHOLD: {best_thr:.3f}")
    return best_thr


# ----------------------------
# Semáforo (entorno)
# ----------------------------
def print_traffic_light(
    df: pd.DataFrame,
    adx1d: pd.Series,
    adx4: pd.DataFrame,
    fg_series: Optional[pd.Series],
    funding_series: Optional[pd.Series],
    args,
):
    import pandas as pd

    def _light(ok, warn=False):
        if ok:
            return "🟢"
        return "🟡" if warn else "🔴"

    adx1d_val = adx1d.reindex(df.index).ffill().iloc[-1]
    adx4_val = adx4["ADX_14"].reindex(df.index).ffill().iloc[-1]
    fg_now = np.nan
    fund_now = np.nan
    if fg_series is not None:
        fg_now = fg_series.reindex(df.index).ffill().iloc[-1]
    if funding_series is not None:
        fund_now = funding_series.reindex(df.index).ffill().iloc[-1]

    adx1d_ok = adx1d_val >= args.adx1d_min
    adx4_ok = adx4_val >= getattr(args, "adx4_min", 0)
    sent_ok = (
        (not math.isnan(fg_now))
        and (not math.isnan(fund_now))
        and (args.fg_long_min <= fg_now <= args.fg_short_max)
        and (abs(fund_now) <= args.funding_bias)
    )

    # zona amarilla si está dentro del 10% del umbral
    warn1 = (not adx1d_ok) and (adx1d_val >= 0.9 * args.adx1d_min)
    warn4 = (not adx4_ok) and (adx4_val >= 0.9 * getattr(args, "adx4_min", 0))
    warnS = (not sent_ok) and (
        (not math.isnan(fg_now))
        and (not math.isnan(fund_now))
        and (
            (min(abs(fg_now - args.fg_long_min), abs(args.fg_short_max - fg_now)) <= 0.02)
            or (abs(abs(fund_now) - args.funding_bias) <= 0.002)
        )
    )

    win = df.index >= (df.index[-1] - pd.Timedelta(days=30))
    adx1d_p30 = 100.0 * (adx1d.reindex(df.index).ffill()[win] >= args.adx1d_min).mean()
    adx4_p30 = 100.0 * (adx4["ADX_14"].reindex(df.index).ffill()[win] >= getattr(args, "adx4_min", 0)).mean()

    sent_p30 = np.nan
    if (fg_series is not None) and (funding_series is not None):
        fg_ok30 = fg_series.reindex(df.index).ffill()[win].between(args.fg_long_min, args.fg_short_max)
        fund_ok30 = (funding_series.reindex(df.index).ffill()[win].abs() <= args.funding_bias)
        sent_p30 = 100.0 * (fg_ok30 & fund_ok30).mean()

    sent_now_txt = (
        f"Sent FG {fg_now:.2f}∈[{args.fg_long_min:.2f},{args.fg_short_max:.2f}], |fund| {abs(fund_now):.3f}≤{args.funding_bias:.3f} "
        if not math.isnan(fg_now) and not math.isnan(fund_now)
        else "Sent (no data) "
    )

    sent_30d_txt = f"[30d {sent_p30:.1f}%]" if not math.isnan(sent_p30) else "[30d n/a]"

    print(
        "Semáforo v0 → "
        f"ADX1D {adx1d_val:.1f}/{args.adx1d_min} {_light(adx1d_ok, warn1)} [30d {adx1d_p30:.1f}%] | "
        f"ADX4h {adx4_val:.1f}/{getattr(args,'adx4_min',0)} {_light(adx4_ok, warn4)} [30d {adx4_p30:.1f}%] | "
        f"{sent_now_txt}"
        f"{_light(sent_ok, warnS)} {sent_30d_txt}"
    )


# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Runner ProfileA — RF Sentiment EXP")
    p.add_argument("--primary_only", action="store_true", help="Ignora secundarios (compatibilidad).")

    # Freeze & lock
    p.add_argument("--freeze_end", type=str, default=None, help="YYYY-MM-DD (fin del período de evaluación).")
    p.add_argument("--repro_lock", action="store_true", help="Lock para reproducibilidad (omite micro-grid).")

    # Señales / umbrales
    p.add_argument("--threshold", type=float, default=0.61)
    p.add_argument("--no_sentiment", action="store_true", help="Desactiva uso de sentimiento.")
    p.add_argument("--use_sentiment", action="store_true", help="Activa uso de sentimiento.")
    p.add_argument("--fg_csv", type=str, default="./data/sentiment/fear_greed.csv")
    p.add_argument("--funding_csv", type=str, default="./data/sentiment/funding_rates.csv")
    p.add_argument("--fg_long_min", type=float, default=-0.18)
    p.add_argument("--fg_short_max", type=float, default=0.18)
    p.add_argument("--funding_bias", type=float, default=0.01)

    # Gestión (trailing/SL/TP) — quedan para trazabilidad
    p.add_argument("--pt_mode", type=str, default="pct", choices=["pct"], help="Modo de parámetros de trade.")
    p.add_argument("--sl_pct", type=float, default=0.02)
    p.add_argument("--tp_pct", type=float, default=0.10)
    p.add_argument("--trail_pct", type=float, default=0.015)

    # ADX
    p.add_argument("--adx_daily_source", type=str, default="resample", choices=["resample"], help="Fuente de ADX1D.")
    p.add_argument("--adx1d_len", type=int, default=14)
    p.add_argument("--adx1d_min", type=float, default=30.0)
    p.add_argument("--adx4_min", type=float, default=18.0)

    # Extra
    p.add_argument("--traffic_light", action="store_true", help="Imprimir semáforo del entorno.")
    args = p.parse_args()

    # Determina modo
    mode = "freeze" if args.freeze_end else "live"

    # 1) Datos
    df = fetch_ohlcv_4h(limit=5400)
    if args.freeze_end:
        end = parse_iso_date(args.freeze_end)
        df = df.loc[:end].copy()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()

    # 2) ADX
    adx4 = compute_adx4h(df, length=14)
    adx4_ok = (adx4["ADX_14"] >= args.adx4_min).reindex(df.index).fillna(False)
    adx4_pass = int(adx4_ok.sum())
    adx4_pct = 100.0 * adx4_ok.mean()
    print(f"ADX4h gate → {adx4_pass}/{len(df)} bars ({adx4_pct:.2f}%) pass — thr {args.adx4_min:.1f}")

    adx1d = compute_adx1d_resample(df, length=args.adx1d_len)
    adx1d_ok = (adx1d >= args.adx1d_min).reindex(df.index).fillna(False)
    adx1d_pass = int(adx1d_ok.sum())
    adx1d_pct = 100.0 * adx1d_ok.mean()
    print("ADX1D gate → ADX1D → {}/{} bars ({:.2f}%) pass".format(adx1d_pass, len(df), adx1d_pct))
    print(f"Source: resample len: {args.adx1d_len}")

    # 3) Sentimiento (opcional)
    fg_series = None
    funding_series = None
    sent_ok = None
    if args.use_sentiment and not args.no_sentiment:
        fg_series = load_timeseries_csv(args.fg_csv)
        funding_series = load_timeseries_csv(args.funding_csv)
        sent_ok, _ = sentiment_gate(
            df, fg_series, funding_series, args.fg_long_min, args.fg_short_max, args.funding_bias, verbose=True
        )
    else:
        print("Sentiment gate (disabled by flags).")

    # 4) Semáforo del entorno
    if args.traffic_light:
        print_traffic_light(df, adx1d, adx4, fg_series, funding_series, args)

    # 5) Micro-grid threshold
    thr = micro_grid_threshold(
        base_thr=float(args.threshold),
        df=df,
        adx4=adx4,
        adx1d=adx1d,
        adx1d_min=args.adx1d_min,
        adx4_min=args.adx4_min,
        sent_ok=sent_ok if (args.use_sentiment and not args.no_sentiment) else None,
        repro_lock=args.repro_lock,
    )

    # 6) Señal final con threshold elegido
    model_score = build_model_score(df, adx4, adx1d, sent_ok if (args.use_sentiment and not args.no_sentiment) else None)
    signal = (model_score >= thr) & adx1d_ok & adx4_ok
    if args.use_sentiment and not args.no_sentiment and sent_ok is not None:
        signal = signal & sent_ok

    # FIX de longitudes/índices en pos
    pos = pd.Series(
        np.where(signal.shift(1).fillna(False).values, 1, 0),
        index=df.index,
        dtype=np.int8,
    )

    # 7) Métricas
    m90 = evaluate_window_metrics(df, pos, 90)
    m180 = evaluate_window_metrics(df, pos, 180)

    # 8) Artefactos
    ensure_dir("./reports")
    ts = utcnow().strftime("%Y%m%d_%H%M%S")
    sum_path = f"./reports/summary_rf_sentiment_EXP_{ts}.json"
    wf_path = f"./reports/walkforward_rf_sentiment_EXP_{ts}.csv"

    # Walkforward CSV
    wf = pd.DataFrame(
        {
            "timestamp": df.index.tz_convert("UTC").astype("datetime64[ns]"),
            "close": df["close"].values,
            "pos": pos.values,
        }
    )
    wf.to_csv(wf_path, index=False)

    # Summary JSON (mismo esquema que vienes usando)
    summary = {
        "mode": mode,
        "threshold": float(thr),
        "adx1d_min": float(args.adx1d_min),
        "adx4_min": float(args.adx4_min),

        "net90": float(m90.net),
        "pf90": float(m90.pf),
        "score90": float(m90.score),
        "mdd90": float(m90.mdd),
        "trades90": float(m90.trades),
        "win90": float(m90.winrate),

        "net180": float(m180.net),
        "pf180": float(m180.pf),
        "score180": float(m180.score),
        "mdd180": float(m180.mdd),
        "trades180": float(m180.trades),
        "win180": float(m180.winrate),

        # Holdout: no se estima aquí (0s para compatibilidad)
        "holdout_net": 0.0,
        "holdout_pf": 0.0,
        "holdout_score": 0.0,
        "holdout_mdd": -0.0,
        "holdout_trades": 0.0,

        "timestamp_utc": utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary_path": os.path.abspath(sum_path),
        # Trazabilidad de flags
        "use_sentiment": bool(args.use_sentiment and not args.no_sentiment),
        "fg_csv": args.fg_csv,
        "funding_csv": args.funding_csv,
        "fg_long_min": float(args.fg_long_min),
        "fg_short_max": float(args.fg_short_max),
        "funding_bias": float(args.funding_bias),
        "pt_mode": args.pt_mode,
        "sl_pct": float(args.sl_pct),
        "tp_pct": float(args.tp_pct),
        "trail_pct": float(args.trail_pct),
        "adx_daily_source": args.adx_daily_source,
        "adx1d_len": int(args.adx1d_len),
    }
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Summary: {sum_path}")
    print(f"✅ Walkforward: {wf_path}\n")

    print(
        "Pick snapshot → "
        f"mode={mode} thr={thr:.2f} | "
        f"score180={m180.score:.3f} net180={m180.net:.3f} pf180={m180.pf:.3f} mdd180={m180.mdd:.3f}"
    )


if __name__ == "__main__":
    main()