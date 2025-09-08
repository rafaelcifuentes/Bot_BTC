#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corazón ❤️ — Runner de señales con sentimiento (Fear&Greed + Funding)
"""

import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import pandas_ta as ta

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None
    print("⚠️ ccxt no disponible. Instala con: pip install ccxt", file=sys.stderr)


# --------------------------- Utilidades CSV flex ---------------------------

DATE_CANDIDATES = ["timestamp", "time", "date", "datetime", "dt", "created_at"]

def _parse_datetime_flex(series: pd.Series) -> pd.DatetimeIndex:
    s = series.copy()
    if np.issubdtype(s.dtype, np.number):
        vals = pd.to_numeric(s, errors="coerce")
        if np.nanmax(vals.values) > 1e12:
            ts = pd.to_datetime(vals, unit="ms", utc=True)
        elif np.nanmax(vals.values) > 1e9:
            ts = pd.to_datetime(vals, unit="s", utc=True)
        else:
            ts = pd.to_datetime(vals, errors="coerce", utc=True)
    else:
        ts = pd.to_datetime(s, errors="coerce", utc=True)
    return pd.DatetimeIndex(ts).tz_convert(None)

def _read_csv_flex(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    cols_lower = {c.lower(): c for c in df.columns}
    date_col = None
    for cand in DATE_CANDIDATES:
        if cand in cols_lower:
            date_col = cols_lower[cand]; break
    if date_col is None:
        date_col = df.columns[0]
    ts = _parse_datetime_flex(df[date_col])
    df = df.copy()
    df["__ts__"] = ts
    df = df.dropna(subset=["__ts__"]).set_index("__ts__").sort_index()
    return df


# --------------------------- Descarga OHLCV ---------------------------

def _fetch_ccxt(
    exchange_id: str = "binanceus",
    symbol_ccxt: Optional[str] = None,
    timeframe: str = "4h",
    days: int = 1460,
    freeze_end: Optional[pd.Timestamp] = None,
    max_bars: Optional[int] = None,
) -> pd.DataFrame:
    """
    OHLCV paginado desde ccxt (límite ~1000). Índice naive (sin tz).
    """
    if ccxt is None:
        return pd.DataFrame()

    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    if symbol_ccxt is None:
        symbol_ccxt = "BTC/USD" if exchange_id.lower() == "binanceus" else "BTC/USDT"

    if freeze_end is None:
        fe_utc = pd.Timestamp.now(tz="UTC")
    else:
        fe = pd.Timestamp(freeze_end)
        fe_utc = fe.tz_localize("UTC") if fe.tzinfo is None else fe.tz_convert("UTC")

    end_ms = int(fe_utc.timestamp() * 1000)
    start_utc = fe_utc - pd.Timedelta(days=days)
    since = int(start_utc.timestamp() * 1000)

    all_rows: List[List[float]] = []
    limit_req = 1000

    while True:
        ohlcv = ex.fetch_ohlcv(symbol_ccxt, timeframe=timeframe, since=since, limit=limit_req)
        if not ohlcv:
            break
        all_rows += ohlcv
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < limit_req:
            break
        time.sleep(ex.rateLimit / 1000.0)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    ts_idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)).tz_convert(None)
    df = df.drop(columns=["timestamp"])
    df.index = ts_idx
    df = df.sort_index()

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    if freeze_end is not None:
        fe = pd.Timestamp(freeze_end)
        fe_naive = fe if fe.tzinfo is None else fe.tz_convert("UTC").tz_localize(None)
        df = df.loc[:fe_naive]

    if max_bars is not None and len(df) > max_bars:
        df = df.iloc[-max_bars:]

    return df


# --------------------------- Cargadores Sentimiento/Funding ---------------------------

def load_fear_greed(path: str) -> pd.DataFrame:
    try:
        df = _read_csv_flex(path)
    except Exception as e:
        print(f"⚠️  No pude leer Fear&Greed ({path}): {e}. Uso serie neutra (0).")
        return pd.DataFrame({"fg_norm": [0.0], "fg_ema3": [0.0]}, index=[pd.Timestamp("1970-01-01")])

    cols_lower = {c.lower(): c for c in df.columns}
    val_col = None
    for cand in ["fg_norm", "fgraw", "fg_raw", "value", "score", "fear_greed", "fg"]:
        if cand in cols_lower:
            val_col = cols_lower[cand]; break
    if val_col is None:
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if num_cols:
            val_col = num_cols[0]
        else:
            print(f"⚠️  Fear&Greed sin columna de valor reconocible. Uso 0.")
            return pd.DataFrame({"fg_norm": [0.0], "fg_ema3": [0.0]}, index=[pd.Timestamp("1970-01-01")])

    vals = pd.to_numeric(df[val_col], errors="coerce")
    if vals.max() > 1.5 or vals.min() < -1.5:
        vals = (vals - 50.0) / 50.0  # normaliza 0..100 → -1..+1
    fg = pd.DataFrame({"fg_norm": vals}, index=df.index).sort_index()
    fg = fg.resample("4h").ffill()
    fg["fg_ema3"] = fg["fg_norm"].ewm(span=3, adjust=False).mean()
    fg = fg.clip(lower=-1, upper=1)
    return fg

def load_funding(path: str, symbol_like: str = "BTC") -> pd.DataFrame:
    try:
        df = _read_csv_flex(path)
    except Exception as e:
        print(f"⚠️  No pude leer Funding ({path}): {e}. Uso serie neutra (0).")
        return pd.DataFrame({"funding": [0.0], "fund_24h": [0.0]}, index=[pd.Timestamp("1970-01-01")])

    sym_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ["symbol", "pair", "ticker", "market", "instrument", "name"]:
            sym_col = c; break
    if sym_col is not None:
        df = df[df[sym_col].astype(str).str.contains(symbol_like, case=False, na=False)]

    val_col = None
    lower = {c.lower(): c for c in df.columns}
    for cand in ["funding", "funding_rate", "rate", "fundrate", "fundingrate", "predicted_rate", "predictedfundingrate", "funding_8h"]:
        if cand in lower:
            val_col = lower[cand]; break
    if val_col is None:
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if num_cols:
            val_col = num_cols[0]
        else:
            print(f"⚠️  Funding sin columna de valor reconocible. Uso 0.")
            return pd.DataFrame({"funding": [0.0], "fund_24h": [0.0]}, index=[pd.Timestamp("1970-01-01")])

    vals = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)
    if vals.abs().median() > 0.5:
        vals = vals / 100.0  # si estaba en %

    fu = pd.DataFrame({"funding": vals}, index=df.index).sort_index()
    fu = fu.resample("4h").ffill()
    fu["fund_24h"] = fu["funding"].rolling(6, min_periods=1).mean()
    return fu


# --------------------------- Features, ADX, ML ---------------------------

def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema12"] = ta.ema(out["close"], length=12)
    out["ema26"] = ta.ema(out["close"], length=26)
    out["rsi14"] = ta.rsi(out["close"], length=14)
    atr = ta.atr(out["high"], out["low"], out["close"], length=14)
    out["atr14"] = atr
    out["atrp"] = (atr / out["close"]).clip(lower=0, upper=1)
    out["ret1"] = out["close"].pct_change()
    out["ema_slope"] = out["ema12"] / out["ema26"] - 1.0
    out["volu_z"] = (out["volume"] - out["volume"].rolling(50).mean()) / (out["volume"].rolling(50).std() + 1e-9)
    return out

def add_adx_blocks(df_4h: pd.DataFrame, adx1d_len: int = 14) -> pd.DataFrame:
    out = df_4h.copy()
    adx4 = ta.adx(out["high"], out["low"], out["close"], length=14)
    out["adx4"] = adx4["ADX_14"]

    d = (
        out[["open", "high", "low", "close"]]
        .resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    adx1 = ta.adx(d["high"], d["low"], d["close"], length=adx1d_len)
    d["adx1d"] = adx1[f"ADX_{adx1d_len}"]
    out["adx1d"] = d["adx1d"].reindex(out.index).ffill()
    return out

def build_ml_xy(df: pd.DataFrame, include_cols: List[str], shift_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[include_cols].copy()
    if shift_features:
        X = X.shift(1)  # evita leakage
    y = (df["close"].shift(-1) > df["close"]).astype(int)
    m = X.dropna().index.intersection(y.dropna().index)
    return X.loc[m], y.loc[m]

def train_rf(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ))
    ])
    pipe.fit(X, y)
    return pipe


# --------------------------- Backtest SL/TP/Trailing (%) ---------------------------

@dataclass
class ExecParams:
    threshold: float = 0.59
    fee_pct: float = 0.0004
    slip_pct: float = 0.0005
    pt_mode: str = "pct"
    sl_pct: float = 0.02
    tp_pct: float = 0.10
    trail_pct: float = 0.015
    capital: float = 10000.0
    risk_perc: float = 0.0075

def gate_env(row, fg_long_min, fg_short_max, funding_bias, adx1d_min, adx4_min, no_gates: bool):
    if no_gates:
        return True, True
    if not (np.isfinite(row.get("adx1d", np.nan)) and np.isfinite(row.get("adx4", np.nan))):
        return False, False
    if row["adx1d"] < adx1d_min or row["adx4"] < adx4_min:
        return False, False
    ok_long = (row.get("fg_norm", 0.0) <= fg_long_min) and (row.get("fund_24h", 0.0) <= funding_bias)
    ok_short = (row.get("fg_norm", 0.0) >= fg_short_max) and (row.get("fund_24h", 0.0) >= funding_bias)
    return ok_long, ok_short

def simulate_trades(df: pd.DataFrame, prob_up: pd.Series, params: ExecParams,
                    fg_long_min: float, fg_short_max: float, funding_bias: float,
                    adx1d_min: float, adx4_min: float, no_gates: bool) -> pd.DataFrame:
    equity = params.capital
    trades = []
    in_pos = False
    side = 0
    qty = 0.0
    entry = 0.0
    trail_price = np.nan

    idx = df.index
    for i in range(len(idx) - 1):
        t = idx[i]
        nxt = idx[i + 1]
        row = df.loc[t]
        nxt_row = df.loc[nxt]
        p_up = float(prob_up.loc[t]) if t in prob_up.index else np.nan
        if not np.isfinite(p_up):
            continue

        ok_long, ok_short = gate_env(row, fg_long_min, fg_short_max, funding_bias, adx1d_min, adx4_min, no_gates)

        if in_pos:
            hi = float(nxt_row["high"]); lo = float(nxt_row["low"]); close = float(nxt_row["close"])

            def exit_p(px):
                px_eff = px * (1 - params.slip_pct) if side == +1 else px * (1 + params.slip_pct)
                fee = px_eff * abs(qty) * params.fee_pct
                return px_eff, fee

            pnl = None
            if side == +1:
                sl = entry * (1 - params.sl_pct)
                tp = entry * (1 + params.tp_pct)
                if np.isfinite(params.trail_pct) and params.trail_pct > 0:
                    trail_price = max(trail_price, hi)
                    trail_stop = trail_price * (1 - params.trail_pct)
                else:
                    trail_stop = -np.inf
                if lo <= sl:
                    px, fee = exit_p(sl); pnl = (px - entry) * qty - fee
                elif hi >= tp:
                    px, fee = exit_p(tp); pnl = (px - entry) * qty - fee
                elif close <= trail_stop:
                    px, fee = exit_p(close); pnl = (px - entry) * qty - fee
            else:
                sl = entry * (1 + params.sl_pct)
                tp = entry * (1 - params.tp_pct)
                if np.isfinite(params.trail_pct) and params.trail_pct > 0:
                    trail_price = min(trail_price, lo)
                    trail_stop = trail_price * (1 + params.trail_pct)
                else:
                    trail_stop = +np.inf
                if hi >= sl:
                    px, fee = exit_p(sl); pnl = (entry - px) * abs(qty) - fee
                elif lo <= tp:
                    px, fee = exit_p(tp); pnl = (entry - px) * abs(qty) - fee
                elif close >= trail_stop:
                    px, fee = exit_p(close); pnl = (entry - px) * abs(qty) - fee

            if pnl is not None:
                equity += pnl
                trades.append({"exit_time": nxt, "side": side, "entry": entry, "exit": px,
                               "qty": qty, "pnl": pnl, "equity": equity})
                in_pos = False
                side = 0; qty = 0.0; entry = 0.0; trail_price = np.nan

        if not in_pos:
            open_px = float(nxt_row["open"])

            def entry_p(px, s):
                px_eff = px * (1 + params.slip_pct) if s == +1 else px * (1 - params.slip_pct)
                fee = px_eff * params.fee_pct
                return px_eff, fee

            p_dn = 1.0 - p_up
            cand = []
            if ok_long and p_up >= params.threshold:
                cand.append((+1, p_up))
            if ok_short and p_dn >= params.threshold:
                cand.append((-1, p_dn))
            if cand:
                cand.sort(key=lambda x: x[1], reverse=True)
                sel_side, _ = cand[0]
                px_eff, _ = entry_p(open_px, sel_side)

                sl_dist = params.sl_pct * px_eff
                if sl_dist <= 0:
                    continue
                risk_cash = params.capital * params.risk_perc
                q = max(risk_cash / sl_dist, 0.0)
                if q <= 0:
                    continue

                in_pos = True
                side = sel_side
                qty = q
                entry = px_eff
                trail_price = entry

    tr = pd.DataFrame(trades)
    if tr.empty:
        tr = pd.DataFrame(columns=["exit_time", "side", "entry", "exit", "qty", "pnl", "equity"])
    tr = tr.set_index("exit_time").sort_index()
    return tr

def metrics_from_trades(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return dict(net=0.0, pf=np.nan, win_rate=0.0, trades=0, mdd=0.0, sortino=np.nan)

    wins = tr[tr["pnl"] > 0]["pnl"].sum()
    losses = tr[tr["pnl"] < 0]["pnl"].sum()
    pf = np.inf if losses == 0 else (wins / abs(losses))
    wr = (tr["pnl"] > 0).mean() * 100.0

    if "equity" in tr.columns and tr["equity"].notna().any():
        eq = tr["equity"].ffill()
    else:
        eq = tr["pnl"].cumsum()
    peak = eq.cummax()
    dd = (eq - peak) / peak.replace(0, np.nan)
    mdd = dd.min() if dd.notna().any() else 0.0

    rets = tr["pnl"]
    downside = rets[rets < 0]
    sortino = (rets.mean() / (downside.std() + 1e-9)) if len(downside) else np.inf

    return dict(net=float(tr["pnl"].sum()),
                pf=float(pf),
                win_rate=float(wr),
                trades=int(len(tr)),
                mdd=float(mdd),
                sortino=float(sortino))

def slice_last_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    start = df.index.max() - pd.Timedelta(days=days)
    return df.loc[df.index > start]


# --------------------------- CLI / Main ---------------------------

def parse_grid(s: str) -> List[float]:
    s = s.strip()
    if ":" in s:
        a, b, step = s.split(":")
        a = float(a); b = float(b); step = float(step)
        n = int(round((b - a) / step)) + 1
        return [round(a + i * step, 10) for i in range(n)]
    else:
        return [float(x) for x in s.split(",") if x.strip()]

def add_blocks_and_predict(src: pd.DataFrame, fg: pd.DataFrame, fu: pd.DataFrame, args):
    base = add_ta_features(src)
    base = add_adx_blocks(base, adx1d_len=args.adx1d_len)
    base = base.join(fg[["fg_norm", "fg_ema3"]], how="left").join(fu[["funding", "fund_24h"]], how="left")
    base = base.dropna()

    feat_cols = ["ema12", "ema26", "rsi14", "atrp", "ret1", "ema_slope", "volu_z",
                 "adx4", "adx1d", "fg_norm", "fg_ema3", "fund_24h"]
    X, y = build_ml_xy(base, feat_cols, shift_features=True)
    model = train_rf(X, y)

    X_pred = base[feat_cols].shift(1)
    prob_idx = X_pred.dropna().index
    prob_up = pd.Series(np.nan, index=base.index)
    if len(prob_idx):
        prob_up.loc[prob_idx] = model.predict_proba(X_pred.loc[prob_idx])[:, 1]
    return base, prob_up

def env_diagnostics(base: pd.DataFrame, prob_up: pd.Series, args):
    # % de barras que pasan cada gate y % de probs ≥ threshold
    adx_ok = ((base["adx1d"] >= args.adx1d_min) & (base["adx4"] >= args.adx4_min)).mean() if len(base) else 0.0
    long_ok = ((base["fg_norm"] <= args.fg_long_min) & (base["fund_24h"] <= args.funding_bias)).mean() if len(base) else 0.0
    short_ok = ((base["fg_norm"] >= args.fg_short_max) & (base["fund_24h"] >= args.funding_bias)).mean() if len(base) else 0.0
    prob_ok = (prob_up >= args.threshold).mean() if prob_up.notna().any() else 0.0
    print(f"🔎 Gates → ADX ok: {adx_ok:.1%} | long gate: {long_ok:.1%} | short gate: {short_ok:.1%} | prob≥th: {prob_ok:.1%}")

def run_once(src, fg, fu, args, threshold: float):
    base, prob_up = add_blocks_and_predict(src, fg, fu, args)
    env_diagnostics(base, prob_up, args)

    params = ExecParams(
        threshold=threshold,
        fee_pct=args.fee_pct,
        slip_pct=args.slip_pct,
        sl_pct=args.sl_pct,
        tp_pct=args.tp_pct,
        trail_pct=args.trail_pct,
        capital=args.capital,
        risk_perc=args.risk_perc
    )

    tr = simulate_trades(base, prob_up, params,
                         fg_long_min=args.fg_long_min,
                         fg_short_max=args.fg_short_max,
                         funding_bias=args.funding_bias,
                         adx1d_min=args.adx1d_min,
                         adx4_min=args.adx4_min,
                         no_gates=args.no_gates)

    def m_for(d):
        tr_d = tr.loc[slice_last_days(tr, d).index] if not tr.empty else tr
        return metrics_from_trades(tr_d)

    m30, m60, m90 = m_for(30), m_for(60), m_for(90)
    enriched = pd.DataFrame.from_dict({30: m30, 60: m60, 90: m90}, orient="index")
    enriched.index.name = "days"
    return enriched, tr

def main():
    p = argparse.ArgumentParser(description="Corazón ❤️ — Sentiment runner")
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default="4h")
    p.add_argument("--period", default="1460d")
    p.add_argument("--freeze_end", default=None, help='YYYY-MM-DD HH:MM')
    p.add_argument("--max_bars", type=int, default=None)

    p.add_argument("--fg_csv", default="./data/sentiment/fear_greed.csv")
    p.add_argument("--funding_csv", default="./data/sentiment/funding_rates.csv")
    p.add_argument("--fg_long_min", type=float, default=-0.18)
    p.add_argument("--fg_short_max", type=float, default=0.18)
    p.add_argument("--funding_bias", type=float, default=0.01)

    p.add_argument("--adx1d_len", type=int, default=14)
    p.add_argument("--adx1d_min", type=float, default=30.0)
    p.add_argument("--adx4_min", type=float, default=18.0)

    p.add_argument("--threshold", type=float, default=0.59)
    p.add_argument("--sweep_threshold", default=None, help="Ej: '0.56,0.58,0.60' o '0.56:0.64:0.01'")

    p.add_argument("--pt_mode", default="pct", choices=["pct"])
    p.add_argument("--sl_pct", type=float, default=0.02)
    p.add_argument("--tp_pct", type=float, default=0.10)
    p.add_argument("--trail_pct", type=float, default=0.015)

    p.add_argument("--fee_pct", type=float, default=0.0004)
    p.add_argument("--slip_pct", type=float, default=0.0005)
    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--risk_perc", type=float, default=0.0075)

    p.add_argument("--out_csv", default="reports/corazon_metrics.csv")

    # Compat / control
    p.add_argument("--primary_only", action="store_true")
    p.add_argument("--repro_lock", action="store_true")
    p.add_argument("--use_sentiment", action="store_true")
    p.add_argument("--adx_daily_source", default="resample")
    p.add_argument("--no_gates", action="store_true", help="Desactiva filtros de entorno (ADX/F&G/Funding)")

    args = p.parse_args()

    exchange_id = os.getenv("EXCHANGE", "binanceus").lower()
    symbol_ccxt = args.symbol or ("BTC/USD" if exchange_id == "binanceus" else "BTC/USDT")

    if isinstance(args.period, str) and args.period.endswith("d"):
        try:
            days = int(args.period[:-1])
        except Exception:
            days = 1460
    else:
        try:
            days = int(args.period)
        except Exception:
            days = 1460

    freeze = pd.to_datetime(args.freeze_end) if args.freeze_end else None

    print(f"⚙️  EXCHANGE={exchange_id}  symbol={symbol_ccxt}  timeframe={args.timeframe}")
    src = _fetch_ccxt(exchange_id, symbol_ccxt, timeframe=args.timeframe, days=days,
                      freeze_end=freeze, max_bars=args.max_bars)
    if src.empty:
        print("❌ No se obtuvieron velas de ccxt.")
        sys.exit(1)
    print(f"Rango bruto/naive: {src.index.min()} → {src.index.max()}  | velas={len(src)}")

    fg = load_fear_greed(args.fg_csv)
    fu = load_funding(args.funding_csv, "BTC")

    # Alineación temporal común
    common_start = max(
        src.index.min(),
        fg.index.min() if not fg.empty else src.index.min(),
        fu.index.min() if not fu.empty else src.index.min()
    )
    src = src.loc[src.index >= common_start]
    print(f"Rango alineado: {src.index.min()} → {src.index.max()}  | velas={len(src)}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    if args.sweep_threshold:
        grid = parse_grid(args.sweep_threshold)
        rows = []
        for th in grid:
            enriched, tr = run_once(src, fg, fu, args, threshold=th)
            rows.append({
                "threshold": th,
                "pf_30d": enriched.loc[30, "pf"], "wr_30d": enriched.loc[30, "win_rate"], "mdd_30d": enriched.loc[30, "mdd"], "net_30d": enriched.loc[30, "net"], "trades_30d": enriched.loc[30, "trades"],
                "pf_60d": enriched.loc[60, "pf"], "wr_60d": enriched.loc[60, "win_rate"], "mdd_60d": enriched.loc[60, "mdd"], "net_60d": enriched.loc[60, "net"], "trades_60d": enriched.loc[60, "trades"],
                "pf_90d": enriched.loc[90, "pf"], "wr_90d": enriched.loc[90, "win_rate"], "mdd_90d": enriched.loc[90, "mdd"], "net_90d": enriched.loc[90, "net"], "trades_90d": enriched.loc[90, "trades"],
            })
        sweep = pd.DataFrame(rows)
        ts = time.strftime("%Y%m%d_%H%M%S")
        sweep_path = f"reports/corazon_sweep_threshold_{ts}.csv"
        os.makedirs(os.path.dirname(sweep_path), exist_ok=True)
        sweep.to_csv(sweep_path, index=False)
        print("\n== Sweep de threshold (Corazón / SAFE) ==")
        print("Grid:", grid)
        print("\n== Resultado sweep ==")
        print(sweep.to_string(index=False))
        print("CSV sweep ->", sweep_path)

        enriched, _ = run_once(src, fg, fu, args, threshold=grid[-1])
        enriched["roi_pct"] = 100 * enriched["net"] / float(args.capital)
        enriched.to_csv(args.out_csv, index=True)
        enriched.to_csv(args.out_csv.replace(".csv", "_plus.csv"), index=True)
        print("CSV ->", args.out_csv)
        print("CSV+ ->", args.out_csv.replace(".csv", "_plus.csv"))
        return

    enriched, tr = run_once(src, fg, fu, args, threshold=args.threshold)
    print("\n== Variante Corazón (SAFE, shift(1)) ==")
    print(enriched[["net", "pf", "win_rate", "trades", "mdd"]].to_string())

    enriched["roi_pct"] = 100 * enriched["net"] / float(args.capital)
    enriched.to_csv(args.out_csv, index=True)
    enriched.to_csv(args.out_csv.replace(".csv", "_plus.csv"), index=True)
    print("CSV ->", args.out_csv)
    print("CSV+ ->", args.out_csv.replace(".csv", "_plus.csv"))

    if not tr.empty:
        ts = time.strftime("%Y%m%d_%H%M%S")
        tr_path = f"reports/corazon_trades_{ts}.csv"
        tr.to_csv(tr_path)
        print("CSV trades ->", tr_path)


if __name__ == "__main__":
    main()