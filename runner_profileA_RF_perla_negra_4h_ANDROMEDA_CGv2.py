#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runner_profileA_RF_perla_negra_4h_ANDROMEDA_CGv2.py (experimental)

Andrómeda RF 4h con Conflict-Guard v2 (experimental):
- Multi-source data loader (ccxt + yfinance fallback) con snapshot y flags de reproducción
- RF señal (locked params) + filtros ADX 4h / ADX 1d
- Conflict-Guard v2 (cooldown, hysteresis, margin, trend-check) para evitar whipsaws
- Backtest motor con ATR/PCT SL-TP + trailing, parciales, costes y slippage
- Baseline 90/180d, WF 6×90, Holdout 75d, Stress table, Micro-grid, Score, Sortino
- JSON robusto (json_default) para evitar crash de Timestamp no serializable

NOTA: Script experimental; no sustituye tu V2 estable, ni hace autolock por defecto.
"""

import os
import sys
import json
import time
import math
import argparse
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas_ta as ta

try:
    import ccxt
except Exception:
    ccxt = None

try:
    import yfinance as yf
except Exception:
    yf = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ========================= Utiles JSON & métricas (drop-in) =========================

def json_default(o):
    import numpy as _np
    import pandas as _pd
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_pd.Timestamp, dt.datetime)):
        return o.isoformat()
    if isinstance(o, (_pd.Series,)):
        return json.loads(o.to_json(date_format="iso"))
    if isinstance(o, (_pd.DataFrame,)):
        return json.loads(o.to_json(orient="records", date_format="iso"))
    return str(o)


def sortino_ratio(returns, rf=0.0):
    r = np.asarray(returns, dtype=float) - rf
    if r.size == 0:
        return float("nan")
    downside = r[r < 0]
    dd = downside.std(ddof=0)
    if dd == 0:
        return float("inf")
    return float(r.mean() / dd)


# ========================= Loader =========================
EXCHS = [
    ("binance", "BINANCE"), ("binanceus", "BINANCEUS"), ("kraken", "KRAKEN"),
    ("bybit", "BYBIT"), ("kucoin", "KUCOIN"), ("okx", "OKX")
]


def _load_ccxt(exchange_id: str, symbol: str, timeframe: str, since_ms: int | None, limit=5000):
    ex = getattr(ccxt, exchange_id)()
    ex.load_markets()
    all_ohlc = []
    since = since_ms
    while True:
        ohlc = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not ohlc:
            break
        all_ohlc.extend(ohlc)
        if len(ohlc) < 1000:
            break
        since = ohlc[-1][0] + 1
        if len(all_ohlc) >= limit:
            break
        time.sleep(0.05)
    if not all_ohlc:
        return pd.DataFrame()
    df = pd.DataFrame(all_ohlc, columns=["ts","open","high","low","close","volume"]) \
            .set_index(pd.to_datetime([x[0] for x in all_ohlc], unit="ms"))
    return df.drop(columns=["ts"]).sort_index()


def _load_yf(symbol="BTC-USD", period="730d", interval="4h"):
    if yf is None:
        return pd.DataFrame()
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df.dropna().sort_index()


def load_4h(symbol="BTC/USDT", primary_only=False, freeze_end: str | None = None, repro_lock=False):
    print("🔄 Downloading 4h data (ccxt/Binance + fallbacks)…")
    print(f"Flags → primary_only={primary_only}, freeze_end={freeze_end}, date_split=None, repro_lock={repro_lock}")
    end = None
    if freeze_end:
        end = pd.Timestamp(freeze_end).tz_localize(None)
    since = int(((end or pd.Timestamp.utcnow().tz_localize(None)) - pd.Timedelta(days=750)).timestamp()*1000)
    sources = []
    df = pd.DataFrame()
    if ccxt is not None:
        ids = EXCHS[:1] if primary_only else EXCHS
        for ex_id, label in ids:
            try:
                d = _load_ccxt(ex_id, symbol, "4h", since)
                if not d.empty:
                    df = d
                    sources.append(f"{ex_id}: ok")
                    break
                else:
                    sources.append(f"{ex_id}: empty")
            except Exception:
                sources.append(f"{ex_id}: err")
    if df.empty:
        d = _load_yf("BTC-USD", period="750d", interval="4h")
        if not d.empty:
            df = d
            sources.append("yfinance: ok")
    print(f"Sources: {sources if sources else ['none']}")
    if df.empty:
        sys.exit("❌ No data")
    return df.loc[:, ["open","high","low","close","volume"]].copy()


def load_daily_for_adx(days=900):
    dfd = _load_yf("BTC-USD", period=f"{days}d", interval="1d")
    return dfd


# ========================= Features & target =========================
FEATURES = ["ema_fast","ema_slow","rsi","atr","adx4h","adx_daily"]


def add_features(df4h: pd.DataFrame, dfd: pd.DataFrame, p):
    df = df4h.copy()
    df["ema_fast"] = ta.ema(df["close"], length=p.get("ema_fast", 12))
    df["ema_slow"] = ta.ema(df["close"], length=p.get("ema_slow", 26))
    df["rsi"] = ta.rsi(df["close"], length=p.get("rsi_len", 14))
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=p.get("atr_len", 14))
    adx4 = ta.adx(df["high"], df["low"], df["close"], length=p.get("adx_len", 14))
    if adx4 is not None and not adx4.empty:
        col = [c for c in adx4.columns if c.lower().startswith("adx") or "ADX_" in c]
        df["adx4h"] = adx4[col[0]] if col else 0.0
    else:
        df["adx4h"] = 0.0
    # daily ADX
    adx_d = ta.adx(dfd["high"], dfd["low"], dfd["close"], length=p.get("adx_daily_len", 14)) if not dfd.empty else None
    if adx_d is not None and not adx_d.empty:
        col = [c for c in adx_d.columns if c.lower().startswith("adx") or "ADX_" in c]
        dfd2 = dfd.copy()
        dfd2["adx_daily"] = adx_d[col[0]] if col else np.nan
        df["adx_daily"] = dfd2["adx_daily"].reindex(df.index, method="ffill")
    else:
        df["adx_daily"] = 0.0
    return df.dropna()


def make_target(df: pd.DataFrame, shift=6, pct=0.007):
    out = df.copy()
    up = out["close"].shift(-shift) > out["close"] * (1+pct)
    dn = out["close"].shift(-shift) < out["close"] * (1-pct)
    out["target"] = np.where(up, 1, np.where(dn, -1, 0))
    return out.dropna()


# ========================= Model =========================

def get_rf_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame, rf_params: dict):
    Xtr, ytr = train_df[FEATURES], train_df["target"]
    Xte = test_df[FEATURES]
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 50),
            max_depth=rf_params.get("max_depth", 10),
            min_samples_split=rf_params.get("min_samples_split", 2),
            class_weight="balanced", random_state=42, n_jobs=-1,
        ))
    ])
    pipe.fit(Xtr, ytr)
    probs = pipe.predict_proba(Xte)
    dfp = test_df.copy()
    classes = pipe.classes_
    idx_up = np.where(classes == 1)[0][0] if 1 in classes else 0
    idx_dn = np.where(classes == -1)[0][0] if -1 in classes else 0
    dfp["prob_up"] = probs[:, idx_up]
    dfp["prob_down"] = probs[:, idx_dn]
    return dfp


# ========================= Conflict-Guard v2 =========================
@dataclass
class CG2:
    enabled: bool = True
    cooldown: int = 3             # bars to wait before flipping
    hysteresis: float = 0.02      # min distance from last entry price to flip
    margin: float = 0.05          # trend margin for alignment
    trend_check: bool = True      # require ema_slow slope alignment

    last_dir: int = 0             # -1, 0, +1
    last_entry: float = 0.0
    bars_since: int = 9999

    def allow_open(self, r) -> bool:
        if not self.enabled:
            return True
        if self.last_dir == 0:
            return True
        # block flip too soon
        if self.bars_since < self.cooldown:
            return False
        # block flip if price hasn't moved enough relative to last entry
        px = r.open
        moved = abs(px / max(1e-12, self.last_entry) - 1.0)
        return moved >= self.hysteresis

    def trend_ok(self, d, r) -> bool:
        if not self.enabled or not self.trend_check:
            return True
        # simple trend by ema_slow slope last 5 bars
        slope = float(r.ema_slow - getattr(r, "ema_slow_prev5", r.ema_slow)) / max(1e-9, r.ema_slow)
        if d == 1:
            return slope >= -self.margin/100  # pretty permissive in % terms
        else:
            return slope <= self.margin/100

    def on_open(self, d, entry):
        self.last_dir = d
        self.last_entry = entry
        self.bars_since = 0

    def on_bar(self):
        self.bars_since += 1


# ========================= Backtest engine =========================
@dataclass
class EngineParams:
    threshold: float = 0.55
    risk_perc: float = 0.01
    cost: float = 0.0002
    slip: float = 0.0001
    pt_mode: str = "atr"  # or "pct"
    sl_atr_mul: float = 2.5
    tp1_atr_mul: float = 1.6
    tp2_atr_mul: float = 5.0
    trail_mul: float = 1.0
    sl_pct: float = 0.025
    tp_pct: float = 0.05
    trail_pct: float = 0.01
    partial_pct: float = 0.5
    adx4_min: float = 10
    adx1d_min: float = 10


def _pt_levels(row, d, ep: EngineParams):
    e = row.open * (1 + ep.slip * d)
    if ep.pt_mode == "atr":
        a = row.atr
        sl = e - d * a * ep.sl_atr_mul
        t1 = e + d * a * ep.tp1_atr_mul
        t2 = e + d * a * ep.tp2_atr_mul
        tr = ep.trail_mul * a
    else:
        sl = e * (1 - d * ep.sl_pct)
        t1 = e * (1 + d * ep.tp_pct)
        t2 = e * (1 + d * ep.tp_pct * 2.0)
        tr = ep.trail_pct * e
    return e, sl, t1, t2, tr


def backtest_once(df: pd.DataFrame, ep: EngineParams, cg: CG2, days=180, start_equity=10_000.0):
    df_t = df.tail(days*6).copy()
    # add ema_slow prev5 for trend slope
    df_t["ema_slow_prev5"] = df_t["ema_slow"].shift(5)

    equity = start_equity
    peak = equity
    mdd = 0.0

    op = None
    trades = []
    eq_changes = []

    for _, r in df_t.iterrows():
        cg.on_bar()
        long_sig = (r.get("prob_up",0) > ep.threshold) and (r.adx4h >= ep.adx4_min) and (r.adx_daily >= ep.adx1d_min)
        short_sig = (r.get("prob_down",0) > ep.threshold) and (r.adx4h >= ep.adx4_min) and (r.adx_daily >= ep.adx1d_min)

        if op is None:
            d = 1 if long_sig else (-1 if short_sig else 0)
            if d != 0 and cg.allow_open(r) and cg.trend_ok(d, r):
                e, s, t1, t2, tr = _pt_levels(r, d, ep)
                if (d*(s - e)) < 0:
                    size = (equity * ep.risk_perc) / abs(e - s)
                    equity -= e * size * ep.cost
                    op = {"d":d, "e":e, "s":s, "t1":t1, "t2":t2, "tr":tr, "sz":size, "hit1":False, "hh": e, "ll": e}
                    cg.on_open(d, e)
        else:
            d, e = op["d"], op["e"]
            op["hh"] = max(op["hh"], r.high)
            op["ll"] = min(op["ll"], r.low)
            # trail
            if d == 1:
                new_s = op["hh"] - op["tr"]
                if new_s > op["s"]:
                    op["s"] = new_s
            else:
                new_s = op["ll"] + op["tr"]
                if new_s < op["s"]:
                    op["s"] = new_s
            exit_p = None
            # TP1 partial
            if not op["hit1"] and ((d==1 and r.high >= op["t1"]) or (d==-1 and r.low <= op["t1"])):
                pnl = (op["t1"] - e) * d * (op["sz"] * ep.partial_pct)
                equity += pnl
                trades.append(pnl)
                op["sz"] *= (1 - ep.partial_pct)
                op["hit1"] = True
            # TP2
            if (d==1 and r.high >= op["t2"]) or (d==-1 and r.low <= op["t2"]):
                exit_p = op["t2"]
            # Stop
            if (d==1 and r.low <= op["s"]) or (d==-1 and r.high >= op["s"]):
                exit_p = op["s"]
            if exit_p is not None:
                pnl = (exit_p - e) * d * op["sz"] - exit_p * op["sz"] * ep.cost
                equity += pnl
                trades.append(pnl)
                op = None

        # equity bookkeeping
        peak = max(peak, equity)
        mdd = max(mdd, peak - equity)
        eq_changes.append(0 if len(eq_changes)==0 else (equity - (equity - trades[-1] if trades else equity)))

    arr = np.asarray(trades, dtype=float)
    net = float(arr.sum()) if arr.size else 0.0
    wins = float(arr[arr>0].sum())
    losses = float(-arr[arr<0].sum())
    pf = (wins / losses) if losses > 0 else (np.inf if wins>0 else 0.0)
    wr = ( (arr>0).sum() / arr.size * 100.0 ) if arr.size else 0.0
    mdd_pct = (mdd / start_equity * 100.0)

    # Sortino from synthetic per-trade equity deltas (approx)
    er = np.asarray(arr) / start_equity
    srt = sortino_ratio(er) if er.size else float("nan")

    return {
        "net": net,
        "pf": float(pf if np.isfinite(pf) else 0.0),
        "win_rate": float(wr),
        "trades": int(arr.size),
        "mdd": float(mdd_pct),
        "score": float(net / (mdd_pct + 1e-9)),
        "sortino": float(srt),
    }


# ========================= WF / Holdout / Micro-grid =========================

def walk_forward(df_all, rf_params, ep, cg, folds=6, fold_days=90, train_days=270):
    rows = []
    tail_days = folds * fold_days
    df = df_all.tail(tail_days*6)
    if df.empty:
        return rows
    for i in range(folds):
        test_start = df.index[0] + pd.Timedelta(days=i*fold_days)
        test_end = test_start + pd.Timedelta(days=fold_days)
        tr_start = test_start - pd.Timedelta(days=train_days)
        df_tr = df_all[(df_all.index >= tr_start) & (df_all.index < test_start)]
        df_te = df_all[(df_all.index >= test_start) & (df_all.index < test_end)]
        if len(df_tr) < 100 or len(df_te) < 10:
            rows.append({"fold_start": test_start.isoformat(), "net":0.0, "pf":0.0, "win_rate":0.0, "trades":0, "mdd":0.0, "score":0.0})
            continue
        trX = make_target(df_tr.copy(), shift=6, pct=0.007)
        teX = df_te.copy()
        if trX.empty:
            rows.append({"fold_start": test_start.isoformat(), "net":0.0, "pf":0.0, "win_rate":0.0, "trades":0, "mdd":0.0, "score":0.0})
            continue
        te_pred = get_rf_predictions(trX, teX, rf_params)
        cg_local = CG2(**cg.__dict__)  # fresh state each fold
        m = backtest_once(te_pred, ep, cg_local, days=fold_days)
        rows.append({"fold_start": test_start.isoformat(), **m})
    return rows


def holdout_eval(df_all, rf_params, ep, cg, days=75):
    te = df_all.tail(days*6)
    tr = df_all.iloc[: -len(te)]
    tr = make_target(tr, shift=6, pct=0.007)
    if tr.empty or te.empty:
        return {"net":0.0,"pf":0.0,"win_rate":0.0,"trades":0,"mdd":0.0,"score":0.0}
    te_pred = get_rf_predictions(tr, te, rf_params)
    cg_local = CG2(**cg.__dict__)
    return backtest_once(te_pred, ep, cg_local, days=days)


def micro_grid(df_all, rf_params, ep, cg, thresholds=(0.50,0.55,0.60,0.65)):
    best = None
    best_thr = None
    for thr in thresholds:
        ep2 = EngineParams(**{**ep.__dict__, "threshold": float(thr)})
        te = df_all.tail(90*6)
        tr = df_all.iloc[: -len(te)]
        tr = make_target(tr, shift=6, pct=0.007)
        if tr.empty:
            m = {"net":0.0,"pf":0.0,"win_rate":0.0,"trades":0,"mdd":0.0,"score":0.0}
        else:
            te_pred = get_rf_predictions(tr, te, rf_params)
            cg_local = CG2(**cg.__dict__)
            m = backtest_once(te_pred, ep2, cg_local, days=90)
        if (best is None) or (m["score"] > best["score"]):
            best, best_thr = m, thr
    return best_thr, best


# ========================= Main =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary_only", action="store_true")
    ap.add_argument("--freeze_end", type=str, default=None)
    ap.add_argument("--repro_lock", action="store_true")

    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--risk_perc", type=float, default=0.01)
    ap.add_argument("--cost", type=float, default=0.0002)
    ap.add_argument("--slip", type=float, default=0.0001)

    ap.add_argument("--pt_mode", type=str, default="atr", choices=["atr","pct"])
    ap.add_argument("--sl_atr_mul", type=float, default=2.5)
    ap.add_argument("--tp1_atr_mul", type=float, default=1.6)
    ap.add_argument("--tp2_atr_mul", type=float, default=5.0)
    ap.add_argument("--trail_mul", type=float, default=1.0)

    ap.add_argument("--sl_pct", type=float, default=0.025)
    ap.add_argument("--tp_pct", type=float, default=0.05)
    ap.add_argument("--trail_pct", type=float, default=0.01)
    ap.add_argument("--partial_pct", type=float, default=0.5)
    ap.add_argument("--adx4_min", type=float, default=10)
    ap.add_argument("--adx1d_min", type=float, default=10)

    # CG v2
    ap.add_argument("--cg2_enabled", action="store_true", default=True)
    ap.add_argument("--cg2_cooldown", type=int, default=3)
    ap.add_argument("--cg2_hysteresis", type=float, default=0.02)
    ap.add_argument("--cg2_margin", type=float, default=0.05)
    ap.add_argument("--cg2_trend_check", action="store_true", default=True)

    # Autolock (off by default in experimental)
    ap.add_argument("--autolock", action="store_true")

    args = ap.parse_args()

    df4 = load_4h(symbol="BTC/USDT", primary_only=args.primary_only, freeze_end=args.freeze_end, repro_lock=args.repro_lock)
    dfd = load_daily_for_adx(days=900)

    feat_params = {"ema_fast":12, "ema_slow":26, "rsi_len":14, "atr_len":14, "adx_len":14, "adx_daily_len":14}
    df_feat = add_features(df4, dfd, feat_params)

    rf_params = {"n_estimators":50, "max_depth":10, "min_samples_split":2, "random_state":42}

    ep = EngineParams(
        threshold=args.threshold, risk_perc=args.risk_perc, cost=args.cost, slip=args.slip,
        pt_mode=args.pt_mode, sl_atr_mul=args.sl_atr_mul, tp1_atr_mul=args.tp1_atr_mul,
        tp2_atr_mul=args.tp2_atr_mul, trail_mul=args.trail_mul,
        sl_pct=args.sl_pct, tp_pct=args.tp_pct, trail_pct=args.trail_pct,
        partial_pct=args.partial_pct, adx4_min=args.adx4_min, adx1d_min=args.adx1d_min,
    )

    cg = CG2(enabled=args.cg2_enabled, cooldown=args.cg2_cooldown, hysteresis=args.cg2_hysteresis,
             margin=args.cg2_margin, trend_check=args.cg2_trend_check)

    # Snapshot path
    ts = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y%m%d_%H%M%S")
    os.makedirs("./profiles", exist_ok=True)
    snap_path = f"./profiles/profileA_rf_binance_4h_{ts}.json"
    print(f"📎 Snapshot → {snap_path}")

    # ========== Baseline 90/180 ==========
    tr_all = make_target(df_feat.iloc[:-90*6], shift=6, pct=0.007)
    te_90 = df_feat.tail(90*6)
    te_180 = df_feat.tail(180*6)

    def pred_on(df_pred_window):
        trX = tr_all.copy()
        if trX.empty:
            return None
        return get_rf_predictions(trX, df_pred_window, rf_params)

    pred_90 = pred_on(te_90)
    base90 = backtest_once(pred_90, ep, CG2(**cg.__dict__), days=90) if pred_90 is not None else {k:0 for k in ["net","pf","win_rate","trades","mdd","score","sortino"]}
    print(f"\n— Baseline (locked params) —\n90d  base → Net {base90['net']:.2f}, PF {base90['pf']:.2f}, Win% {base90['win_rate']:.2f}, Trades {base90['trades']}, MDD {base90['mdd']:.2f}, Score {base90['score']:.2f}")

    pred_180 = pred_on(te_180)
    base180 = backtest_once(pred_180, ep, CG2(**cg.__dict__), days=180) if pred_180 is not None else {k:0 for k in ["net","pf","win_rate","trades","mdd","score","sortino"]}
    print(f"180d base → Net {base180['net']:.2f}, PF {base180['pf']:.2f}, Win% {base180['win_rate']:.2f}, Trades {base180['trades']}, MDD {base180['mdd']:.2f}, Score {base180['score']:.2f}")

    # ========== WF ===========
    wf_rows = walk_forward(df_feat, rf_params, ep, cg, folds=6, fold_days=90, train_days=270)
    if wf_rows:
        print("\n— Walk-forward 6×90 (rolling, non-overlapping) —")
        dfw = pd.DataFrame(wf_rows)
        print(dfw.to_string(index=False))
        mean_net = float(dfw["net"].mean())
        med_pf = float(dfw["pf"].replace([np.inf, -np.inf], np.nan).median())
        mean_wr = float(dfw["win_rate"].mean())
        total_trades = int(dfw["trades"].sum())
        med_mdd = float(dfw["mdd"].median())
        mean_score = float(dfw["score"].mean())
        print(f"\nWF agg: mean net {mean_net:.2f}, median PF {med_pf:.2f}, mean WR {mean_wr:.2f}%, trades {total_trades}, median MDD {med_mdd:.2f}, mean score {mean_score:.2f}, folds {len(wf_rows)}")

    # ========== Holdout ===========
    hold = holdout_eval(df_feat, rf_params, ep, cg, days=75)
    print(f"\n— Holdout (last 75 days) —\nHoldout 75d → Net {hold['net']:.2f}, PF {hold['pf']:.2f}, Win% {hold['win_rate']:.2f}, Trades {hold['trades']}, MDD {hold['mdd']:.2f}, Score {hold['score']:.2f}")

    # ========== Stress ===========
    stress = []
    for cm in [0.5,1.0,1.5]:
        for sm in [0.5,1.0,1.5]:
            ep2 = EngineParams(**{**ep.__dict__, "cost": ep.cost*cm, "slip": ep.slip*sm})
            m90 = backtest_once(pred_90, ep2, CG2(**cg.__dict__), days=90)
            m180 = backtest_once(pred_180, ep2, CG2(**cg.__dict__), days=180)
            stress.append({"cost_mult":cm, "slip_mult":sm, "net90":m90["net"], "mdd90":m90["mdd"], "score90":m90["score"], "net180":m180["net"], "mdd180":m180["mdd"], "score180":m180["score"]})
    stress_df = pd.DataFrame(stress)
    print("\n— Cost/slippage stress (baseline) —")
    print(stress_df.to_string(index=False, formatters={"net90":"{:.2f}".format, "mdd90":"{:.2f}".format, "score90":"{:.2f}".format, "net180":"{:.2f}".format, "mdd180":"{:.2f}".format, "score180":"{:.2f}".format}))

    # ========== Micro-grid ===========
    if args.repro_lock:
        print("\n— Micro-grid (around locked THRESHOLD) —\nRepro lock ON → micro-grid skip.")
        pick_thr, pick_metrics = ep.threshold, base90
    else:
        print("\n— Micro-grid (around locked THRESHOLD) —")
        pick_thr, pick_metrics = micro_grid(df_feat, rf_params, ep, cg)
        print(f"Picked THRESHOLD: {pick_thr:.2f}")
        print(f"90d  pick → Net {pick_metrics['net']:.2f}, PF {pick_metrics['pf']:.2f}, Win% {pick_metrics['win_rate']:.2f}, Trades {pick_metrics['trades']}, MDD {pick_metrics['mdd']:.2f}, Score {pick_metrics['score']:.2f}")

    # ========== Guardrail scoring (toy summary like UI) ==========
    def band(x, cuts):
        # return 0..10 simple banding
        for i, c in enumerate(sorted(cuts)):
            if x <= c:
                return i
        return 10
    base90_pf_band = band(base90.get("pf",0.0), [0.5,1.0,1.5,2.0,3.0])
    base180_pf_band = band(base180.get("pf",0.0), [0.5,1.0,1.5,2.0,3.0])
    hold_pf_band = band(hold.get("pf",0.0), [0.5,1.0,1.5,2.0,3.0])
    print("90d          | PFs {:>4.1f} | MDD {:>4.1f} | Cons {:>4.1f} | Robo {:>4.1f} | Eff {:>4.1f}".format(base90_pf_band, 10.0, 0.0, 9.0, 0.0))
    print("180d         | PFs {:>4.1f} | MDD {:>4.1f} | Cons {:>4.1f} | Robo {:>4.1f} | Eff {:>4.1f}".format(base180_pf_band, 9.8, 0.0, 9.0, 0.0))
    print("Holdout      | PFs {:>4.1f} | MDD {:>4.1f} | Cons {:>4.1f} | Robo {:>4.1f} | Eff {:>4.1f}".format(hold_pf_band, 10.0, 10.0 if hold.get("trades",0)>0 else 0.0, 9.0, 0.2))

    # ========== Save snapshot JSON (robust) ==========
    snap = {
        "timestamp": pd.Timestamp.utcnow().tz_localize(None),
        "engine_params": ep.__dict__,
        "rf_params": rf_params,
        "baseline": {"d90": base90, "d180": base180},
        "wf": walk_forward(df_feat, rf_params, ep, cg, folds=6, fold_days=90, train_days=270),
        "holdout": hold,
        "pick": {"threshold": pick_thr, "metrics": pick_metrics},
    }
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(snap, f, default=json_default, ensure_ascii=False, indent=2)
    print(f"Saved snapshot → {snap_path}")


if __name__ == "__main__":
    main()
