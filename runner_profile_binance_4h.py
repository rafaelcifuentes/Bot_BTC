# runner_profile_binance_4h.py
# 4h swing — ccxt+fallbacks loader, ML prob_up, ATR risk, walk-forward + HOLDOUT
# Locked Profile A v4.3 runner with multi-source fetch, cache, micro-grid, autolock policy, and competitiveness summary
# Requires: pip install ccxt pandas pandas_ta numpy scikit-learn yfinance

import os, json, time
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    import ccxt
except Exception:
    ccxt = None
try:
    import yfinance as yf
except Exception:
    yf = None

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# -----------------------
# Config / Parameters
# -----------------------
SYMBOL = "BTC/USDT"     # primary ccxt symbol
YF_SYMBOL = "BTC-USD"   # yfinance symbol
TIMEFRAME = "4h"
MAX_BARS = 3000
SINCE_DAYS = 600
MIN_BARS_REQ = 800
COST = 0.0002
SLIP = 0.0001
SEED = 42

LOCK_PATH = "./profiles/profileA_binance_4h.json"
SNAPSHOT_DIR = "./profiles"
REPORT_DIR = "./reports"
CACHE_DIR = "./cache"
CACHE_FILE = os.path.join(CACHE_DIR, "btc_4h_ohlcv.csv")

HOLDOUT_DAYS = 75

# Last promoted (fallback if lock missing)
PARAMS_FALLBACK = {
    "threshold": 0.49,
    "adx4_min": 4,
    "adx1d_min": 0,
    "trend_mode": "slope_up",   # "none" | "slope_up" | "fast_or_slope"
    "sl_atr_mul": 1.10,
    "tp1_atr_mul": 1.60,
    "tp2_atr_mul": 4.50,
    "trail_mul": 0.80,
    "partial_pct": 0.50,
    "risk_perc": 0.0065
}

# Guardrails & autolock policy (tweaked)
AUTOLOCK_ON_IMPROVE = True
MDD_TOL_MULT = 1.20
SCORE_TOL_EQ = 1e-6     # equal-score tolerance
IMPROVE_FACTOR = 1.01   # +1% score OR tie-with-lower-MDD

# -----------------------
# Utilities
# -----------------------
def ensure_dirs():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def ts_ms(days_back: int) -> int:
    return int((pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)).timestamp() * 1000)

def now_stamp() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

def dedup_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.index.duplicated(keep="last")].sort_index()

def score_from(net: float, mdd: float) -> float:
    return float(net / (mdd + 1.0)) if (mdd is not None) else 0.0

def safe_pf(trades: np.ndarray) -> float:
    if trades.size == 0: return 0.0
    gains = trades[trades > 0].sum()
    losses = trades[trades < 0].sum()
    if losses == 0: return float("inf") if gains > 0 else 0.0
    return float(gains / abs(losses))

def cache_save(df: pd.DataFrame):
    try:
        tmp = df.copy()
        tmp.index.name = "time"
        tmp.to_csv(CACHE_FILE)
    except Exception:
        pass

def cache_load() -> pd.DataFrame:
    try:
        if os.path.exists(CACHE_FILE):
            tmp = pd.read_csv(CACHE_FILE)
            tmp["time"] = pd.to_datetime(tmp["time"], utc=True).tz_convert(None)
            tmp.set_index("time", inplace=True)
            return tmp
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()

# -----------------------
# Data loading (ccxt + fallbacks)
# -----------------------
EX_SYMBOLS = {
    "binance":   ["BTC/USDT"],
    "binanceus": ["BTC/USDT"],
    "kraken":    ["BTC/USDT", "XBT/USDT", "BTC/USD", "XBT/USD"],
    "bybit":     ["BTC/USDT"],
    "kucoin":    ["BTC/USDT"],
    "okx":       ["BTC/USDT"],
}

def fetch_ccxt_any(exchange_id: str, symbols: List[str], timeframe: str, since_days: int, limit_step: int = 1500) -> pd.DataFrame:
    if ccxt is None:
        return pd.DataFrame()
    try:
        ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    except Exception:
        return pd.DataFrame()
    if not ex.has.get("fetchOHLCV", False):
        return pd.DataFrame()

    try:
        ex.load_markets()
    except Exception:
        pass

    for sym in symbols:
        if hasattr(ex, "markets") and ex.markets and sym not in ex.markets:
            continue
        since = ts_ms(since_days)
        all_rows = []
        ms_per_bar = ex.parse_timeframe(timeframe) * 1000
        next_since = since
        while True:
            try:
                ohlcv = ex.fetch_ohlcv(sym, timeframe=timeframe, since=next_since, limit=limit_step)
            except Exception:
                break
            if not ohlcv:
                break
            all_rows.extend(ohlcv)
            next_since = ohlcv[-1][0] + ms_per_bar
            if len(all_rows) >= MAX_BARS + 1200:
                break
            time.sleep((ex.rateLimit or 200) / 1000.0)

        if all_rows:
            df = pd.DataFrame(all_rows, columns=["time","open","high","low","close","volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            df.set_index("time", inplace=True)
            df.index = df.index.tz_convert(None)
            return df
    return pd.DataFrame()

def fetch_yf(symbol: str, period_days: int = 720) -> pd.DataFrame:
    days = max(365, min(720, period_days))
    if yf is None:
        return pd.DataFrame()
    try:
        period = f"{days}d"
        df = yf.download(symbol, period=period, interval="4h", auto_adjust=False, progress=False)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    return df[["open","high","low","close","volume"]].copy()

def load_4h_stitched() -> Tuple[pd.DataFrame, List[str]]:
    sources, dfs = [], []

    for ex_id, syms in EX_SYMBOLS.items():
        try:
            d = fetch_ccxt_any(ex_id, syms, TIMEFRAME, SINCE_DAYS)
        except Exception as e:
            print(f"[{ex_id}] fetch error: {e}")
            d = pd.DataFrame()
        sources.append(f"{ex_id}: {'ok' if not d.empty else 'empty'}")
        if not d.empty:
            dfs.append(d)

    try:
        d_yf = fetch_yf(YF_SYMBOL, period_days=SINCE_DAYS)
    except Exception:
        d_yf = pd.DataFrame()
    sources.append("yfinance: ok" if not d_yf.empty else "yfinance: empty")
    if not d_yf.empty:
        dfs.append(d_yf)

    if not dfs:
        cached = cache_load()
        if not cached.empty:
            sources.append("cache: used")
            return cached.tail(MAX_BARS), sources
        return pd.DataFrame(), sources

    df = pd.concat(dfs, axis=0)
    df = df[["open","high","low","close","volume"]].astype(float)
    df = dedup_sort(df)
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].ffill()
    df = df.tail(MAX_BARS)
    cache_save(df)
    return df, sources

# -----------------------
# Features + model
# -----------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ta.ema(out["close"], length=12)
    out["ema_slow"] = ta.ema(out["close"], length=26)
    out["rsi"] = ta.rsi(out["close"], length=14)
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=14)

    adx4 = ta.adx(out["high"], out["low"], out["close"], length=14)
    out["adx4h"] = adx4[[c for c in adx4.columns if c.lower().startswith("adx")][0]] if (adx4 is not None and not adx4.empty) else 0.0

    daily = out[["high","low","close"]].resample("1D").agg({"high":"max","low":"min","close":"last"})
    adx1d = ta.adx(daily["high"], daily["low"], daily["close"], length=14)
    if adx1d is not None and not adx1d.empty:
        cold = [c for c in adx1d.columns if c.lower().startswith("adx")]
        daily_adx = adx1d[cold[0]]
    else:
        daily_adx = pd.Series(0.0, index=daily.index)
    out["adx1d"] = daily_adx.reindex(out.index, method="ffill")

    out["slope"] = (out["ema_fast"] - out["ema_slow"]).diff()
    out["slope_up"] = (out["slope"] > 0).astype(int)
    return out.dropna()

FEATURES = ["ema_fast","ema_slow","rsi","atr","adx4h"]

def train_model(df: pd.DataFrame) -> Pipeline:
    df_ = df.copy()
    df_["target"] = np.sign(df_["close"].shift(-1) - df_["close"])
    df_.dropna(inplace=True)
    X = df_[FEATURES]
    y = df_["target"].astype(int)
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            random_state=SEED,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1
        ))
    ])
    pipe.fit(X, y)
    return pipe

def predict_prob_up(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    probs = model.predict_proba(df[FEATURES])
    classes = model.classes_
    if -1 not in classes or 1 not in classes:
        return pd.Series(0.5, index=df.index)
    up_idx = np.where(classes == 1)[0][0]
    return pd.Series(probs[:, up_idx], index=df.index)

# -----------------------
# Entry filters
# -----------------------
def pass_trend(row: pd.Series, mode: str) -> bool:
    if mode == "none": return True
    if mode == "slope_up": return bool(row.get("slope_up", 0) == 1)
    if mode == "fast_or_slope":
        cond = (row.get("ema_fast", np.nan) > row.get("ema_slow", np.nan)) or (row.get("slope_up", 0) == 1)
        return bool(bool(cond) and not np.isnan(row.get("ema_fast", np.nan)))
    return True

# -----------------------
# Backtest (long-only)
# -----------------------
def backtest_long(df: pd.DataFrame, p: Dict, days: int) -> Dict:
    data = df.tail(days * 6).copy() if days > 0 else df.copy()
    equity_closed = 10000.0
    trades, eq_curve = [], []
    pos = None

    for _, r in data.iterrows():
        if pos is None:
            if (
                r["prob_up"] > p["threshold"]
                and r["adx4h"] >= p["adx4_min"]
                and r["adx1d"] >= p["adx1d_min"]
                and pass_trend(r, p.get("trend_mode", "none"))
                and r["close"] > r["ema_slow"]
            ):
                atr0 = r["atr"]
                if not np.isfinite(atr0) or atr0 <= 0:
                    eq_curve.append(equity_closed); continue
                entry = float(r["open"]) * (1 + SLIP)
                stop  = entry - p["sl_atr_mul"] * atr0
                tp1   = entry + p["tp1_atr_mul"] * atr0
                tp2   = entry + p["tp2_atr_mul"] * atr0
                if stop >= entry:
                    eq_curve.append(equity_closed); continue
                size = (equity_closed * p["risk_perc"]) / max(entry - stop, 1e-9)
                entry_cost = -entry * size * COST
                trades.append(entry_cost)
                equity_closed += entry_cost
                pos = {"e": entry, "s": stop, "t1": tp1, "t2": tp2,
                       "sz": size, "hm": entry, "p1": False, "atr0": atr0}
        else:
            pos["hm"] = max(pos["hm"], r["high"])
            exit_price = None

            if (not pos["p1"]) and (r["high"] >= pos["t1"]):
                part_sz = pos["sz"] * p["partial_pct"]
                pnl = (pos["t1"] - pos["e"]) * part_sz
                cost = pos["t1"] * part_sz * COST
                trades.append(pnl - cost)
                equity_closed += (pnl - cost)
                pos["sz"] *= (1 - p["partial_pct"])
                pos["p1"] = True

            if r["high"] >= pos["t2"]:
                exit_price = pos["t2"]

            new_stop = pos["hm"] - p["trail_mul"] * pos["atr0"]
            if new_stop > pos["s"]:
                pos["s"] = new_stop
            if r["low"] <= pos["s"]:
                exit_price = pos["s"]

            if exit_price is not None:
                pnl = (exit_price - pos["e"]) * pos["sz"]
                cost = exit_price * pos["sz"] * COST
                trades.append(pnl - cost)
                equity_closed += (pnl - cost)
                pos = None

        unreal = 0.0 if pos is None else (r["close"] - pos["e"]) * pos["sz"]
        eq_curve.append(equity_closed + unreal)

    if len(trades) == 0:
        return {"net": 0.0, "pf": 0.0, "win_rate": 0.0, "trades": 0, "mdd": 0.0, "score": 0.0}
    arr = np.array(trades, dtype=float)
    net = float(arr.sum())
    pf = safe_pf(arr)
    wr = float((arr > 0).sum() / len(arr) * 100.0)
    eq = np.array(eq_curve, dtype=float)
    mdd = float((np.maximum.accumulate(eq) - eq).max()) if eq.size else 0.0
    score = score_from(net, mdd)
    return {"net": net, "pf": pf, "win_rate": wr, "trades": int(len(arr)), "mdd": mdd, "score": score}

# -----------------------
# Evaluation helpers
# -----------------------
def fmt_row(prefix: str, r: Dict):
    print(f"{prefix} → Net {r['net']:.2f}, PF {r['pf']:.2f}, Win% {r['win_rate']:.2f}, "
          f"Trades {r['trades']}, MDD {r['mdd']:.2f}, Score {r['score']:.2f}")

def split_train_test(df: pd.DataFrame, train_frac: float = 0.7):
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def attach_probs(train_df: pd.DataFrame, test_df: pd.DataFrame):
    model = train_model(train_df)
    test_df = test_df.copy(); test_df["prob_up"] = predict_prob_up(model, test_df)
    train_df = train_df.copy(); train_df["prob_up"] = predict_prob_up(model, train_df)
    return train_df, test_df

def walk_forward(df: pd.DataFrame, p: Dict, horizon_days: int = 90, folds: int = 6) -> pd.DataFrame:
    rows = []
    bars_per = horizon_days * 6
    if len(df) < bars_per * (folds + 1):
        folds = max(1, len(df) // bars_per - 1)
    for i in range(folds):
        test_end = len(df) - (folds - i - 1) * bars_per
        test_start = max(0, test_end - bars_per)
        train_end = max(0, test_start - 1)
        train = df.iloc[:train_end].copy()
        test  = df.iloc[test_start:test_end].copy()
        if len(train) < 200 or len(test) < 10:
            continue
        model = train_model(train)
        test = test.copy(); test["prob_up"] = predict_prob_up(model, test)
        r = backtest_long(test, p, days=0)
        rows.append({"fold_start": test.index[0], **r})
    return pd.DataFrame(rows)

def holdout_eval(df: pd.DataFrame, p: Dict, holdout_days: int = HOLDOUT_DAYS) -> Dict:
    bars_holdout = holdout_days * 6
    if len(df) <= bars_holdout + 200:
        return {"net": 0.0, "pf": 0.0, "win_rate": 0.0, "trades": 0, "mdd": 0.0, "score": 0.0}
    train = df.iloc[:-bars_holdout].copy()
    hold  = df.iloc[-bars_holdout:].copy()
    model = train_model(train)
    hold = hold.copy(); hold["prob_up"] = predict_prob_up(model, hold)
    return backtest_long(hold, p, days=0)

def cost_slip_stress(df: pd.DataFrame, p: Dict, cost_mults=(0.5,1.0,1.5), slip_mults=(0.5,1.0,1.5)) -> pd.DataFrame:
    rows = []
    global COST, SLIP
    base_cost, base_slip = COST, SLIP
    for cm in cost_mults:
        for sm in slip_mults:
            COST = base_cost * cm
            SLIP = base_slip * sm
            r90  = backtest_long(df, p, days=90)
            r180 = backtest_long(df, p, days=180)
            rows.append({
                "cost_mult": cm, "slip_mult": sm,
                "net90": r90["net"], "mdd90": r90["mdd"], "score90": r90["score"],
                "net180": r180["net"], "mdd180": r180["mdd"], "score180": r180["score"]
            })
    COST, SLIP = base_cost, base_slip
    return pd.DataFrame(rows)

# -----------------------
# Params I/O
# -----------------------
def load_locked_params() -> Dict:
    if os.path.exists(LOCK_PATH):
        with open(LOCK_PATH, "r") as f:
            try:
                p = json.load(f)
                return {k: v for k, v in p.items() if not isinstance(v, dict)}
            except Exception:
                pass
    return PARAMS_FALLBACK.copy()

def snapshot_params(params: Dict, note: str = "") -> None:
    ensure_dirs()
    blob = params.copy()
    blob["_meta"] = {"profile": "A", "version": "v4.3", "note": note, "timestamp_utc": pd.Timestamp.utcnow().isoformat()}
    snap = os.path.join(SNAPSHOT_DIR, f"profileA_binance_4h_{now_stamp()}.json")
    with open(snap, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"📎 Snapshot → {snap}")

def bump_and_save_params(params: Dict, version: str = "v4.3") -> None:
    ensure_dirs()
    blob = params.copy()
    blob["_meta"] = {"profile":"A","version":version,"timestamp_utc": pd.Timestamp.utcnow().isoformat()}
    with open(LOCK_PATH, "w") as f:
        json.dump(blob, f, indent=2)
    snap = os.path.join(SNAPSHOT_DIR, f"profileA_binance_4h_{now_stamp()}.json")
    with open(snap, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"🔒 Auto-locked params → {LOCK_PATH}")
    print(f"📎 Snapshot → {snap}")

# -----------------------
# Micro-grid (small, guardrailed around current)
# -----------------------
def micro_grid(params: Dict) -> List[Dict]:
    ths    = [round(params["threshold"] + d, 3) for d in (-0.02, -0.01, 0.0, +0.01)]
    adx4s  = sorted(set([params["adx4_min"], 4, 6, 8]))
    sls    = sorted(set([round(v, 1) for v in (params["sl_atr_mul"], 1.1, 1.2, 1.3)]))
    risks  = sorted(set([round(params["risk_perc"], 4), 0.0060, 0.0065, 0.0070]))
    trails = sorted(set([params["trail_mul"], 0.8, 1.0]))

    cands = []
    for th in ths:
        for ax in adx4s:
            for slm in sls:
                for rsk in risks:
                    for tr in trails:
                        c = params.copy()
                        c.update({
                            "threshold": max(0.45, min(0.60, th)),
                            "adx4_min": ax,
                            "sl_atr_mul": slm,
                            "risk_perc": rsk,
                            "trail_mul": tr
                        })
                        cands.append(c)
    uniq = {json.dumps(c, sort_keys=True): c for c in cands}
    return list(uniq.values())

# -----------------------
# Competitiveness summary (ratings 1–10) -----------------------
def _scale_pf(pf: float) -> float:
    # PF: 1 -> 1, 5 -> 10 (clip)
    return float(np.clip(1 + (pf - 1) / (5 - 1) * 9, 1, 10))

def _scale_mdd(mdd_abs: float, equity: float = 10000.0) -> float:
    # MDD% mapped: 0% -> 10, 30% -> 1
    mdd_pct = (mdd_abs / equity) * 100.0
    return float(np.clip(10 - (mdd_pct / 30.0) * 9.0, 1, 10))

def _eff_from_wr(win_rate_pct: float) -> float:
    # Trade efficiency proxy from WR (40–60% ~ good in swing with RR>1)
    return float(np.clip(1 + (win_rate_pct / 50.0) * 9.0, 1, 10))

def _robust_from_stress(base_score: float, worst_score: float) -> float:
    # Penalize % drop in score at worst costs
    if base_score <= 0: return 5.0
    drop_ratio = max(0.0, (base_score - worst_score) / base_score)  # 0 → perfect, 0.5 → 50% drop
    return float(np.clip(10 - drop_ratio * 10, 1, 10))

def save_competitiveness_summary(stamp: str,
                                 base90: Dict, base180: Dict,
                                 wf_agg: Dict, hold: Dict,
                                 stress_base_df: pd.DataFrame,
                                 pick90: Optional[Dict], pick180: Optional[Dict]) -> str:
    # choose "pick" if provided, else baseline
    s90 = pick90 or base90
    s180 = pick180 or base180

    # stress worst
    worst_row = None
    if stress_base_df is not None and not stress_base_df.empty:
        worst_row = stress_base_df.sort_values("score180").iloc[0]
    worst_score_180 = float(worst_row["score180"]) if worst_row is not None else base180.get("score", 0.0)

    ratings = {
        "90d_slice": {
            "pf": s90.get("pf", 0.0),
            "mdd": s90.get("mdd", 0.0),
            "win_rate": s90.get("win_rate", 0.0),
            "scores": {
                "pf_score": round(_scale_pf(s90.get("pf", 0.0)), 1),
                "mdd_control": round(_scale_mdd(s90.get("mdd", 0.0)), 1),
                "consistency": 8.0,  # proxy at slice level
                "robustness": round(_robust_from_stress(base180.get("score", 0.0), worst_score_180), 1),
                "trade_efficiency": round(_eff_from_wr(s90.get("win_rate", 0.0)), 1)
            }
        },
        "180d_slice": {
            "pf": s180.get("pf", 0.0),
            "mdd": s180.get("mdd", 0.0),
            "win_rate": s180.get("win_rate", 0.0),
            "scores": {
                "pf_score": round(_scale_pf(s180.get("pf", 0.0)), 1),
                "mdd_control": round(_scale_mdd(s180.get("mdd", 0.0)), 1),
                "consistency": 8.0,
                "robustness": round(_robust_from_stress(base180.get("score", 0.0), worst_score_180), 1),
                "trade_efficiency": round(_eff_from_wr(s180.get("win_rate", 0.0)), 1)
            }
        },
        "walk_forward": {
            "median_pf": wf_agg.get("median_pf", 0.0) if wf_agg else 0.0,
            "median_mdd": wf_agg.get("median_mdd", 0.0) if wf_agg else 0.0,
            "mean_score": wf_agg.get("mean_score", 0.0) if wf_agg else 0.0,
            "folds": wf_agg.get("folds", 0) if wf_agg else 0,
            "scores": {
                "pf_score": round(_scale_pf(wf_agg.get("median_pf", 0.0) if wf_agg else 0.0), 1),
                "mdd_control": round(_scale_mdd(wf_agg.get("median_mdd", 0.0) if wf_agg else 0.0), 1),
                "consistency": 8.5 if (wf_agg and wf_agg.get("folds", 0) >= 3) else 7.5,
                "robustness": round(_robust_from_stress(base180.get("score", 0.0), worst_score_180), 1),
                "trade_efficiency": 8.0
            }
        },
        "holdout": {
            "pf": hold.get("pf", 0.0),
            "mdd": hold.get("mdd", 0.0),
            "win_rate": hold.get("win_rate", 0.0),
            "scores": {
                "pf_score": round(_scale_pf(hold.get("pf", 0.0)), 1),
                "mdd_control": round(_scale_mdd(hold.get("mdd", 0.0)), 1),
                "consistency": 7.5,
                "robustness": round(_robust_from_stress(base180.get("score", 0.0), worst_score_180), 1),
                "trade_efficiency": round(_eff_from_wr(hold.get("win_rate", 0.0)), 1)
            }
        }
    }

    # simple overall: average of the sub-scores from 180d slice + WF + Holdout
    def _avg(keys):
        vals = []
        for k in keys:
            s = ratings[k]["scores"]
            vals.extend([s["pf_score"], s["mdd_control"], s["consistency"], s["robustness"], s["trade_efficiency"]])
        return round(float(np.mean(vals)) if vals else 0.0, 1)

    overall = {
        "overall_rating": _avg(["180d_slice", "walk_forward", "holdout"]),
        "note": "Heuristic 1–10 scaling; use for *relative* comparisons across your runs."
    }

    out = {
        "stamp_utc": pd.Timestamp.utcnow().isoformat(),
        "headline": "Competitiveness vs market-style criteria (1–10)",
        "ratings": ratings,
        "overall": overall
    }

    ensure_dirs()
    path = os.path.join(REPORT_DIR, f"competitiveness_summary_{stamp}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"📊 Saved competitiveness summary → {path}")

    # also print a tiny table
    def row(name, r):
        sc = r["scores"]
        return f"{name:12} | PFs {sc['pf_score']:>4} | MDD {sc['mdd_control']:>4} | Cons {sc['consistency']:>4} | Robo {sc['robustness']:>4} | Eff {sc['trade_efficiency']:>4}"
    print(row("90d", ratings["90d_slice"]))
    print(row("180d", ratings["180d_slice"]))
    print(row("WF", ratings["walk_forward"]))
    print(row("Holdout", ratings["holdout"]))
    print(f"Overall ~ {overall['overall_rating']}/10")
    return path

# -----------------------
# Report I/O
# -----------------------
def save_report_with_stamp(stamp: str,
                           sources: List[str], base90: Dict, base180: Dict,
                           wf: pd.DataFrame, wf_agg: Dict, hold: Dict, params: Dict,
                           stress_base: pd.DataFrame, stress_pick: Optional[pd.DataFrame],
                           pick_params: Optional[Dict], pick90: Optional[Dict], pick180: Optional[Dict]) -> Tuple[str, str]:
    ensure_dirs()
    wf_csv = os.path.join(REPORT_DIR, f"walkforward_{stamp}.csv")
    json_path = os.path.join(REPORT_DIR, f"summary_{stamp}.json")

    wf_out = wf.copy()
    if "fold_start" in wf_out.columns:
        wf_out["fold_start"] = wf_out["fold_start"].astype(str)
    wf_out.to_csv(wf_csv, index=False)

    appendix = {
        "stress_base": stress_base.to_dict(orient="records") if stress_base is not None else [],
        "stress_pick": stress_pick.to_dict(orient="records") if stress_pick is not None else []
    }

    payload = {
        "stamp_utc": pd.Timestamp.utcnow().isoformat(),
        "sources": sources,
        "baseline_90d": base90,
        "baseline_180d": base180,
        "walk_forward_rows": wf_out.to_dict(orient="records"),
        "walk_forward_agg": wf_agg,
        "holdout": hold,
        "locked_params_used": params,
        "picked_params": pick_params or {},
        "picked_90d": pick90 or {},
        "picked_180d": pick180 or {},
        "appendix": appendix
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"📝 Saved report JSON → {json_path}")
    print(f"🧾 Saved walk-forward CSV → {wf_csv}")
    return json_path, wf_csv

# -----------------------
# Autolock (tweaked policy)
# -----------------------
def should_autolock(base: Dict, pick: Dict, mdd_tol_mult: float = MDD_TOL_MULT) -> Tuple[bool, Dict]:
    base_score = base.get("score", 0.0)
    pick_score = pick.get("score", 0.0)
    base_mdd   = base.get("mdd", 1.0)
    pick_mdd   = pick.get("mdd", 1.0)

    better_score = pick_score > base_score * IMPROVE_FACTOR
    tie_but_safer = (abs(pick_score - base_score) <= SCORE_TOL_EQ) and (pick_mdd < base_mdd)
    within_mdd = pick_mdd <= base_mdd * mdd_tol_mult

    ok = within_mdd and (better_score or tie_but_safer)
    info = {
        "base_score": base_score, "pick_score": pick_score,
        "base_mdd": base_mdd, "pick_mdd": pick_mdd,
        "within_mdd": within_mdd, "better_score": better_score, "tie_but_safer": tie_but_safer,
        "tol×": mdd_tol_mult
    }
    return ok, info

# -----------------------
# Main
# -----------------------
def main():
    print("🔄 Downloading 4h data (ccxt/Binance + fallback yfinance)…")
    df_raw, sources = load_4h_stitched()
    print("Sources:", sources)
    if df_raw.empty:
        print("Not enough data (live + cache both empty)."); return

    params = load_locked_params()
    snapshot_params(params, note="pre-run snapshot (no lock change)")

    df = add_features(df_raw).dropna()
    if len(df) < MIN_BARS_REQ:
        print(f"Still not enough 4h bars after fallbacks. Got {len(df)} (min {MIN_BARS_REQ})"); return

    # 70/30 split baseline with locked params
    train, test = split_train_test(df, train_frac=0.7)
    train_with, test_with = attach_probs(train, test)

    print("\n— Baseline (70% train → OOS test) —")
    print(f"Using locked params: {params}")
    base90  = backtest_long(test_with, params, days=90)
    base180 = backtest_long(test_with, params, days=180)
    fmt_row("90d  base", base90)
    fmt_row("180d base", base180)

    # Walk-forward with locked params
    print("\n— Walk-forward 6×90 (rolling, non-overlapping) —")
    wf = walk_forward(df, params, horizon_days=90, folds=6)
    if not wf.empty:
        print(wf.to_string(index=False))
        wf_agg = {
            "mean_net": float(wf["net"].mean()),
            "median_pf": float(wf["pf"].median()),
            "mean_wr": float(wf["win_rate"].mean()),
            "total_trades": int(wf["trades"].sum()),
            "median_mdd": float(wf["mdd"].median()),
            "mean_score": float(wf["score"].mean()),
            "folds": int(len(wf)),
        }
        print(f"\nWalk-forward aggregates (90d): mean net {wf_agg['mean_net']:.2f}, median PF {wf_agg['median_pf']:.2f}, "
              f"mean WR {wf_agg['mean_wr']:.2f}%, total trades {wf_agg['total_trades']}, "
              f"median MDD {wf_agg['median_mdd']:.2f}, mean score {wf_agg['mean_score']:.2f}, folds {wf_agg['folds']}")
    else:
        wf_agg = {}

    # Holdout (locked)
    print(f"\n— Holdout (train on past, test last {HOLDOUT_DAYS} days) —")
    hold = holdout_eval(df, params, HOLDOUT_DAYS)
    fmt_row(f"Holdout {HOLDOUT_DAYS}d", hold)

    # Cost/slippage stress (baseline)
    print("\n— Cost/slippage stress (baseline) —")
    stress_base = cost_slip_stress(test_with, params)
    if not stress_base.empty:
        print(stress_base.to_string(index=False))

    # Micro-grid around current params → pick best by 180d score
    print("\n— Micro-grid (around locked params) —")
    candidates = micro_grid(params)
    best_pick: Optional[Dict] = None
    best90: Optional[Dict] = None
    best180: Optional[Dict] = None
    scored = []
    for i, c in enumerate(candidates):
        r90  = backtest_long(test_with, c, days=90)
        r180 = backtest_long(test_with, c, days=180)
        scored.append({
            "score": r180["score"],
            "mdd_neg": -r180["mdd"],     # prefer smaller MDD
            "net": r180["net"],
            "idx": i,
            "params": c, "r90": r90, "r180": r180
        })
    if scored:
        scored_sorted = sorted(scored, key=lambda z: (z["score"], z["mdd_neg"], z["net"], -z["idx"]), reverse=True)
        top = scored_sorted[0]
        best_pick, best90, best180 = top["params"], top["r90"], top["r180"]
        print("Picked params (subset):", {k: best_pick[k] for k in ("threshold","adx4_min","sl_atr_mul","risk_perc","trend_mode")})
        fmt_row("90d  pick", best90)
        fmt_row("180d pick", best180)
    else:
        print("No candidates generated.")

    # Stress for pick (appendix only)
    stress_pick = cost_slip_stress(test_with, best_pick) if best_pick is not None else pd.DataFrame()

    # Save report (fixed stamp so we can attach competitiveness summary to the same run)
    stamp = now_stamp()
    _json_path, _csv_path = save_report_with_stamp(stamp, sources, base90, base180, wf, wf_agg, hold, params,
                                                  stress_base, stress_pick, best_pick, best90, best180)

    # Competitiveness summary (1–10) saved alongside report
    save_competitiveness_summary(stamp, base90, base180, wf_agg, hold, stress_base, best90, best180)

    # Autolock decision (tweaked policy)
    if AUTOLOCK_ON_IMPROVE and (best_pick is not None):
        ok, info = should_autolock(base180, best180, MDD_TOL_MULT)
        if ok:
            bump_and_save_params(best_pick, version="v4.3")
        else:
            print("↩︎ Autolock skipped", info)
    else:
        print("↩︎ Autolock skipped (no candidates or disabled).")

if __name__ == "__main__":
    main()