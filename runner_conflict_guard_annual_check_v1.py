# runner_conflict_guard_annual_check_v1.py
# Annual check (~360d) for Andromeda Conflict-Guard (LONG+SHORT) using fixed or stored weights.
# - Loads 4h BTC data via ccxt + fallbacks + yfinance (stitched)
# - Trains the ML model only on PRE-window data
# - Tests inside the year window (~360 days)
# - Computes SOLO LONG, SOLO SHORT, and COMBINED (conflict-guard) metrics for $10,000
# - Optional benchmark check: disabled, fixed pct, or band (min,max)
# - Writes JSON summary to ./reports/annual_check_YYYYMMDD_HHMM.json

import os, re, json, time, argparse, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------- optional deps ----------
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
# Config / Paths
# -----------------------
SYMBOL = "BTC/USDT"
YF_SYMBOL = "BTC-USD"
TIMEFRAME = "4h"
MAX_BARS = 3000
SINCE_DAYS = 600
MIN_BARS_REQ = 800

BASE_COST = 0.0002
BASE_SLIP = 0.0001
SEED = 42

# Lock files (LONG/SHORT) — same as Andromeda v4.3
LOCK_LONG  = "./profiles/profileA_binance_4h.json"
LOCK_SHORT = "./profiles/profileA_short_binance_4h.json"

# Active conflict-guard weights file
WEIGHTS_FILE = "./profiles/andromeda_conflict_guard_weights.json"

# Dirs
SNAPSHOT_DIR = "./profiles"
REPORT_DIR   = "./reports"
CACHE_DIR    = "./cache"
CACHE_FILE   = os.path.join(CACHE_DIR, "btc_4h_ohlcv.csv")

# -----------------------
# Utils
# -----------------------
def ensure_dirs():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def ts_ms(days_back: int) -> int:
    return int((pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)).timestamp() * 1000)

def now_stamp() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")

def dedup_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.index.duplicated(keep="last")].sort_index()

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
# Data loading (ccxt + fallbacks + yfinance)
# -----------------------
EX_SYMBOLS_ALL = {
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
            try:
                time.sleep((ex.rateLimit or 200) / 1000.0)
            except Exception:
                pass
        if all_rows:
            df = pd.DataFrame(all_rows, columns=["time","open","high","low","close","volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            df.set_index("time", inplace=True)
            df.index = df.index.tz_convert(None)
            return df
    return pd.DataFrame()

def fetch_yf(symbol: str, period_days: int = 720) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    days = max(365, min(720, period_days))
    try:
        df = yf.download(symbol, period=f"{days}d", interval="4h", auto_adjust=False, progress=False)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    return df[["open","high","low","close","volume"]].copy()

def load_4h_stitched(primary_only: bool=False) -> Tuple[pd.DataFrame, List[str]]:
    sources, dfs = [], []
    ex_map = {"binance": EX_SYMBOLS_ALL["binance"]} if primary_only else EX_SYMBOLS_ALL
    for ex_id, syms in ex_map.items():
        try:
            d = fetch_ccxt_any(ex_id, syms, TIMEFRAME, SINCE_DAYS)
        except Exception:
            d = pd.DataFrame()
        sources.append(f"{ex_id}: {'ok' if not d.empty else 'empty'}")
        if not d.empty: dfs.append(d)
    try:
        d_yf = fetch_yf(YF_SYMBOL, period_days=SINCE_DAYS)
    except Exception:
        d_yf = pd.DataFrame()
    sources.append("yfinance: ok" if not d_yf.empty else "yfinance: empty")
    if not d_yf.empty and not primary_only:
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
# Features + model (Andromeda)
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
    out["slope_down"] = (out["slope"] < 0).astype(int)
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
# Locks / params
# -----------------------
def load_lock(path: str, fallback: Dict) -> Dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                p = json.load(f)
                return {k: v for k, v in p.items() if not isinstance(v, dict)}
            except Exception:
                pass
    return fallback.copy()

FALLBACK_LONG = {
    "threshold": 0.46, "adx4_min": 4, "adx1d_min": 0, "trend_mode": "slope_up",
    "sl_atr_mul": 1.10, "tp1_atr_mul": 1.60, "tp2_atr_mul": 4.50, "trail_mul": 0.80,
    "partial_pct": 0.50, "risk_perc": 0.007
}
FALLBACK_SHORT = {
    "threshold": 0.47, "adx4_min": 4, "adx1d_min": 0, "trend_mode": "slope_down",
    "sl_atr_mul": 1.30, "tp1_atr_mul": 1.60, "tp2_atr_mul": 4.50, "trail_mul": 0.80,
    "partial_pct": 0.50, "risk_perc": 0.0045
}

# -----------------------
# Entry filters
# -----------------------
def pass_trend(row: pd.Series, mode: str) -> bool:
    if mode == "none": return True
    if mode == "slope_up":   return bool(row.get("slope_up", 0) == 1)
    if mode == "slope_down": return bool(row.get("slope_down", 0) == 1)
    if mode == "fast_or_slope":
        cond = (row.get("ema_fast", np.nan) > row.get("ema_slow", np.nan)) or (row.get("slope_up", 0) == 1)
        return bool(cond) and np.isfinite(row.get("ema_fast", np.nan))
    return True

# -----------------------
# Backtests (ATR) — identical logic to Andromeda
# -----------------------
def step_trade_long(r, p, SLIP, COST):
    atr0 = r["atr"]
    if not np.isfinite(atr0) or atr0 <= 0: return None
    entry = float(r["open"]) * (1 + SLIP)
    stop  = entry - p["sl_atr_mul"] * atr0
    tp1   = entry + p["tp1_atr_mul"] * atr0
    tp2   = entry + p["tp2_atr_mul"] * atr0
    if stop >= entry: return None
    size = (p["_equity"] * p["risk_perc"]) / max(entry - stop, 1e-9)
    entry_cost = -entry * size * COST
    p["_equity"] += entry_cost
    p["_trades"].append(entry_cost)
    return {"dir":"long","e": entry, "s": stop, "t1": tp1, "t2": tp2, "sz": size, "hm": entry, "p1": False, "atr0": atr0}

def step_trade_short(r, p, SLIP, COST):
    atr0 = r["atr"]
    if not np.isfinite(atr0) or atr0 <= 0: return None
    entry = float(r["open"]) * (1 - SLIP)
    stop  = entry + p["sl_atr_mul"] * atr0
    tp1   = entry - p["tp1_atr_mul"] * atr0
    tp2   = entry - p["tp2_atr_mul"] * atr0
    if stop <= entry: return None
    size = (p["_equity"] * p["risk_perc"]) / max(stop - entry, 1e-9)
    entry_cost = -entry * size * COST
    p["_equity"] += entry_cost
    p["_trades"].append(entry_cost)
    return {"dir":"short","e": entry, "s": stop, "t1": tp1, "t2": tp2, "sz": size, "lm": entry, "p1": False, "atr0": atr0}

def run_portfolio(df: pd.DataFrame, pL: Dict, pS: Dict, wL: float, wS: float,
                  cost=BASE_COST, slip=BASE_SLIP, start_equity: float = 10000.0,
                  return_equity: bool = False) -> Dict:
    # pesos escalan risk_perc de cada pierna
    pL = pL.copy(); pS = pS.copy()
    pL["risk_perc"] *= float(wL)
    pS["risk_perc"] *= float(wS)

    equity_closed = float(start_equity)
    trades = []
    eq_curve = []
    pos = None

    for _, r in df.iterrows():
        # cierre
        if pos is not None:
            if pos["dir"] == "long":
                pos["hm"] = max(pos["hm"], r["high"])
                exit_price = None
                if (not pos["p1"]) and (r["high"] >= pos["t1"]):
                    part_sz = pos["sz"] * pL["partial_pct"]
                    pnl = (pos["t1"] - pos["e"]) * part_sz
                    fee = pos["t1"] * part_sz * cost
                    trades.append(pnl - fee); equity_closed += (pnl - fee)
                    pos["sz"] *= (1 - pL["partial_pct"]); pos["p1"] = True
                if r["high"] >= pos["t2"]:
                    exit_price = pos["t2"]
                new_stop = pos["hm"] - pL["trail_mul"] * pos["atr0"]
                if new_stop > pos["s"]: pos["s"] = new_stop
                if r["low"] <= pos["s"]:
                    exit_price = pos["s"]
                if exit_price is not None:
                    pnl = (exit_price - pos["e"]) * pos["sz"]
                    fee = exit_price * pos["sz"] * cost
                    trades.append(pnl - fee); equity_closed += (pnl - fee); pos = None
            else:
                low_mark = min(pos.get("lm", pos["e"]), r["low"])
                pos["lm"] = low_mark
                exit_price = None
                if (not pos["p1"]) and (r["low"] <= pos["t1"]):
                    part_sz = pos["sz"] * pS["partial_pct"]
                    pnl = (pos["e"] - pos["t1"]) * part_sz
                    fee = pos["t1"] * part_sz * cost
                    trades.append(pnl - fee); equity_closed += (pnl - fee)
                    pos["sz"] *= (1 - pS["partial_pct"]); pos["p1"] = True
                if r["low"] <= pos["t2"]:
                    exit_price = pos["t2"]
                new_stop = pos["lm"] + pS["trail_mul"] * pos["atr0"]
                if new_stop < pos["s"]: pos["s"] = new_stop
                if r["high"] >= pos["s"]:
                    exit_price = pos["s"]
                if exit_price is not None:
                    pnl = (pos["e"] - exit_price) * pos["sz"]
                    fee = exit_price * pos["sz"] * cost
                    trades.append(pnl - fee); equity_closed += (pnl - fee); pos = None

        # apertura (guardia de conflicto: 1 posición a la vez)
        if pos is None:
            ok_long = (
                (r["prob_up"] > pL["threshold"])
                and (r["adx4h"] >= pL["adx4_min"]) and (r["adx1d"] >= pL["adx1d_min"])
                and pass_trend(r, pL.get("trend_mode","none"))
                and r["close"] > r["ema_slow"]
            )
            ok_short = (
                (r["prob_up"] < (1 - pS["threshold"]))
                and (r["adx4h"] >= pS["adx4_min"]) and (r["adx1d"] >= pS["adx1d_min"])
                and pass_trend(r, pS.get("trend_mode","none"))
                and r["close"] < r["ema_slow"]
            )
            if ok_long:
                pL["_equity"] = equity_closed; pL["_trades"] = trades
                pos = step_trade_long(r, pL, SLIP=BASE_SLIP, COST=BASE_COST)
                equity_closed = pL["_equity"]
            elif ok_short:
                pS["_equity"] = equity_closed; pS["_trades"] = trades
                pos = step_trade_short(r, pS, SLIP=BASE_SLIP, COST=BASE_COST)
                equity_closed = pS["_equity"]

        # curva
        unreal = 0.0
        if pos is not None:
            if pos["dir"] == "long":
                unreal = (r["close"] - pos["e"]) * pos["sz"]
            else:
                unreal = (pos["e"] - r["close"]) * pos["sz"]
        eq_curve.append(equity_closed + unreal)

    arr = np.array(trades, dtype=float)
    net = float(arr.sum()) if arr.size else 0.0
    pf = safe_pf(arr) if arr.size else 0.0
    wr = float((arr > 0).sum() / len(arr) * 100.0) if arr.size else 0.0
    eq = np.array(eq_curve, dtype=float)
    mdd = float((np.maximum.accumulate(eq) - eq).max()) if eq.size else 0.0
    score = float(net / (mdd + 1e-9)) if mdd > 0 else net
    res = {"net": net, "pf": pf, "wr": wr, "trades": int(len(arr)), "mdd": mdd, "score": score,
           "final_equity": float((eq[-1] if eq.size else start_equity))}
    if return_equity:
        res["_equity_curve"] = eq_curve
    return res

# -----------------------
# Helpers
# -----------------------
def parse_cap_mode(cap_mode: str, mdd_long: float, mdd_short: float) -> Optional[float]:
    if not cap_mode or cap_mode.lower() == "none":
        return None
    m = re.match(r"^(min_leg|max_leg)\*([0-9]*\.?[0-9]+)$", cap_mode.strip())
    if not m:
        return None
    leg = m.group(1); mult = float(m.group(2))
    base = min(mdd_long, mdd_short) if leg == "min_leg" else max(mdd_long, mdd_short)
    return base * mult

def load_weights(default_wL: float = 0.35, default_wS: float = 0.65) -> Tuple[float, float, Dict]:
    meta = {}
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, "r") as f:
                obj = json.load(f)
                wL = float(obj.get("wL", default_wL))
                wS = float(obj.get("wS", default_wS))
                meta = obj.get("_meta", {})
                return wL, wS, meta
        except Exception:
            pass
    return default_wL, default_wS, meta

def to_iso(x) -> str:
    try:
        return pd.Timestamp(x).isoformat()
    except Exception:
        return str(x)

def parse_bench_band(s: str) -> Tuple[float, float]:
    try:
        a, b = s.split(",")
        lo = float(a.strip()); hi = float(b.strip())
        if lo > hi: lo, hi = hi, lo
        return lo, hi
    except Exception:
        # default band 40–60 if parsing fails
        return 40.0, 60.0

# -----------------------
# CLI
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Annual (~360d) check for Andromeda Conflict-Guard")
    ap.add_argument("--freeze-end", type=str, required=False, default=None,
                    help='End of test window UTC, e.g., "2025-08-20 12:10"')
    ap.add_argument("--days", type=int, default=360, help="Window length in days (default 360)")
    ap.add_argument("--wL", type=float, default=None, help="Override weight LONG (else read from weights file)")
    ap.add_argument("--wS", type=float, default=None, help="Override weight SHORT (else read from weights file)")

    # Benchmark options (replaces the old fixed 118%)
    ap.add_argument("--bench-mode", type=str, choices=["none","fixed","band"], default="band",
                    help="Benchmark mode: none, fixed (percentage), or band (min,max percent). Default: band")
    ap.add_argument("--bench-fixed-pct", type=float, default=50.0,
                    help="If bench-mode=fixed, target net %% on $10k (e.g., 50 => $5,000 net).")
    ap.add_argument("--bench-band", type=str, default="40,60",
                    help='If bench-mode=band, percentage band "min,max" (default "40,60").')

    ap.add_argument("--cap-mode", type=str, default="min_leg*1.05",
                    help='Cap formula for MDD feasibility (e.g., "min_leg*1.05", "max_leg*1.10", or "none")')
    ap.add_argument("--export-equity", action="store_true", help="Export combined equity curve CSV")
    return ap.parse_args()

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    ensure_dirs()

    # Load data
    print("🔄 Loading 4h data (ccxt + fallbacks + yfinance)…")
    df_raw, sources = load_4h_stitched(primary_only=False)
    if df_raw.empty:
        print("No data available."); return
    print("✨ Adding features…")
    df = add_features(df_raw).dropna()
    if len(df) < MIN_BARS_REQ:
        print(f"Not enough 4h bars ({len(df)} < {MIN_BARS_REQ})."); return

    # Determine window
    if args.freeze_end:
        t_end = pd.Timestamp(args.freeze_end, tz="UTC").tz_convert(None)
    else:
        t_end = pd.Timestamp.utcnow()  # UTC naive
    bars_per_day = 6  # 4h bars / day
    bars_window = int(args.days * bars_per_day)
    df = df.loc[:t_end].copy()
    if len(df) < bars_window + 200:
        print("Not enough history to carve out pre-train + window."); return
    test = df.iloc[-bars_window:].copy()
    train = df.iloc[: -bars_window].copy()

    print(f"🔒 Year window: {to_iso(test.index[0])} → {to_iso(test.index[-1])}  ({len(test)} bars ~ {len(test)/6:.1f} days)")
    print("🧠 Training model on PRE-window only…")
    model = train_model(train)
    test["prob_up"] = predict_prob_up(model, test)

    # Load locks
    pL = load_lock(LOCK_LONG,  FALLBACK_LONG)
    pS = load_lock(LOCK_SHORT, FALLBACK_SHORT)

    # Weights
    wL_file, wS_file, meta = load_weights()
    wL = args.wL if args.wL is not None else wL_file
    wS = args.wS if args.wS is not None else wS_file

    # Solo legs (for cap)
    rL = run_portfolio(test.copy(), pL, pS, wL=1.0, wS=0.0)
    rS = run_portfolio(test.copy(), pL, pS, wL=0.0, wS=1.0)

    # Combined
    combined = run_portfolio(test.copy(), pL, pS, wL=wL, wS=wS, return_equity=args.export_equity)

    # Cap / feasibility
    cap = parse_cap_mode(args.cap_mode, rL["mdd"], rS["mdd"])
    feasible = (cap is None) or (combined["mdd"] <= cap)

    # -------- Benchmark section (new configurable modes) --------
    bench_info = {"mode": args.bench_mode}
    verdict = None
    delta = None

    if args.bench_mode == "fixed":
        bench_pct = float(args.bench_fixed_pct)
        bench_net_target = 10000.0 * (bench_pct / 100.0)
        delta = combined["net"] - bench_net_target
        verdict = "✅ meets target" if combined["net"] >= bench_net_target else "⬇️ below target"
        bench_info.update({
            "fixed_pct": bench_pct,
            "net_target": bench_net_target,
            "delta_vs_target": float(delta),
            "verdict": verdict
        })

    elif args.bench_mode == "band":
        lo, hi = parse_bench_band(args.bench_band)
        lo_abs = 10000.0 * (lo / 100.0)
        hi_abs = 10000.0 * (hi / 100.0)
        in_band = (combined["net"] >= lo_abs) and (combined["net"] <= hi_abs)
        verdict = "✅ within band" if in_band else ("⬆️ above band" if combined["net"] > hi_abs else "⬇️ below band")
        bench_info.update({
            "band_pct": {"min": lo, "max": hi},
            "band_abs": {"min": lo_abs, "max": hi_abs},
            "in_band": bool(in_band),
            "verdict": verdict
        })

    # Print summary
    print("\n=== YEAR CHECK (~{}d) — Conflict-Guard ===".format(args.days))
    print("Sources:", sources)
    print("\n— Separate legs (for cap reference) —")
    print(f"LONG  (10k) → Net {rL['net']:.2f}, PF {rL['pf']:.2f}, WR {rL['wr']:.2f}%, Trades {rL['trades']}, MDD {rL['mdd']:.2f}, FinalEq {rL['final_equity']:.2f}")
    print(f"SHORT (10k) → Net {rS['net']:.2f}, PF {rS['pf']:.2f}, WR {rS['wr']:.2f}%, Trades {rS['trades']}, MDD {rS['mdd']:.2f}, FinalEq {rS['final_equity']:.2f}")

    print("\n— Combined Conflict-Guard —")
    cap_str = f"{cap:.2f}" if cap is not None else "n/a"
    print(f"wL {wL:.2f}, wS {wS:.2f} | Net {combined['net']:.2f}, PF {combined['pf']:.2f}, WR {combined['wr']:.2f}%, Trades {combined['trades']}, MDD {combined['mdd']:.2f}, Score {combined['score']:.2f}, FinalEq {combined['final_equity']:.2f} | feasible={feasible} | cap={cap_str}")

    if args.bench_mode != "none":
        print("\n— Benchmark check —")
        if args.bench_mode == "fixed":
            print(f"Target: +{bench_info['fixed_pct']:.1f}% on $10,000 → Net target ${bench_info['net_target']:,.2f}")
            print(f"Result vs target → Δ Net: {bench_info['delta_vs_target']:,.2f} → {bench_info['verdict']}")
        else:
            lo = bench_info["band_pct"]["min"]; hi = bench_info["band_pct"]["max"]
            lo_abs = bench_info["band_abs"]["min"]; hi_abs = bench_info["band_abs"]["max"]
            print(f"Band: {lo:.1f}%–{hi:.1f}% on $10,000 → Net band ${lo_abs:,.2f}–${hi_abs:,.2f}")
            print(f"Result → {bench_info['verdict']}")

    # Export equity CSV
    stamp = now_stamp()
    exports = {}
    if args.export_equity and ("_equity_curve" in combined):
        eq_path = os.path.join(REPORT_DIR, f"annual_equity_{stamp}.csv")
        eq_series = pd.Series(combined["_equity_curve"], index=test.index[:len(combined["_equity_curve"])])
        eq_series.to_csv(eq_path, header=["equity"])
        exports["equity_csv"] = eq_path
        print(f"🧾 Saved combined equity curve → {eq_path}")

    # Save JSON summary (avoid non-serializable objects)
    summary = {
        "stamp_utc": to_iso(pd.Timestamp.utcnow()),
        "window": {"start": to_iso(test.index[0]), "end": to_iso(test.index[-1]), "bars": int(len(test)), "days": float(len(test)/6.0)},
        "weights": {"wL": float(wL), "wS": float(wS), "from_file_meta": meta},
        "legs": {
            "long": {k: (float(v) if isinstance(v, (int,float,np.floating)) else v) for k,v in rL.items()},
            "short":{k: (float(v) if isinstance(v, (int,float,np.floating)) else v) for k,v in rS.items()},
        },
        "combined": {k: (float(v) if isinstance(v, (int,float,np.floating)) else v) for k,v in combined.items() if not k.startswith("_")},
        "cap_mode": args.cap_mode,
        "cap_value": (None if cap is None else float(cap)),
        "feasible": bool(feasible),
        "benchmark": bench_info,
        "sources": sources
    }
    json_path = os.path.join(REPORT_DIR, f"annual_check_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📝 Saved annual check summary → {json_path}")

if __name__ == "__main__":
    main()