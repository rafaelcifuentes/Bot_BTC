# runner_andromeda_year_compare_v1.py
# Year (~360d) comparison: Separate bots (LONG  + SHORT) vs Conflict-Guard combined
# - Loads ccxt + fallbacks + yfinance, Andromeda v4.3 features, RandomForest prob_up
# - Trains ONLY on data BEFORE the 360d window; evaluates INSIDE the window
# - Reports PF, Net Profit, MDD, WR, Trades, Final Equity
# - Cap mode "min_leg*X" computed en porcentaje y convertido a monto para el combinado
# - Safe JSON (timestamps/numpy) + configurable starting equity
#
# Ejemplos (zsh: comillas si usas '*'):
#   ./.venv/bin/python runner_andromeda_year_compare_v1.py \
#     --freeze-end "2025-08-12 12:05" \
#     --wL 0.35 --wS 0.65 \
#     --mdd_cap_mode "min_leg*1.05" \
#     --eq0-combined 20000 --eq0-perbot 10000
#
#   ./.venv/bin/python runner_andromeda_year_compare_v1.py \
#     --freeze-end "2025-08-12 12:05" \
#     --wL 0.35 --wS 0.65 \
#     --mdd_cap_mode "none" \
#     --eq0-combined 10000 --eq0-perbot 10000

import os, json, time, warnings, argparse
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---- optional deps ----
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
# Config
# -----------------------
SYMBOL = "BTC/USDT"
YF_SYMBOL = "BTC-USD"
TIMEFRAME = "4h"
MAX_BARS = 3000
SINCE_DAYS = 900          # más historial para entrenar antes de la ventana anual
MIN_BARS_REQ = 1200       # mínimo de barras 4h para poder entrenar y testear

BASE_COST = 0.0002
BASE_SLIP = 0.0001
SEED = 42

# Locks
LOCK_LONG  = "./profiles/profileA_binance_4h.json"
LOCK_SHORT = "./profiles/profileA_short_binance_4h.json"

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

def now_stamp() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

def dedup_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.index.duplicated(keep="last")].sort_index()

def json_safe(o):
    # convierte tipos no serializables (Timestamp, numpy.*) a nativos JSON
    if isinstance(o, (pd.Timestamp, pd.Timedelta)):
        return str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o

def safe_pf(trades: np.ndarray) -> float:
    if trades.size == 0: return 0.0
    gains = trades[trades > 0].sum()
    losses = trades[trades < 0].sum()
    if losses == 0: return float("inf") if gains > 0 else 0.0
    return float(gains / abs(losses))

# -----------------------
# Data loading (ccxt + fallbacks + yf)
# -----------------------
EX_SYMBOLS_ALL = {
    "binance":   ["BTC/USDT"],
    "binanceus": ["BTC/USDT"],
    "kraken":    ["BTC/USDT", "XBT/USDT", "BTC/USD", "XBT/USD"],
    "bybit":     ["BTC/USDT"],
    "kucoin":    ["BTC/USDT"],
    "okx":       ["BTC/USDT"],
}

def ts_ms(days_back: int) -> int:
    return int((pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)).timestamp() * 1000)

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
            if len(all_rows) >= MAX_BARS + 1500:
                break
            time.sleep((ex.rateLimit or 200) / 1000.0) if hasattr(ex, "rateLimit") else None
        if all_rows:
            df = pd.DataFrame(all_rows, columns=["time","open","high","low","close","volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            df.set_index("time", inplace=True)
            df.index = df.index.tz_convert(None)
            return df
    return pd.DataFrame()

def fetch_yf(symbol: str, period_days: int = 900) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    days = max(365, min(730, period_days))
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
# Features + model (Andromeda core)
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
# Locks
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
# Trade steps (ATR, parciales, trailing)
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

# -----------------------
# Runs
# -----------------------
def run_one_leg(df: pd.DataFrame, p: Dict, eq0: float, direction: str, cost=BASE_COST, slip=BASE_SLIP) -> Dict:
    # direction: "long" o "short"
    p = p.copy()
    equity_closed = float(eq0)
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
                    part_sz = pos["sz"] * p["partial_pct"]
                    pnl = (pos["t1"] - pos["e"]) * part_sz
                    fee = pos["t1"] * part_sz * cost
                    trades.append(pnl - fee); equity_closed += (pnl - fee)
                    pos["sz"] *= (1 - p["partial_pct"]); pos["p1"] = True
                if r["high"] >= pos["t2"]: exit_price = pos["t2"]
                new_stop = pos["hm"] - p["trail_mul"] * pos["atr0"]
                if new_stop > pos["s"]: pos["s"] = new_stop
                if r["low"] <= pos["s"]: exit_price = pos["s"]
                if exit_price is not None:
                    pnl = (exit_price - pos["e"]) * pos["sz"]
                    fee = exit_price * pos["sz"] * cost
                    trades.append(pnl - fee); equity_closed += (pnl - fee); pos = None
            else:
                low_mark = min(getattr(pos, "lm", pos["e"]), r["low"])
                pos["lm"] = low_mark
                exit_price = None
                if (not pos["p1"]) and (r["low"] <= pos["t1"]):
                    part_sz = pos["sz"] * p["partial_pct"]
                    pnl = (pos["e"] - pos["t1"]) * part_sz
                    fee = pos["t1"] * part_sz * cost
                    trades.append(pnl - fee); equity_closed += (pnl - fee)
                    pos["sz"] *= (1 - p["partial_pct"]); pos["p1"] = True
                if r["low"] <= pos["t2"]: exit_price = pos["t2"]
                new_stop = pos["lm"] + p["trail_mul"] * pos["atr0"]
                if new_stop < pos["s"]: pos["s"] = new_stop
                if r["high"] >= pos["s"]: exit_price = pos["s"]
                if exit_price is not None:
                    pnl = (pos["e"] - exit_price) * pos["sz"]
                    fee = exit_price * pos["sz"] * cost
                    trades.append(pnl - fee); equity_closed += (pnl - fee); pos = None

        # apertura (solo entradas de la dirección pedida)
        if pos is None:
            if direction == "long":
                ok = (
                    (r["prob_up"] > p["threshold"])
                    and (r["adx4h"] >= p["adx4_min"]) and (r["adx1d"] >= p["adx1d_min"])
                    and pass_trend(r, p.get("trend_mode", "none"))
                    and r["close"] > r["ema_slow"]
                )
                if ok:
                    p["_equity"] = equity_closed; p["_trades"] = trades
                    pos = step_trade_long(r, p, SLIP=slip, COST=cost)
                    equity_closed = p["_equity"]
            else:  # short
                ok = (
                    (r["prob_up"] < (1 - p["threshold"]))
                    and (r["adx4h"] >= p["adx4_min"]) and (r["adx1d"] >= p["adx1d_min"])
                    and pass_trend(r, p.get("trend_mode", "none"))
                    and r["close"] < r["ema_slow"]
                )
                if ok:
                    p["_equity"] = equity_closed; p["_trades"] = trades
                    pos = step_trade_short(r, p, SLIP=slip, COST=cost)
                    equity_closed = p["_equity"]

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
    return {"net": net, "pf": pf, "wr": wr, "trades": int(len(arr)), "mdd": mdd, "final_eq": equity_closed}

def run_conflict_guard(df: pd.DataFrame, pL: Dict, pS: Dict, wL: float, wS: float, eq0: float,
                       cost=BASE_COST, slip=BASE_SLIP) -> Dict:
    # escala risk_perc por pesos
    pL = pL.copy(); pS = pS.copy()
    pL["risk_perc"] *= float(wL)
    pS["risk_perc"] *= float(wS)

    equity_closed = float(eq0)
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
                if r["high"] >= pos["t2"]: exit_price = pos["t2"]
                new_stop = pos["hm"] - pL["trail_mul"] * pos["atr0"]
                if new_stop > pos["s"]: pos["s"] = new_stop
                if r["low"] <= pos["s"]: exit_price = pos["s"]
                if exit_price is not None:
                    pnl = (exit_price - pos["e"]) * pos["sz"]
                    fee = exit_price * pos["sz"] * cost
                    trades.append(pnl - fee); equity_closed += (pnl - fee); pos = None
            else:
                low_mark = min(getattr(pos, "lm", pos["e"]), r["low"])
                pos["lm"] = low_mark
                exit_price = None
                if (not pos["p1"]) and (r["low"] <= pos["t1"]):
                    part_sz = pos["sz"] * pS["partial_pct"]
                    pnl = (pos["e"] - pos["t1"]) * part_sz
                    fee = pos["t1"] * part_sz * cost
                    trades.append(pnl - fee); equity_closed += (pnl - fee)
                    pos["sz"] *= (1 - pS["partial_pct"]); pos["p1"] = True
                if r["low"] <= pos["t2"]: exit_price = pos["t2"]
                new_stop = pos["lm"] + pS["trail_mul"] * pos["atr0"]
                if new_stop < pos["s"]: pos["s"] = new_stop
                if r["high"] >= pos["s"]: exit_price = pos["s"]
                if exit_price is not None:
                    pnl = (pos["e"] - exit_price) * pos["sz"]
                    fee = exit_price * pos["sz"] * cost
                    trades.append(pnl - fee); equity_closed += (pnl - fee); pos = None

        # apertura (conflict-guard: no abrir si hay posición viva)
        if pos is None:
            ok_long = (
                (r["prob_up"] > pL["threshold"])
                and (r["adx4h"] >= pL["adx4_min"]) and (r["adx1d"] >= pL["adx1d_min"])
                and pass_trend(r, pL.get("trend_mode", "none"))
                and r["close"] > r["ema_slow"]
            )
            ok_short = (
                (r["prob_up"] < (1 - pS["threshold"]))
                and (r["adx4h"] >= pS["adx4_min"]) and (r["adx1d"] >= pS["adx1d_min"])
                and pass_trend(r, pS.get("trend_mode", "none"))
                and r["close"] < r["ema_slow"]
            )
            if ok_long:
                pL["_equity"] = equity_closed; pL["_trades"] = trades
                pos = step_trade_long(r, pL, SLIP=slip, COST=cost)
                equity_closed = pL["_equity"]
            elif ok_short:
                pS["_equity"] = equity_closed; pS["_trades"] = trades
                pos = step_trade_short(r, pS, SLIP=slip, COST=cost)
                equity_closed = pS["_equity"]

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
    return {"net": net, "pf": pf, "wr": wr, "trades": int(len(arr)), "mdd": mdd, "final_eq": equity_closed}

# -----------------------
# Window split (year)
# -----------------------
def compute_year_window(df: pd.DataFrame, freeze_end: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    # toma últimas ~360d (2160 barras de 4h) hasta freeze_end (o fin del dataset)
    if freeze_end:
        ts_end = pd.Timestamp(freeze_end, tz="UTC").tz_convert(None)
        df = df.loc[:ts_end].copy()
    else:
        ts_end = df.index.max()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), None, None

    bars_year = int(360 * 24 / 4)  # 2160
    if len(df) < bars_year + 200:
        return pd.DataFrame(), pd.DataFrame(), None, None

    test = df.iloc[-bars_year:].copy()
    ts_start = test.index[0]
    train = df.loc[:ts_start - pd.Timedelta(seconds=1)].copy()
    return train, test, ts_start, ts_end

# -----------------------
# CLI
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Year compare: Separate vs Conflict-Guard combined (Andromeda v4.3 core)")
    ap.add_argument("--freeze-end", type=str, default=None, help='UTC "YYYY-MM-DD HH:MM" cutoff (inclusive)')
    ap.add_argument("--wL", type=float, default=0.60, help="Weight for LONG (scales risk_perc)")
    ap.add_argument("--wS", type=float, default=0.40, help="Weight for SHORT (scales risk_perc)")
    ap.add_argument("--mdd_cap_mode", type=str, default="min_leg*1.05", help='Cap mode: "none" or "min_leg*X" (quote in zsh)')
    ap.add_argument("--eq0-combined", type=float, default=20000.0, help="Starting equity for combined Conflict-Guard")
    ap.add_argument("--eq0-perbot", type=float, default=10000.0, help="Starting equity for each separate bot (LONG and SHORT)")
    return ap.parse_args()

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    ensure_dirs()

    # Load data
    df_raw, sources = load_4h_stitched(primary_only=False)
    if df_raw.empty:
        print("No data available."); return

    df = add_features(df_raw).dropna()
    if len(df) < MIN_BARS_REQ:
        print(f"Not enough 4h bars ({len(df)} < {MIN_BARS_REQ})."); return

    # Window
    train, test, ts_start, ts_end = compute_year_window(df, args.freeze_end)
    if train is None or test is None or test.empty or len(train) < 300:
        print("Year window split failed (insufficient data)."); return

    print(f"🔒 Year window: {ts_start} → {ts_end}  ({len(test)} bars ~ {len(test)/6:.1f} days)")
    print("🧠 Training model on PRE-window only…\n")

    # Model on PRE-window
    model = train_model(train)
    test = test.copy()
    test["prob_up"] = predict_prob_up(model, test)

    # Load locks
    pL = load_lock(LOCK_LONG,  FALLBACK_LONG)
    pS = load_lock(LOCK_SHORT, FALLBACK_SHORT)

    # Separate bots (each with its own equity)
    rL = run_one_leg(test.copy(), pL, eq0=args["eq0_perbot"] if isinstance(args, dict) else args.__dict__["eq0-perbot"] if False else args.__dict__["eq0-perbot"] if "eq0-perbot" in args.__dict__ else args.__dict__["eq0_perbot"] , direction="long")
    # The above line is messy in some editors; do it clean:
    eq0_perbot = getattr(args, "eq0_perbot")
    rL = run_one_leg(test.copy(), pL, eq0=eq0_perbot, direction="long")
    rS = run_one_leg(test.copy(), pS, eq0=eq0_perbot, direction="short")
    sep_net = rL["net"] + rS["net"]
    sep_final = eq0_perbot + rL["net"] + eq0_perbot + rS["net"]

    # Combined conflict-guard (single equity)
    eq0_combined = getattr(args, "eq0_combined")
    rC = run_conflict_guard(test.copy(), pL, pS, wL=args.wL, wS=args.wS, eq0=eq0_combined)

    # Cap mode (normalize MDDs to % and map to combined equity)
    cap_abs = None
    cap_pct = None
    feasible = True
    mode = (args.mdd_cap_mode or "none").strip().lower()
    if mode == "none":
        feasible = True
    else:
        # parse "min_leg*X"
        if mode.startswith("min_leg*"):
            try:
                mult = float(mode.split("*")[1])
            except Exception:
                mult = 1.05
            # MDD% en cada pierna (separados, mismo eq0_perbot)
            mdd_pct_L = rL["mdd"] / max(eq0_perbot, 1e-9)
            mdd_pct_S = rS["mdd"] / max(eq0_perbot, 1e-9)
            cap_pct = mult * min(mdd_pct_L, mdd_pct_S)
            cap_abs = cap_pct * eq0_combined
            feasible = (rC["mdd"] <= cap_abs)
        else:
            feasible = True

    # Print
    print("=== YEAR SIM (~360d) — Exact pre-train / in-window test ===")
    print("Sources:", sources)
    print("\n— Separate bots (independientes) —")
    print(f"LONG  ({eq0_perbot:,.0f}) → Net {rL['net']:.2f}, PF {rL['pf']:.2f}, WR {rL['wr']:.2f}%, Trades {rL['trades']}, MDD {rL['mdd']:.2f}, FinalEq {eq0_perbot + rL['net']:.2f}")
    print(f"SHORT ({eq0_perbot:,.0f}) → Net {rS['net']:.2f}, PF {rS['pf']:.2f}, WR {rS['wr']:.2f}%, Trades {rS['trades']}, MDD {rS['mdd']:.2f}, FinalEq {eq0_perbot + rS['net']:.2f}")
    print(f"SUMA dos bots → Net {sep_net:.2f}, FinalEq {sep_final:.2f}")

    print("\n— Conflict-guard combinado (un solo bot, {eq}) —".format(eq=f"{eq0_combined:,.0f}"))
    print(f"wL {args.wL:.2f}, wS {args.wS:.2f} | Net {rC['net']:.2f}, PF {rC['pf']:.2f}, WR {rC['wr']:.2f}%, Trades {rC['trades']}, MDD {rC['mdd']:.2f}, FinalEq {eq0_combined + rC['net']:.2f} | feasible={feasible} | cap={cap_abs:.2f if cap_abs is not None else float('nan')}")

    # Comparison
    delta_net = rC["net"] - sep_net
    delta_final = (eq0_combined + rC["net"]) - sep_final
    pref = "COMBINED (conflict-guard)" if delta_net > 0 else "SEPARATE (dos bots)"
    print("\n— Comparison (COMBINED vs SEPARATE) —")
    print(f"Δ Net Profit: {delta_net:+.2f}  |  Δ Final Equity: {delta_final:+.2f}")
    print(f"▶ Preferencia por Net Profit: {pref}")
    if cap_abs is not None:
        print(f"▶ Cap (% sobre combinado): {cap_pct*100:.2f}%  | MDD_combined {rC['mdd']:.2f} vs Cap_abs {cap_abs:.2f} → feasible={feasible}")

    # Save JSON
    stamp = now_stamp()
    out = {
        "stamp_utc": pd.Timestamp.utcnow().isoformat(),
        "window_start": str(ts_start),
        "window_end": str(ts_end),
        "eq0": {"combined": float(eq0_combined), "per_bot": float(eq0_perbot)},
        "weights": {"wL": float(args.wL), "wS": float(args.wS)},
        "long": {k: json_safe(v) for k,v in rL.items()},
        "short": {k: json_safe(v) for k,v in rS.items()},
        "separate_sum": {"net": json_safe(sep_net), "final_eq": json_safe(sep_final)},
        "combined": {k: json_safe(v) for k,v in rC.items()},
        "cap_mode": args.mdd_cap_mode,
        "cap_abs": json_safe(cap_abs),
        "cap_pct": json_safe(cap_pct),
        "feasible": bool(feasible),
        "comparison": {
            "delta_net": json_safe(delta_net),
            "delta_final_eq": json_safe(delta_final),
            "preference": pref
        },
        "sources": sources
    }
    path = os.path.join(REPORT_DIR, f"year_compare_{stamp}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=json_safe)
    print(f"\n📝 Saved year compare summary → {path}")

if __name__ == "__main__":
    main()