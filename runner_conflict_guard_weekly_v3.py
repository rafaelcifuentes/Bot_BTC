# runner_conflict_guard_weekly_v3.py — Conflict-Guard semanal (v3)
# Semillas ampliadas + micro-grid + regla híbrida de autolock
# Salva pesos en ./profiles/andromeda_conflict_guard_weights.json
# Reportes: ./reports/conflict_guard_weekly_YYYYMMDD_HHMMSS.json y ..._latest.json

import os, json, time, argparse, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import pandas_ta as ta

# ML
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -------- try ccxt/yfinance ----------
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
# Config / Parámetros
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

# Locks
LOCK_LONG  = "./profiles/profileA_binance_4h.json"
LOCK_SHORT = "./profiles/profileA_short_binance_4h.json"

# Dirs
SNAPSHOT_DIR = "./profiles"
REPORT_DIR   = "./reports"
CACHE_DIR    = "./cache"
CACHE_FILE   = os.path.join(CACHE_DIR, "btc_4h_ohlcv.csv")
WEIGHTS_FILE = "./profiles/andromeda_conflict_guard_weights.json"

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

def ts_ms(days_back: int) -> int:
    return int((pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)).timestamp() * 1000)

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
            time.sleep((ex.rateLimit or 200) / 1000.0) if hasattr(ex, "rateLimit") else None
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
# Backtests con ATR: long y short (idénticos a Andromeda, invertidos para short)
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

def run_portfolio(df: pd.DataFrame, pL: Dict, pS: Dict, wL: float, wS: float, cost=BASE_COST, slip=BASE_SLIP) -> Dict:
    # pesos escalan el risk_perc de cada pierna
    pL = pL.copy(); pS = pS.copy()
    pL["risk_perc"] *= float(wL)
    pS["risk_perc"] *= float(wS)

    equity_closed = 10000.0
    trades = []
    eq_curve = []
    pos = None

    for _, r in df.iterrows():
        # cierre de posición
        if pos is not None:
            if pos["dir"] == "long":
                pos["hm"] = max(pos["hm"], r["high"])
                exit_price = None
                if (not pos["p1"]) and (r["high"] >= pos["t1"]):
                    part_sz = pos["sz"] * pL["partial_pct"]
                    pnl = (pos["t1"] - pos["e"]) * part_sz
                    cost_fee = pos["t1"] * part_sz * cost
                    trades.append(pnl - cost_fee); equity_closed += (pnl - cost_fee)
                    pos["sz"] *= (1 - pL["partial_pct"]); pos["p1"] = True
                if r["high"] >= pos["t2"]: exit_price = pos["t2"]
                new_stop = pos["hm"] - pL["trail_mul"] * pos["atr0"]
                if new_stop > pos["s"]: pos["s"] = new_stop
                if r["low"] <= pos["s"]: exit_price = pos["s"]
                if exit_price is not None:
                    pnl = (exit_price - pos["e"]) * pos["sz"]
                    cost_fee = exit_price * pos["sz"] * cost
                    trades.append(pnl - cost_fee); equity_closed += (pnl - cost_fee); pos = None
            else:  # short
                low_mark = min(getattr(pos, "lm", pos["e"]), r["low"])
                pos["lm"] = low_mark
                exit_price = None
                if (not pos["p1"]) and (r["low"] <= pos["t1"]):
                    part_sz = pos["sz"] * pS["partial_pct"]
                    pnl = (pos["e"] - pos["t1"]) * part_sz
                    cost_fee = pos["t1"] * part_sz * cost
                    trades.append(pnl - cost_fee); equity_closed += (pnl - cost_fee)
                    pos["sz"] *= (1 - pS["partial_pct"]); pos["p1"] = True
                if r["low"] <= pos["t2"]: exit_price = pos["t2"]
                new_stop = pos["lm"] + pS["trail_mul"] * pos["atr0"]
                if new_stop < pos["s"]: pos["s"] = new_stop
                if r["high"] >= pos["s"]: exit_price = pos["s"]
                if exit_price is not None:
                    pnl = (pos["e"] - exit_price) * pos["sz"]
                    cost_fee = exit_price * pos["sz"] * cost
                    trades.append(pnl - cost_fee); equity_closed += (pnl - cost_fee); pos = None

        # conflict-guard: abrir solo si no hay posición
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
    score = float(net / (mdd + 1e-9)) if mdd > 0 else net
    return {"net": net, "pf": pf, "wr": wr, "trades": int(len(arr)), "mdd": mdd, "score": score}

# -----------------------
# Helpers de evaluación
# -----------------------
def cap_from_mode(rL: Dict, rS: Dict, mode: str) -> float:
    mode = (mode or "").strip().lower()
    if mode.startswith("min_leg"):
        # ejemplo "min_leg*1.05"
        mult = 1.0
        if "*" in mode:
            try:
                mult = float(mode.split("*")[1])
            except Exception:
                mult = 1.05
        return min(rL["mdd"], rS["mdd"]) * mult
    elif mode.startswith("max_leg"):
        mult = 1.0
        if "*" in mode:
            try:
                mult = float(mode.split("*")[1])
            except Exception:
                mult = 1.0
        return max(rL["mdd"], rS["mdd"]) * mult
    else:
        # número directo
        try:
            return float(mode)
        except Exception:
            return min(rL["mdd"], rS["mdd"]) * 1.05

def parse_seed_list(seed_str: str) -> List[Tuple[float,float]]:
    if not seed_str:
        return [(0.60,0.40),(0.15,0.40),(0.55,0.45),(0.50,0.50),(0.70,0.30)]
    out = []
    for pair in seed_str.split(";"):
        a,b = pair.split(",")
        out.append((float(a), float(b)))
    return out

def local_grid_around(wL: float, wS: float, step: float=0.05) -> List[Tuple[float,float]]:
    cands = []
    for dl in (-step, 0.0, +step):
        for ds in (-step, 0.0, +step):
            L = round(min(max(wL+dl, 0.0), 1.5), 2)  # permitimos >1.0 por si el usuario quiere over-weight (se evaluará por regla)
            S = round(min(max(wS+ds, 0.0), 1.5), 2)
            if L==0 and S==0: continue
            cands.append((L,S))
    # único
    uniq = list({(x,y) for (x,y) in cands})
    return sorted(uniq)

# -----------------------
# Runner
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Andromeda Conflict-Guard Weekly v3")
    ap.add_argument("--freeze-end", type=str, default=None, help='Cortar datos hasta esta fecha/hora UTC, ej: "2025-08-10 04:00"')
    ap.add_argument("--cap-mode", type=str, default="min_leg*1.05", help='Cap de MDD: "min_leg*1.05", "max_leg*1.10" o número')
    ap.add_argument("--net-tol", type=float, default=0.20, help="Tolerancia de caída de Net vs baseline (ej. 0.20 = 20%)")
    ap.add_argument("--seed-weights", type=str, default="", help='Semillas "wL,wS;wL,wS;..." (por defecto 5 semillas)')
    ap.add_argument("--grid-step", type=float, default=0.05, help="Paso del micro-grid alrededor de cada semilla")
    args = ap.parse_args()

    ensure_dirs()

    # Data
    print("🔄 Loading 4h data (ccxt + fallbacks + yfinance)…")
    df_raw, sources = load_4h_stitched(primary_only=False)
    if df_raw.empty:
        print("No data available."); return
    if args.freeze_end:
        t_end = pd.Timestamp(args.freeze_end, tz="UTC").tz_convert(None)
        df_raw = df_raw.loc[:t_end].copy()

    print("✨ Adding features…")
    df = add_features(df_raw).dropna()
    if len(df) < MIN_BARS_REQ:
        print(f"Not enough 4h bars ({len(df)} < {MIN_BARS_REQ})."); return

    # split 70/30 temporal
    cut = int(len(df) * 0.7)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    # modelo prob_up
    print("🧠 Training model on train / scoring OOS…")
    model = train_model(train)
    test["prob_up"] = predict_prob_up(model, test)

    # cargar locks
    pL = load_lock(LOCK_LONG,  FALLBACK_LONG)
    pS = load_lock(LOCK_SHORT, FALLBACK_SHORT)

    # métricas por pierna (solo OOS)
    soloL = test.copy()
    rL = run_portfolio(soloL, pL, pS, wL=1.0, wS=0.0)

    soloS = test.copy()
    rS = run_portfolio(soloS, pL, pS, wL=0.0, wS=1.0)

    # baseline = pesos activos actuales (si hay); si no, 0.60/0.40
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, "r") as f:
                stored = json.load(f)
            wL_base = float(stored.get("wL", 0.60))
            wS_base = float(stored.get("wS", 0.40))
        except Exception:
            wL_base, wS_base = 0.60, 0.40
    else:
        wL_base, wS_base = 0.60, 0.40

    base = run_portfolio(test.copy(), pL, pS, wL=wL_base, wS=wS_base)

    # cap de MDD
    mdd_cap = cap_from_mode(rL, rS, args.cap_mode)

    # semillas + micro-grid
    seeds = parse_seed_list(args.seed_weights)
    all_cands = []
    for (a,b) in seeds:
        all_cands.append((round(a,2), round(b,2)))
        all_cands.extend(local_grid_around(a,b, step=args.grid_step))
    # único y limpio
    all_cands = sorted(list({(round(x,2), round(y,2)) for (x,y) in all_cands}))

    best = None
    for (wL, wS) in all_cands:
        r = run_portfolio(test.copy(), pL, pS, wL, wS)
        feasible = (r["mdd"] <= mdd_cap)
        # Regla híbrida vs baseline
        score_up = (r["score"] > base["score"])
        net_ok   = (r["net"] >= base["net"] * (1 - args.net_tol))
        if feasible and score_up and net_ok:
            if (best is None) or (r["score"] > best["score"]):
                best = {"wL": wL, "wS": wS, **r, "feasible": True}

    print("\n— Portfolio with conflict-guard (OOS zone) —")
    print(f"Solo LONG  → Net {rL['net']:.2f}, PF {rL['pf']:.2f}, WR {rL['wr']:.2f}%, Trades {rL['trades']}, MDD {rL['mdd']:.2f}, Score {rL['score']:.2f}")
    print(f"Solo SHORT → Net {rS['net']:.2f}, PF {rS['pf']:.2f}, WR {rS['wr']:.2f}%, Trades {rS['trades']}, MDD {rS['mdd']:.2f}, Score {rS['score']:.2f}")
    print(f"Baseline   → wL {wL_base:.2f}, wS {wS_base:.2f} | Net {base['net']:.2f}, PF {base['pf']:.2f}, WR {base['wr']:.2f}%, Trades {base['trades']}, MDD {base['mdd']:.2f}, Score {base['score']:.2f}")

    if best is not None:
        print(f"Selected   → wL {best['wL']:.2f}, wS {best['wS']:.2f} | Net {best['net']:.2f}, PF {best['pf']:.2f}, WR {best['wr']:.2f}%, Trades {best['trades']}, MDD {best['mdd']:.2f}, Score {best['score']:.2f} | feasible={best['feasible']} | cap={mdd_cap:.2f}")
        # Guardar si mejora vs baseline (regla híbrida ya verificada)
        try:
            with open(WEIGHTS_FILE, "w") as f:
                json.dump({
                    "wL": best["wL"],
                    "wS": best["wS"],
                    "meta": {
                        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                        "cap_mode": args.cap_mode,
                        "net_tol": args.net_tol,
                        "baseline": {"wL": wL_base, "wS": wS_base, **base}
                    }
                }, f, indent=2)
            print(f"🔒 Saved new active weights → {WEIGHTS_FILE}")
        except Exception as e:
            print(f"⚠️ Could not save weights: {e}")
    else:
        print(f"↩︎ Autolock omitido (no cumple Score↑ & Net dentro tolerancia & MDD ≤ cap).")

    # Reporte
    stamp = now_stamp()
    weekly = {
        "stamp_utc": pd.Timestamp.utcnow().isoformat(),
        "sources": sources,
        "solo_long": rL, "solo_short": rS,
        "baseline": {"wL": wL_base, "wS": wS_base, **base},
        "selected": best if best is not None else {},
        "cap": mdd_cap,
        "params": {
            "cap_mode": args.cap_mode,
            "net_tol": args.net_tol,
            "seeds": seeds,
            "grid_step": args.grid_step,
            "freeze_end": args.freeze_end or ""
        }
    }
    path = os.path.join(REPORT_DIR, f"conflict_guard_weekly_{stamp}.json")
    with open(path, "w") as f:
        json.dump(weekly, f, indent=2)
    latest = os.path.join(REPORT_DIR, "conflict_guard_weekly_latest.json")
    with open(latest, "w") as f:
        json.dump(weekly, f, indent=2)
    print(f"📝 Saved weekly summary → {path}")
    print(f"📌 Updated latest → {latest}")

if __name__ == "__main__":
    main()