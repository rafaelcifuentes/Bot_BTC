# runner_conflict_guard_weekly_v2.py
# Conflict-Guard (LONG+SHORT) semanal con:
#  - Baseline anual FIJA (no se toca): wL=0.60 / wS=0.40
#  - Micro-grid fina de pesos alrededor de semillas (incluye baseline)
#  - Regla híbrida de autolock: Score↑ y Net >= (1 - net_tol)*Net_base, y MDD <= cap
#  - Reportes a ./reports y, si se autolockea, guarda a ./profiles/andromeda_conflict_guard_weights.json
#
# Requiere: ccxt, pandas, pandas_ta, numpy, scikit-learn, yfinance

import os, json, time, argparse, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import pandas_ta as ta
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

# Locks de estrategias base (no se tocan aquí)
LOCK_LONG  = "./profiles/profileA_binance_4h.json"
LOCK_SHORT = "./profiles/profileA_short_binance_4h.json"

# Pesos baseline ANUAL (FIJOS, referencia; NO se autolockean)
BASELINE_WL = 0.60
BASELINE_WS = 0.40
BASELINE_FILE = "./profiles/andromeda_conflict_guard_baseline.json"

# Pesos activos semanales (si la regla híbrida se cumple)
ACTIVE_WEIGHTS_FILE = "./profiles/andromeda_conflict_guard_weights.json"

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
        since = ts_ms(SINCE_DAYS)
        all_rows = []
        ms_per_bar = ex.parse_timeframe(TIMEFRAME) * 1000
        next_since = since
        while True:
            try:
                ohlcv = ex.fetch_ohlcv(sym, timeframe=TIMEFRAME, since=next_since, limit=limit_step)
            except Exception:
                break
            if not ohlcv:
                break
            all_rows.extend(ohlcv)
            next_since = ohlcv[-1][0] + ms_per_bar
            if len(all_rows) >= MAX_BARS + 1200:
                break
            if hasattr(ex, "rateLimit"):
                time.sleep((ex.rateLimit or 200) / 1000.0)
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
# Backtests con ATR (idénticos a Andromeda; short invertido)
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
    pL = pL.copy(); pS = pS.copy()
    pL["risk_perc"] *= float(wL)
    pS["risk_perc"] *= float(wS)
    equity_closed, trades, eq_curve = 10000.0, [], []
    pos = None

    for _, r in df.iterrows():
        # gestionar posición abierta
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
            else:  # short
                pos["lm"] = min(pos.get("lm", pos["e"]), r["low"])
                exit_price = None
                if (not pos["p1"]) and (r["low"] <= pos["t1"]):
                    part_sz = pos["sz"] * pS["partial_pct"]
                    pnl = (pos["e"] - pos["t1"]) * part_sz
                    fee = pos["t1"] * part_sz * cost
                    trades.append(pnl - fee); equity_closed += (pnl - fee)
                    pos["sz"] *= (1 - pS["partial_pct"]); pos["p1"] = True
                if r["low"] <= pos["t2"]: exit_price = pos["t2"]
                new_stop = pos["lm"] + pS["trail_mul"] * pos["atr0"]
                if new_stop < pos["s"] : pos["s"] = new_stop
                if r["high"] >= pos["s"]: exit_price = pos["s"]
                if exit_price is not None:
                    pnl = (pos["e"] - exit_price) * pos["sz"]
                    fee = exit_price * pos["sz"] * cost
                    trades.append(pnl - fee); equity_closed += (pnl - fee); pos = None

        # apertura (conflict-guard: no abrir si hay posición)
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
            unreal = (r["close"] - pos["e"]) * pos["sz"] if pos["dir"]=="long" else (pos["e"] - r["close"]) * pos["sz"]
        eq_curve.append(equity_closed + unreal)

    arr = np.array(trades, dtype=float)
    net = float(arr.sum()) if arr.size else 0.0
    pf  = safe_pf(arr) if arr.size else 0.0
    wr  = float((arr > 0).sum() / len(arr) * 100.0) if arr.size else 0.0
    eq  = np.array(eq_curve, dtype=float)
    mdd = float((np.maximum.accumulate(eq) - eq).max()) if eq.size else 0.0
    score = float(net / (mdd + 1e-9)) if mdd > 0 else net
    return {"net": net, "pf": pf, "wr": wr, "trades": int(len(arr)), "mdd": mdd, "score": score}

# -----------------------
# CLI
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Conflict-Guard weekly v2 (hybrid autolock + fine micro-grid)")
    ap.add_argument("--primary-only", action="store_true", help="Usar solo Binance en el loader")
    ap.add_argument("--freeze-end", type=str, default=None, help='Cortar datos hasta esta fecha/hora UTC "YYYY-MM-DD HH:MM"')
    ap.add_argument("--net-tol", type=float, default=0.20, help="Tolerancia de caída de Net vs baseline para permitir autolock (0.20 = -20%)")
    ap.add_argument("--cap-mode", type=str, default="min_leg*1.05", choices=["min_leg","min_leg*1.05","min_leg*1.10","none"],
                    help="MDD cap para pesos seleccionados")
    ap.add_argument("--seed-weights", type=str, default="0.60,0.40;0.15,0.40",
                    help="Semillas de (wL,wS) separadas por ';'. Ej: '0.60,0.40;0.15,0.40'")
    return ap.parse_args()

def mdd_cap_from(mode: str, mdd_long: float, mdd_short: float) -> float:
    base = min(mdd_long, mdd_short)
    if mode == "min_leg": return base
    if mode == "min_leg*1.05": return base * 1.05
    if mode == "min_leg*1.10": return base * 1.10
    return float("inf")

def expand_grid_around(seeds: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    # micro-grid fina alrededor de cada semilla (±0.05 y ±0.10), recortando a [0.10, 1.00]
    cand = set()
    steps = [0.0, -0.10, -0.05, +0.05, +0.10]
    for wl, ws in seeds:
        for dwl in steps:
            for dws in steps:
                wL = max(0.10, min(1.00, round(wl + dwl, 2)))
                wS = max(0.10, min(1.00, round(ws + dws, 2)))
                cand.add((wL, wS))
    # asegurar baseline presente
    cand.add((BASELINE_WL, BASELINE_WS))
    return sorted(list(cand))

# -----------------------
# Runner semanal
# -----------------------
def main():
    ensure_dirs()

    # Baseline anual FIJO (snapshot si no existe)
    if not os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "w") as f:
            json.dump({"wL": BASELINE_WL, "wS": BASELINE_WS,
                       "_meta": {"note": "baseline anual FIJO", "ts": pd.Timestamp.utcnow().isoformat()}}, f, indent=2)

    args = parse_args()

    print("🔄 Loading 4h data (ccxt + fallbacks + yfinance)…")
    df_raw, sources = load_4h_stitched(primary_only=args.primary_only)
    if df_raw.empty:
        print("No data available."); return
    if args.freeze_end:
        t_end = pd.Timestamp(args.freeze_end, tz="UTC").tz_convert(None)
        df_raw = df_raw.loc[:t_end].copy()

    print("✨ Adding features…")
    df = add_features(df_raw).dropna()
    if len(df) < MIN_BARS_REQ:
        print(f"Not enough 4h bars ({len(df)} < {MIN_BARS_REQ})."); return

    cut = int(len(df) * 0.7)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    print("🧠 Training model on train / scoring OOS…")
    model = train_model(train)
    test["prob_up"] = predict_prob_up(model, test)

    pL = load_lock(LOCK_LONG,  FALLBACK_LONG)
    pS = load_lock(LOCK_SHORT, FALLBACK_SHORT)

    # Métricas por pierna (OOS)
    rL = run_portfolio(test.copy(), pL, pS, wL=1.0, wS=0.0)
    rS = run_portfolio(test.copy(), pL, pS, wL=0.0, wS=1.0)

    # Baseline (ANUAL FIJO)
    base = run_portfolio(test.copy(), pL, pS, wL=BASELINE_WL, wS=BASELINE_WS)

    # MDD cap
    cap = mdd_cap_from(args.cap_mode, rL["mdd"], rS["mdd"])

    # Semillas + micro-grid fina
    try:
        seeds = []
        for chunk in args.seed_weights.split(";"):
            wl, ws = chunk.split(",")
            seeds.append((float(wl), float(ws)))
    except Exception:
        seeds = [(BASELINE_WL, BASELINE_WS), (0.15, 0.40)]
    grid = expand_grid_around(seeds)

    # Buscar mejor factible (score máximo bajo cap)
    best = None
    for (wL, wS) in grid:
        r = run_portfolio(test.copy(), pL, pS, wL=wL, wS=wS)
        feasible = (r["mdd"] <= cap)
        if feasible and (best is None or r["score"] > best["score"]):
            best = {"wL": wL, "wS": wS, **r, "feasible": True}

    # Si no hay factible, quedarse con baseline
    if best is None:
        best = {"wL": BASELINE_WL, "wS": BASELINE_WS, **base, "feasible": False}

    # Regla híbrida de autolock:
    improve_score = best["score"] > base["score"]
    net_ok = best["net"] >= base["net"] * (1 - args.net_tol)
    mdd_ok = best["mdd"] <= cap
    do_autolock = (best["feasible"] and improve_score and net_ok and mdd_ok)

    # Print resumen
    print("\n— Portfolio with conflict-guard (OOS zone) —")
    print(f"Solo LONG  → Net {rL['net']:.2f}, PF {rL['pf']:.2f}, WR {rL['wr']:.2f}%, Trades {rL['trades']}, MDD {rL['mdd']:.2f}, Score {rL['score']:.2f}")
    print(f"Solo SHORT → Net {rS['net']:.2f}, PF {rS['pf']:.2f}, WR {rS['wr']:.2f}%, Trades {rS['trades']}, MDD {rS['mdd']:.2f}, Score {rS['score']:.2f}")
    print(f"Baseline   → wL {BASELINE_WL:.2f}, wS {BASELINE_WS:.2f} | Net {base['net']:.2f}, PF {base['pf']:.2f}, WR {base['wr']:.2f}%, Trades {base['trades']}, MDD {base['mdd']:.2f}, Score {base['score']:.2f}")
    print(f"Selected   → wL {best['wL']:.2f}, wS {best['wS']:.2f} | Net {best['net']:.2f}, PF {best['pf']:.2f}, WR {best['wr']:.2f}%, Trades {best['trades']}, MDD {best['mdd']:.2f}, Score {best['score']:.2f} | feasible={best['feasible']} | cap={cap:.2f}")

    # Autolock
    if do_autolock:
        with open(ACTIVE_WEIGHTS_FILE, "w") as f:
            json.dump({"wL": best["wL"], "wS": best["wS"],
                       "_meta": {
                           "note": "weights autolocked by weekly v2 (hybrid rule)",
                           "cap_mode": args.cap_mode,
                           "net_tol": args.net_tol,
                           "ts": pd.Timestamp.utcnow().isoformat(),
                           "sources": sources
                       }}, f, indent=2)
        print("🔒 Autolock aplicado (regla híbrida cumplida).")
    else:
        print("↩︎ Autolock omitido (no cumple Score↑ & Net dentro tolerancia & MDD ≤ cap).")

    # Reporte JSON
    stamp = now_stamp()
    summary = {
        "stamp_utc": pd.Timestamp.utcnow().isoformat(),
        "sources": sources,
        "cap_mode": args.cap_mode,
        "cap_value": cap,
        "baseline": {"wL": BASELINE_WL, "wS": BASELINE_WS, **base},
        "long_leg": rL,
        "short_leg": rS,
        "selected": best,
        "autolock": {
            "applied": do_autolock,
            "reasons": {"improve_score": improve_score, "net_ok": net_ok, "mdd_ok": mdd_ok},
            "net_tol": args.net_tol
        }
    }
    out_path = os.path.join(REPORT_DIR, f"conflict_guard_weekly_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(REPORT_DIR, "conflict_guard_weekly_latest.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📝 Saved weekly summary → {out_path}")
    print(f"📌 Updated latest → ./reports/conflict_guard_weekly_latest.json")

if __name__ == "__main__":
    main()