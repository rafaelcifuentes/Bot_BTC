# runner_andromeda_portfolio_v1.py
# Portafolio Andromeda LONG + SHORT
# - Reusa lógica de los runners v4.3 (loader ccxt+fallbacks, features, RF prob_up)
# - Reglas: sin posiciones contrapuestas; prioridad al mayor "edge" (distancia a umbral)
# - Sizing combinado: limita el riesgo por operación al máx(risk_perc_long, risk_perc_short)
# - Reporte unificado + métricas por pierna y combinadas (90d, 180d y holdout 75d)

import os, json, time
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

# ====== Dependencias externas ======
try:
    import ccxt
except Exception:
    ccxt = None
import pandas_ta as ta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    import yfinance as yf
except Exception:
    yf = None

# ====== Constantes comunes ======
SYMBOL = "BTC/USDT"
YF_SYMBOL = "BTC-USD"
TIMEFRAME = "4h"
SINCE_DAYS = 600
MAX_BARS = 3000
MIN_BARS_REQ = 800
SEED = 42
HOLDOUT_DAYS = 75

REPORT_DIR = "./reports"
CACHE_DIR = "./cache"
CACHE_FILE = os.path.join(CACHE_DIR, "btc_4h_ohlcv.csv")
LOCK_LONG = "./profiles/profileA_binance_4h.json"
LOCK_SHORT = "./profiles/profileA_short_binance_4h.json"

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 40)

# ====== Utilidades ======
def ensure_dirs():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def ts_ms(days_back: int) -> int:
    return int((pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)).timestamp() * 1000)

def dedup_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.index.duplicated(keep="last")].sort_index()

def now_stamp() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

def safe_pf(trades: np.ndarray) -> float:
    if trades.size == 0: return 0.0
    gains = trades[trades > 0].sum()
    losses = trades[trades < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / abs(losses))

def score_from(net: float, mdd: float) -> float:
    return float(net / (mdd + 1.0)) if (mdd is not None) else 0.0

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

# ====== Data loading (ccxt + yfinance + cache) ======
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
        ms_per_bar = ex.parse_timeframe(timeframe) * 1000
        next_since = since
        all_rows = []
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

# ====== Features & Model (idéntico base) ======
FEATURES = ["ema_fast","ema_slow","rsi","atr","adx4h"]

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

# ====== Locks ======
FALLBACK_LONG = {"threshold":0.46,"adx4_min":4,"adx1d_min":0,"trend_mode":"slope_up","sl_atr_mul":1.10,"tp1_atr_mul":1.60,"tp2_atr_mul":4.50,"trail_mul":0.80,"partial_pct":0.50,"risk_perc":0.007}
FALLBACK_SHORT={"threshold":0.46,"adx4_min":4,"adx1d_min":0,"trend_mode":"slope_down","sl_atr_mul":1.10,"tp1_atr_mul":1.60,"tp2_atr_mul":4.50,"trail_mul":0.80,"partial_pct":0.50,"risk_perc":0.005}

def load_locked(path: str, fallback: Dict) -> Dict:
    try:
        with open(path, "r") as f:
            p = json.load(f)
            return {k:v for k,v in p.items() if not isinstance(v, dict)}
    except Exception:
        return fallback.copy()

# ====== Reglas de entrada ======
def pass_trend(row: pd.Series, mode: str) -> bool:
    if mode == "none": return True
    if mode == "slope_up": return bool(row.get("slope_up",0)==1)
    if mode == "slope_down": return bool(row.get("slope_down",0)==1)
    if mode == "fast_or_slope":
        cond = (row.get("ema_fast",np.nan) > row.get("ema_slow",np.nan)) or (row.get("slope_up",0)==1)
        return bool(cond) and np.isfinite(row.get("ema_fast",np.nan))
    return True

# ====== Backtest combinado ======
def backtest_combo(df: pd.DataFrame, pL: Dict, pS: Dict,
                   cost: float=0.0002, slip: float=0.0001,
                   days: int = 0) -> Dict:
    """
    - No posiciones contrapuestas.
    - Si ambas señales disparan en la misma vela, se elige la que tenga mayor "edge":
        edge_long  = prob_up - pL['threshold']
        edge_short = (1 - prob_up) - pS['threshold']
    - Sizing: riesgo por operación = min( equity * risk_perc_leg, equity * max(risk_perc_long, risk_perc_short) )
    """
    data = df.tail(days*6).copy() if days>0 else df.copy()
    equity = 10000.0
    trades_L, trades_S, trades_all, eq_curve = [], [], [], []
    pos = None  # {"side": "L"/"S", "e": entry, "s": stop, "t1": tp1, "t2": tp2, "sz": size, "hm"/"lm": high/low mark, "p1": partial_done, "atr0": atr}

    risk_cap = max(pL["risk_perc"], pS["risk_perc"])

    for _, r in data.iterrows():
        # Señales elegibles por filtros
        can_long = (
            (r["prob_up"] > pL["threshold"]) and
            (r["adx4h"] >= pL["adx4_min"]) and
            (r["adx1d"] >= pL["adx1d_min"]) and
            pass_trend(r, pL.get("trend_mode","none")) and
            (r["close"] > r["ema_slow"])
        )
        can_short = (
            ((1.0 - r["prob_up"]) > pS["threshold"]) and
            (r["adx4h"] >= pS["adx4_min"]) and
            (r["adx1d"] >= pS["adx1d_min"]) and
            pass_trend(r, pS.get("trend_mode","none")) and
            (r["close"] < r["ema_slow"])
        )

        # Actualizar trailing si hay posición abierta
        if pos is not None:
            # LONG
            if pos["side"] == "L":
                pos["hm"] = max(pos["hm"], r["high"])
                exit_price = None

                # partial tp1
                if (not pos["p1"]) and (r["high"] >= pos["t1"]):
                    part_sz = pos["sz"] * pL["partial_pct"]
                    pnl = (pos["t1"] - pos["e"]) * part_sz
                    cost_fee = pos["t1"] * part_sz * cost
                    trades_L.append(pnl - cost_fee); trades_all.append(pnl - cost_fee)
                    equity += (pnl - cost_fee)
                    pos["sz"] *= (1 - pL["partial_pct"])
                    pos["p1"] = True

                # tp2
                if r["high"] >= pos["t2"]:
                    exit_price = pos["t2"]

                # trailing
                new_stop = pos["hm"] - pL["trail_mul"] * pos["atr0"]
                if new_stop > pos["s"]:
                    pos["s"] = new_stop
                if r["low"] <= pos["s"]:
                    exit_price = pos["s"]

                if exit_price is not None:
                    pnl = (exit_price - pos["e"]) * pos["sz"]
                    cost_fee = exit_price * pos["sz"] * cost
                    trades_L.append(pnl - cost_fee); trades_all.append(pnl - cost_fee)
                    equity += (pnl - cost_fee)
                    pos = None

            # SHORT
            else:
                pos["lm"] = min(pos["lm"], r["low"])
                exit_price = None

                # partial tp1
                if (not pos["p1"]) and (r["low"] <= pos["t1"]):
                    part_sz = pos["sz"] * pS["partial_pct"]
                    pnl = (pos["e"] - pos["t1"]) * part_sz
                    cost_fee = pos["t1"] * part_sz * cost
                    trades_S.append(pnl - cost_fee); trades_all.append(pnl - cost_fee)
                    equity += (pnl - cost_fee)
                    pos["sz"] *= (1 - pS["partial_pct"])
                    pos["p1"] = True

                # tp2
                if r["low"] <= pos["t2"]:
                    exit_price = pos["t2"]

                # trailing
                new_stop = pos["lm"] + pS["trail_mul"] * pos["atr0"]
                if new_stop < pos["s"]:
                    pos["s"] = new_stop
                if r["high"] >= pos["s"]:
                    exit_price = pos["s"]

                if exit_price is not None:
                    pnl = (pos["e"] - exit_price) * pos["sz"]
                    cost_fee = exit_price * pos["sz"] * cost
                    trades_S.append(pnl - cost_fee); trades_all.append(pnl - cost_fee)
                    equity += (pnl - cost_fee)
                    pos = None

        # Si no hay posición, evaluar entradas (sin contrapuestas)
        if pos is None:
            # conflicto: ambas señales
            chosen = None
            if can_long and can_short:
                edge_L = r["prob_up"] - pL["threshold"]
                edge_S = (1.0 - r["prob_up"]) - pS["threshold"]
                chosen = "L" if edge_L >= edge_S else "S"
            elif can_long:
                chosen = "L"
            elif can_short:
                chosen = "S"

            if chosen is not None and np.isfinite(r["atr"]) and r["atr"] > 0:
                if chosen == "L":
                    entry = float(r["open"]) * (1 + slip)
                    stop  = entry - pL["sl_atr_mul"] * r["atr"]
                    if stop < entry:
                        # sizing to risk cap
                        risk_dollar = min(equity * pL["risk_perc"], equity * risk_cap)
                        size = risk_dollar / max(entry - stop, 1e-9)
                        # entry fee
                        equity += -(entry * size * cost)
                        trades_all.append(-(entry * size * cost))
                        trades_L.append(-(entry * size * cost))
                        pos = {"side":"L","e":entry,"s":stop,"t1":entry + pL["tp1_atr_mul"]*r["atr"],
                               "t2":entry + pL["tp2_atr_mul"]*r["atr"],"sz":size,"hm":entry,"p1":False,"atr0":r["atr"]}
                else:
                    entry = float(r["open"]) * (1 - slip)
                    stop  = entry + pS["sl_atr_mul"] * r["atr"]
                    if stop > entry:
                        risk_dollar = min(equity * pS["risk_perc"], equity * risk_cap)
                        size = risk_dollar / max(stop - entry, 1e-9)
                        equity += -(entry * size * cost)
                        trades_all.append(-(entry * size * cost))
                        trades_S.append(-(entry * size * cost))
                        pos = {"side":"S","e":entry,"s":stop,"t1":entry - pS["tp1_atr_mul"]*r["atr"],
                               "t2":entry - pS["tp2_atr_mul"]*r["atr"],"sz":size,"lm":entry,"p1":False,"atr0":r["atr"]}

        unreal = 0.0
        if pos is not None:
            if pos["side"] == "L":
                unreal = (r["close"] - pos["e"]) * pos["sz"]
            else:
                unreal = (pos["e"] - r["close"]) * pos["sz"]
        eq_curve.append(equity + unreal)

    # Métricas
    arrL, arrS, arrA = np.array(trades_L,dtype=float), np.array(trades_S,dtype=float), np.array(trades_all,dtype=float)
    def stats(arr):
        if arr.size == 0:
            return {"net":0.0,"pf":0.0,"win_rate":0.0,"trades":0}
        net = float(arr.sum())
        pf = safe_pf(arr)
        wr = float((arr > 0).sum() / arr.size * 100.0)
        return {"net":net,"pf":pf,"win_rate":wr,"trades":int(arr.size)}
    stL, stS, stA = stats(arrL), stats(arrS), stats(arrA)
    eq = np.array(eq_curve, dtype=float)
    mdd = float((np.maximum.accumulate(eq) - eq).max()) if eq.size else 0.0
    stA["mdd"] = mdd
    stA["score"] = score_from(stA["net"], mdd)
    return {"long":stL, "short":stS, "combo":stA}

# ====== Pipeline principal ======
def train_attach_probs(df: pd.DataFrame) -> pd.DataFrame:
    # 70/30 temporal por coherencia con los runners
    cut = int(len(df) * 0.7)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    if len(train) < 200 or len(test) < 50:
        return pd.DataFrame()
    model = train_model(train)
    df = df.copy()
    df.loc[test.index, "prob_up"] = predict_prob_up(model, test)
    # Relleno prudente para train con neutro (no se usa para OOS métricas)
    df["prob_up"] = df["prob_up"].fillna(0.5)
    return df

def slice_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    return df.tail(days*6).copy()

def main():
    print("🔄 Loading 4h data (ccxt + fallbacks + yfinance)…")
    df_raw, sources = load_4h_stitched(primary_only=False)
    print("Sources:", sources)
    if df_raw.empty:
        print("No data."); return

    params_L = load_locked(LOCK_LONG, FALLBACK_LONG)
    params_S = load_locked(LOCK_SHORT, FALLBACK_SHORT)

    print("✨ Adding features…")
    df = add_features(df_raw).dropna()
    if len(df) < MIN_BARS_REQ:
        print(f"Not enough bars ({len(df)}<{MIN_BARS_REQ})."); return

    print("🧠 Training model on 70% / scoring 30% OOS…")
    df_scored = train_attach_probs(df)
    if df_scored.empty:
        print("Split too small."); return

    print("\n— Portfolio metrics (OOS zone only) —")
    # 90d
    r90 = backtest_combo(slice_days(df_scored, 90), params_L, params_S)
    # 180d
    r180 = backtest_combo(slice_days(df_scored, 180), params_L, params_S)
    # Holdout 75d (idéntico criterio)
    rH = backtest_combo(slice_days(df_scored, HOLDOUT_DAYS), params_L, params_S)

    def pr(name, R):
        L,S,C = R["long"],R["short"],R["combo"]
        print(f"{name:>8}  LONG → Net {L['net']:.2f}, PF {L['pf']:.2f}, WR {L['win_rate']:.2f}%, Trades {L['trades']}")
        print(f"{'':>8} SHORT → Net {S['net']:.2f}, PF {S['pf']:.2f}, WR {S['win_rate']:.2f}%, Trades {S['trades']}")
        print(f"{'':>8} COMBO → Net {C['net']:.2f}, PF {C['pf']:.2f}, WR {C['win_rate']:.2f}%, Trades {C['trades']}, MDD {C.get('mdd',0.0):.2f}, Score {C.get('score',0.0):.2f}")
    pr("90d", r90)
    pr("180d", r180)
    pr(f"Hold{HOLDOUT_DAYS}d", rH)

    # Guardar JSON de reporte
    ensure_dirs()
    stamp = now_stamp()
    out_json = os.path.join(REPORT_DIR, f"portfolio_summary_{stamp}.json")
    payload = {
        "stamp_utc": pd.Timestamp.utcnow().isoformat(),
        "sources": sources,
        "locks": {"long": params_L, "short": params_S},
        "metrics": {
            "d90": r90, "d180": r180, f"holdout_{HOLDOUT_DAYS}d": rH
        }
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"📝 Saved portfolio summary → {out_json}")

if __name__ == "__main__":
    import time
    main()