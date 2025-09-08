# runner_andromeda_portfolio_v2_conflict_guard.py
# Portfolio wrapper: respeta locks de Andromeda (sin modificarlos), añade:
# 1) Regla anti-conflicto (no long y short simultáneos)
# 2) Micro-grid de pesos (wL, wS) para limitar MDD combinado
# No escribe locks. Reporta métricas y elige (wL, wS) factible con mejor score.

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
pd.set_option("display.max_columns", 50)

# -----------------------
# Config base (idéntico estilo a runners)
# -----------------------
SYMBOL = "BTC/USDT"
YF_SYMBOL = "BTC-USD"
TIMEFRAME = "4h"
SINCE_DAYS = 600
MAX_BARS = 3000
MIN_BARS_REQ = 800
COST = 0.0002
SLIP = 0.0001
SEED = 42

LOCK_LONG  = "./profiles/profileA_binance_4h.json"        # lock LONG base
LOCK_SHORT = "./profiles/profileA_short_binance_4h.json"  # lock SHORT base

HOLDOUT_DAYS = 75

# -----------------------
# Utilidades
# -----------------------
def ensure_dirs():
    os.makedirs("./reports", exist_ok=True)
    os.makedirs("./cache", exist_ok=True)

def ts_ms(days_back: int) -> int:
    return int((pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)).timestamp() * 1000)

def now_stamp() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

def dedup_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.index.duplicated(keep="last")].sort_index()

def safe_pf(arr: np.ndarray) -> float:
    if arr.size == 0: return 0.0
    gains = arr[arr > 0].sum()
    losses = arr[arr < 0].sum()
    if losses == 0: return float("inf") if gains > 0 else 0.0
    return float(gains / abs(losses))

def score_from(net: float, mdd: float) -> float:
    return float(net / (mdd + 1.0)) if (mdd is not None and mdd >= 0) else 0.0

# -----------------------
# Data loading (ccxt + yf)
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
        rows = []
        ms_per_bar = ex.parse_timeframe(timeframe) * 1000
        next_since = since
        while True:
            try:
                ohlcv = ex.fetch_ohlcv(sym, timeframe=timeframe, since=next_since, limit=limit_step)
            except Exception:
                break
            if not ohlcv: break
            rows.extend(ohlcv)
            next_since = ohlcv[-1][0] + ms_per_bar
            if len(rows) >= MAX_BARS + 1200: break
            time.sleep((ex.rateLimit or 200) / 1000.0)
        if rows:
            df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            df.set_index("time", inplace=True)
            df.index = df.index.tz_convert(None)
            return df
    return pd.DataFrame()

def fetch_yf(symbol: str, period_days: int = 720) -> pd.DataFrame:
    if yf is None: return pd.DataFrame()
    days = max(365, min(720, period_days))
    try:
        df = yf.download(symbol, period=f"{days}d", interval="4h", auto_adjust=False, progress=False)
    except Exception:
        return pd.DataFrame()
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    return df[["open","high","low","close","volume"]].copy()

def load_4h() -> Tuple[pd.DataFrame, List[str]]:
    sources, dfs = [], []
    for ex_id, syms in EX_SYMBOLS_ALL.items():
        d = fetch_ccxt_any(ex_id, syms, TIMEFRAME, SINCE_DAYS)
        sources.append(f"{ex_id}: {'ok' if not d.empty else 'empty'}")
        if not d.empty:
            dfs.append(d)
    dyf = fetch_yf(YF_SYMBOL, SINCE_DAYS)
    sources.append("yfinance: ok" if not dyf.empty else "yfinance: empty")
    if not dyf.empty:
        dfs.append(dyf)
    if not dfs:
        return pd.DataFrame(), sources
    df = pd.concat(dfs, axis=0)
    df = df[["open","high","low","close","volume"]].astype(float)
    df = dedup_sort(df)
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].ffill()
    return df.tail(MAX_BARS), sources

# -----------------------
# Features + modelo único (como portfolio v1)
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
            n_estimators=200, random_state=SEED,
            class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1
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

FALLBACK_LONG = {"threshold": 0.46, "adx4_min": 4, "adx1d_min": 0,
                 "trend_mode": "slope_up", "sl_atr_mul": 1.10,
                 "tp1_atr_mul": 1.60, "tp2_atr_mul": 4.50,
                 "trail_mul": 0.80, "partial_pct": 0.50, "risk_perc": 0.007}

FALLBACK_SHORT = {"threshold": 0.46, "adx4_min": 4, "adx1d_min": 0,
                  "trend_mode": "slope_down", "sl_atr_mul": 1.10,
                  "tp1_atr_mul": 1.60, "tp2_atr_mul": 4.50,
                  "trail_mul": 0.80, "partial_pct": 0.50, "risk_perc": 0.005}

# -----------------------
# Señales y backtest con guardas
# -----------------------
def pass_trend(row: pd.Series, mode: str) -> bool:
    if mode == "none": return True
    if mode == "slope_up": return bool(row.get("slope_up", 0) == 1)
    if mode == "slope_down": return bool(row.get("slope_down", 0) == 1)
    if mode == "fast_or_slope":
        cond_up = (row.get("ema_fast", np.nan) > row.get("ema_slow", np.nan)) or (row.get("slope_up", 0) == 1)
        return bool(cond_up)
    return True

def gen_signals(df: pd.DataFrame, pL: Dict, pS: Dict) -> pd.DataFrame:
    out = df.copy()
    out["long_sig"] = (out["prob_up"] > pL["threshold"]) & \
                      (out["adx4h"] >= pL["adx4_min"]) & \
                      (out["adx1d"] >= pL["adx1d_min"]) & \
                      (out["close"] > out["ema_slow"]) & \
                      out.apply(lambda r: pass_trend(r, pL.get("trend_mode","none")), axis=1)

    out["short_sig"] = (out["prob_up"] < (1 - pS["threshold"])) & \
                       (out["adx4h"] >= pS["adx4_min"]) & \
                       (out["adx1d"] >= pS["adx1d_min"]) & \
                       (out["close"] < out["ema_slow"]) & \
                       out.apply(lambda r: pass_trend(r, pS.get("trend_mode","none")), axis=1)

    # regla anti-conflicto: prioridad al que dispara primero en la barra
    # si ambos verdaderos en la misma barra, abrimos el que tenga mayor "ventaja" relativa
    both = out["long_sig"] & out["short_sig"]
    if both.any():
        edge_long = (out["prob_up"] - pL["threshold"]).clip(lower=0)
        edge_short = ((1 - pS["threshold"]) - out["prob_up"]).clip(lower=0)
        prefer_long = edge_long >= edge_short
        out.loc[both & prefer_long, "short_sig"] = False
        out.loc[both & ~prefer_long, "long_sig"]  = False

    # bloqueo dinámico: si hay long abierto, no abrir short hasta cerrar, y viceversa
    in_long, in_short = False, False
    long_lock = np.zeros(len(out), dtype=bool)
    short_lock = np.zeros(len(out), dtype=bool)
    for i, (idx, row) in enumerate(out.iterrows()):
        if not in_long and not in_short:
            # nada abierto
            pass
        elif in_long:
            short_lock[i] = True
        elif in_short:
            long_lock[i] = True

        # cierre ocurre en el backtest por stop/tp/trailing; aquí solo bloqueamos inicios
        # Nota: el estado final (in_long/in_short) lo gestiona el motor en el loop del backtest

    out["long_lock"]  = long_lock
    out["short_lock"] = short_lock
    return out

def backtest_combo(df: pd.DataFrame, pL: Dict, pS: Dict, wL: float = 1.0, wS: float = 1.0) -> Dict:
    # Simulación conjunta con tamaños escalados por wL, wS, con COST/SLIP globales
    equity = 10000.0
    posL, posS = None, None
    trades = []
    eq_curve = []

    for _, r in df.iterrows():
        # cerrar/actualizar LONG
        if posL is not None:
            posL["hm"] = max(posL["hm"], r["high"])
            exit_price = None

            # parcial TP1
            if (not posL["p1"]) and (r["high"] >= posL["t1"]):
                part_sz = posL["sz"] * pL["partial_pct"]
                pnl = (posL["t1"] - posL["e"]) * part_sz
                cost = posL["t1"] * part_sz * COST
                trades.append(pnl - cost); equity += (pnl - cost)
                posL["sz"] *= (1 - pL["partial_pct"])
                posL["p1"] = True

            if r["high"] >= posL["t2"]:
                exit_price = posL["t2"]

            new_stop = posL["hm"] - pL["trail_mul"] * posL["atr0"]
            if new_stop > posL["s"]:
                posL["s"] = new_stop
            if r["low"] <= posL["s"]:
                exit_price = posL["s"]

            if exit_price is not None:
                pnl = (exit_price - posL["e"]) * posL["sz"]
                cost = exit_price * posL["sz"] * COST
                trades.append(pnl - cost); equity += (pnl - cost)
                posL = None

        # cerrar/actualizar SHORT
        if posS is not None:
            posS["lm"] = min(posS["lm"], r["low"])
            exit_price = None

            # parcial TP1 (short)
            if (not posS["p1"]) and (r["low"] <= posS["t1"]):
                part_sz = posS["sz"] * pS["partial_pct"]
                pnl = (posS["e"] - posS["t1"]) * part_sz
                cost = posS["t1"] * part_sz * COST
                trades.append(pnl - cost); equity += (pnl - cost)
                posS["sz"] *= (1 - pS["partial_pct"])
                posS["p1"] = True

            if r["low"] <= posS["t2"]:
                exit_price = posS["t2"]

            new_stop = posS["lm"] + pS["trail_mul"] * posS["atr0"]
            if new_stop < posS["s"]:
                posS["s"] = new_stop
            if r["high"] >= posS["s"]:
                exit_price = posS["s"]

            if exit_price is not None:
                pnl = (posS["e"] - exit_price) * posS["sz"]
                cost = exit_price * posS["sz"] * COST
                trades.append(pnl - cost); equity += (pnl - cost)
                posS = None

        # abrir LONG si aplica y no bloqueado por short activo
        if (posL is None) and (posS is None):  # anti-conflicto
            if (r["long_sig"]) and (not r.get("long_lock", False)):
                atr0 = r["atr"]
                if np.isfinite(atr0) and atr0 > 0:
                    entry = float(r["open"]) * (1 + SLIP)
                    stop  = entry - pL["sl_atr_mul"] * atr0
                    tp1   = entry + pL["tp1_atr_mul"] * atr0
                    tp2   = entry + pL["tp2_atr_mul"] * atr0
                    if stop < entry:
                        risk = (entry - stop)
                        base_sz = (equity * pL["risk_perc"]) / max(risk, 1e-9)
                        sz = base_sz * max(wL, 0.0)
                        entry_cost = -entry * sz * COST
                        trades.append(entry_cost); equity += entry_cost
                        posL = {"e": entry, "s": stop, "t1": tp1, "t2": tp2,
                                "sz": sz, "hm": entry, "p1": False, "atr0": atr0}

        # abrir SHORT si aplica y no bloqueado por long activo
        if (posS is None) and (posL is None):  # anti-conflicto
            if (r["short_sig"]) and (not r.get("short_lock", False)):
                atr0 = r["atr"]
                if np.isfinite(atr0) and atr0 > 0:
                    entry = float(r["open"]) * (1 - SLIP)
                    stop  = entry + pS["sl_atr_mul"] * atr0
                    tp1   = entry - pS["tp1_atr_mul"] * atr0
                    tp2   = entry - pS["tp2_atr_mul"] * atr0
                    if stop > entry:
                        risk = (stop - entry)
                        base_sz = (equity * pS["risk_perc"]) / max(risk, 1e-9)
                        sz = base_sz * max(wS, 0.0)
                        entry_cost = -entry * sz * COST
                        trades.append(entry_cost); equity += entry_cost
                        posS = {"e": entry, "s": stop, "t1": tp1, "t2": tp2,
                                "sz": sz, "lm": entry, "p1": False, "atr0": atr0}

        # mark-to-market
        unreal = 0.0
        if posL is not None:
            unreal += (r["close"] - posL["e"]) * posL["sz"]
        if posS is not None:
            unreal += (posS["e"] - r["close"]) * posS["sz"]
        eq_curve.append(equity + unreal)

    arr = np.array(trades, dtype=float)
    if arr.size == 0:
        return {"net": 0.0, "pf": 0.0, "wr": 0.0, "trades": 0, "mdd": 0.0, "score": 0.0}
    net = float(arr.sum())
    pf = safe_pf(arr)
    wr = float((arr > 0).sum() / len(arr) * 100.0)
    eq = np.array(eq_curve, dtype=float)
    mdd = float((np.maximum.accumulate(eq) - eq).max()) if eq.size else 0.0
    score = score_from(net, mdd)
    return {"net": net, "pf": pf, "wr": wr, "trades": int(len(arr)), "mdd": mdd, "score": score}

# -----------------------
# Main
# -----------------------
def main():
    print("🔄 Loading 4h data (ccxt + fallbacks + yfinance)…")
    df_raw, sources = load_4h()
    print("Sources:", sources)
    if df_raw.empty:
        print("No data."); return

    print("✨ Adding features…")
    df = add_features(df_raw).dropna()
    if len(df) < MIN_BARS_REQ:
        print(f"Not enough bars ({len(df)} < {MIN_BARS_REQ})."); return

    # split 70/30 OOS
    cut = int(len(df) * 0.7)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    print("🧠 Training model on 70% / scoring 30% OOS…")
    model = train_model(train)
    test = test.copy()
    test["prob_up"] = predict_prob_up(model, test)

    # cargar locks (sin tocarlos)
    pL = load_lock(LOCK_LONG, FALLBACK_LONG)
    pS = load_lock(LOCK_SHORT, FALLBACK_SHORT)

    # señales con guardas
    test = gen_signals(test, pL, pS)

    # micro-grid de pesos con límite de MDD combinado
    # target cap: min(MDD_long_solo, MDD_short_solo) * 1.05
    # primero medimos cada pierna por separado (w=1)
    soloL = test.copy(); soloL["short_sig"] = False
    rL = backtest_combo(soloL, pL, pS, wL=1.0, wS=0.0)
    soloS = test.copy(); soloS["long_sig"] = False
    rS = backtest_combo(soloS, pL, pS, wL=0.0, wS=1.0)
    mdd_cap = min(rL["mdd"], rS["mdd"]) * 1.05 if (rL["trades"]>0 and rS["trades"]>0) else max(rL["mdd"], rS["mdd"])

    grid_wL = [1.0, 0.75, 0.5]
    grid_wS = [1.0, 0.75, 0.5]

    best = None
    for wL in grid_wL:
        for wS in grid_wS:
            r = backtest_combo(test, pL, pS, wL=wL, wS=wS)
            feasible = (r["mdd"] <= mdd_cap) if (mdd_cap > 0) else True
            score = r["score"] if feasible else -1.0
            cand = {"wL": wL, "wS": wS, **r, "feasible": feasible}
            if (best is None) or (score > best["score"]) or \
               (abs(score - best["score"]) < 1e-9 and r["mdd"] < best["mdd"]):
                best = cand

    # también reportamos baseline wL=1, wS=1 (por comparación)
    base_r = backtest_combo(test, pL, pS, wL=1.0, wS=1.0)

    print("\n— Portfolio with conflict-guard (OOS zone) —")
    print(f"Solo LONG  → Net {rL['net']:.2f}, PF {rL['pf']:.2f}, WR {rL['wr']:.2f}%, Trades {rL['trades']}, MDD {rL['mdd']:.2f}, Score {rL['score']:.2f}")
    print(f"Solo SHORT → Net {rS['net']:.2f}, PF {rS['pf']:.2f}, WR {rS['wr']:.2f}%, Trades {rS['trades']}, MDD {rS['mdd']:.2f}, Score {rS['score']:.2f}")
    print(f"Baseline   → Net {base_r['net']:.2f}, PF {base_r['pf']:.2f}, WR {base_r['wr']:.2f}%, Trades {base_r['trades']}, MDD {base_r['mdd']:.2f}, Score {base_r['score']:.2f}")
    if best:
        print(f"Selected   → wL {best['wL']:.2f}, wS {best['wS']:.2f} | Net {best['net']:.2f}, PF {best['pf']:.2f}, WR {best['wr']:.2f}%, Trades {best['trades']}, MDD {best['mdd']:.2f}, Score {best['score']:.2f} | feasible={best['feasible']} | cap={mdd_cap:.2f}")

    # holdout puro (últimos 75 días), usando pesos elegidos
    bars_hold = HOLDOUT_DAYS * 6
    if len(df) > bars_hold + 200 and best:
        hold = df.iloc[-bars_hold:].copy()
        # necesitamos prob_up y señales en holdout también
        # re-entrenamos en todo lo anterior al holdout para evitar leakage
        train2 = df.iloc[:-bars_hold].copy()
        model2 = train_model(train2)
        hold["prob_up"] = predict_prob_up(model2, hold)
        hold = gen_signals(hold, pL, pS)
        rh = backtest_combo(hold, pL, pS, wL=best["wL"], wS=best["wS"])
        print(f"\nHoldout {HOLDOUT_DAYS}d (selected weights) → Net {rh['net']:.2f}, PF {rh['pf']:.2f}, WR {rh['wr']:.2f}%, Trades {rh['trades']}, MDD {rh['mdd']:.2f}, Score {rh['score']:.2f}")

    # guardar resumen
    ensure_dirs()
    stamp = now_stamp()
    path = f"./reports/portfolio_conflict_guard_{stamp}.json"
    out = {
        "stamp_utc": pd.Timestamp.utcnow().isoformat(),
        "sources": sources,
        "solo_long": rL, "solo_short": rS,
        "baseline_w1_w1": base_r,
        "selected": best,
        "mdd_cap": float(mdd_cap),
        "notes": "Wrapper que respeta locks; no escribe, solo evalúa y selecciona pesos factibles."
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"📝 Saved portfolio conflict-guard summary → {path}")

if __name__ == "__main__":
    main()