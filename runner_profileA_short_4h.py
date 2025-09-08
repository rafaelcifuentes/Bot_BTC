# runner_profileA_short_4h.py
# 4h swing SHORT — ccxt+fallbacks loader, ML prob_up→prob_down, ATR risk, walk-forward + HOLDOUT
# Sandbox: sin autolock (solo reporta). Requiere: ccxt pandas pandas_ta numpy scikit-learn yfinance

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
# Config / Parámetros (SHORT sandbox)
# -----------------------
SYMBOL = "BTC/USDT"
YF_SYMBOL = "BTC-USD"
TIMEFRAME = "4h"
MAX_BARS = 3000
SINCE_DAYS = 600
MIN_BARS_REQ = 800
COST = 0.0002
SLIP = 0.0001
SEED = 42

LOCK_PATH = "./profiles/profileA_short_binance_4h.json"   # solo snapshot sandbox
SNAPSHOT_DIR = "./profiles"
REPORT_DIR = "./reports"
CACHE_DIR = "./cache"
CACHE_FILE = os.path.join(CACHE_DIR, "btc_4h_ohlcv.csv")

HOLDOUT_DAYS = 75

# Short params iniciales (sandbox, no autolock)
PARAMS_SHORT = {
    "threshold": 0.47,    # se compara contra prob_down
    "adx4_min": 4,
    "adx1d_min": 0,
    "trend_mode": "slope_down",   # "none" | "slope_down" | "fast_or_slope_down"
    "sl_atr_mul": 1.10,
    "tp1_atr_mul": 1.60,
    "tp2_atr_mul": 4.50,
    "trail_mul": 0.80,
    "partial_pct": 0.50,
    "risk_perc": 0.005
}

AUTOLOCK_ON_IMPROVE = False  # sandbox: no promover
MDD_TOL_MULT = 1.20

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
# Data (multi-fuente)
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
        if getattr(ex, "markets", None) and sym not in ex.markets:
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
            if not ohlcv: break
            all_rows.extend(ohlcv)
            next_since = ohlcv[-1][0] + ms_per_bar
            if len(all_rows) >= MAX_BARS + 1200: break
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
    if df.empty: return pd.DataFrame()
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
        if not d.empty: dfs.append(d)
    try:
        d_yf = fetch_yf(YF_SYMBOL, period_days=SINCE_DAYS)
    except Exception:
        d_yf = pd.DataFrame()
    sources.append("yfinance: ok" if not d_yf.empty else "yfinance: empty")
    if not d_yf.empty: dfs.append(d_yf)

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
# Features + modelo
# -----------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ta.ema(out["close"], length=12)
    out["ema_slow"] = ta.ema(out["close"], length=26)
    out["rsi"] = ta.rsi(out["close"], length=14)
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=14)

    adx4 = ta.adx(out["high"], out["low"], out["close"], length=14)
    if adx4 is not None and not adx4.empty:
        col_adx = [c for c in adx4.columns if c.lower().startswith("adx")]
        out["adx4h"] = adx4[col_adx[0]] if col_adx else 0.0
    else:
        out["adx4h"] = 0.0

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
    d = df.copy()
    d["target"] = np.sign(d["close"].shift(-1) - d["close"])
    d.dropna(inplace=True)
    X = d[FEATURES]; y = d["target"].astype(int)
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
# Filtros de entrada (short)
# -----------------------
def pass_trend_short(row: pd.Series, mode: str) -> bool:
    if mode == "none":
        return True
    if mode == "slope_down":
        return bool(row.get("slope_down", 0) == 1)
    if mode == "fast_or_slope_down":
        cond = (row.get("ema_fast", np.nan) < row.get("ema_slow", np.nan)) or (row.get("slope_down", 0) == 1)
        return bool(bool(cond) and not np.isnan(row.get("ema_fast", np.nan)))
    return True

# -----------------------
# Backtest (SHORT)
# -----------------------
def backtest_short(df: pd.DataFrame, p: Dict, days: int) -> Dict:
    data = df.tail(days * 6).copy() if days > 0 else df.copy()

    equity_closed = 10000.0
    trades, eq_curve = [], []
    pos = None

    for _, r in data.iterrows():
        if pos is None:
            prob_up = r["prob_up"]
            prob_down = 1.0 - prob_up
            if (
                prob_down > p["threshold"]
                and r["adx4h"] >= p["adx4_min"]
                and r["adx1d"] >= p["adx1d_min"]
                and pass_trend_short(r, p.get("trend_mode", "none"))
                and r["close"] < r["ema_slow"]
            ):
                atr0 = r["atr"]
                if not np.isfinite(atr0) or atr0 <= 0:
                    eq_curve.append(equity_closed); continue

                entry = float(r["open"]) * (1 - SLIP)             # vender mejor
                stop  = entry + p["sl_atr_mul"] * atr0            # stop por encima
                tp1   = entry - p["tp1_atr_mul"] * atr0
                tp2   = entry - p["tp2_atr_mul"] * atr0
                if stop <= entry:
                    eq_curve.append(equity_closed); continue

                risk_per_unit = max(stop - entry, 1e-9)
                size = (equity_closed * p["risk_perc"]) / risk_per_unit

                entry_cost = -entry * size * COST
                trades.append(entry_cost)
                equity_closed += entry_cost

                pos = {"e": entry, "s": stop, "t1": tp1, "t2": tp2,
                       "sz": size, "lm": entry, "p1": False, "atr0": atr0}
        else:
            # actualizar mínimo favorable (más bajo = a favor del short)
            pos["lm"] = min(pos["lm"], r["low"])
            exit_price = None

            # TP1 parcial si toca
            if (not pos["p1"]) and (r["low"] <= pos["t1"]):
                part_sz = pos["sz"] * p["partial_pct"]
                pnl = (pos["e"] - pos["t1"]) * part_sz           # short: e - t1
                cost = pos["t1"] * part_sz * COST
                trades.append(pnl - cost)
                equity_closed += (pnl - cost)
                pos["sz"] *= (1 - p["partial_pct"])
                pos["p1"] = True

            # TP2 full si toca
            if r["low"] <= pos["t2"]:
                exit_price = pos["t2"]

            # trailing stop para short
            new_stop = pos["lm"] + p["trail_mul"] * pos["atr0"]
            if new_stop < pos["s"]:
                pos["s"] = new_stop

            # stop out si sube
            if r["high"] >= pos["s"]:
                exit_price = pos["s"]

            # cerrar si hay salida
            if exit_price is not None:
                pnl = (pos["e"] - exit_price) * pos["sz"]        # short: e - exit
                cost = exit_price * pos["sz"] * COST
                trades.append(pnl - cost)
                equity_closed += (pnl - cost)
                pos = None

        unreal = 0.0 if pos is None else (pos["e"] - r["close"]) * pos["sz"]
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
# Evaluación
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
        r = backtest_short(test, p, days=0)
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
    return backtest_short(hold, p, days=0)

def cost_slip_stress(df: pd.DataFrame, p: Dict, cost_mults=(0.5,1.0,1.5), slip_mults=(0.5,1.0,1.5)) -> pd.DataFrame:
    rows = []
    global COST, SLIP
    base_cost, base_slip = COST, SLIP
    for cm in cost_mults:
        for sm in slip_mults:
            COST = base_cost * cm
            SLIP = base_slip * sm
            r90  = backtest_short(df, p, days=90)
            r180 = backtest_short(df, p, days=180)
            rows.append({
                "cost_mult": cm, "slip_mult": sm,
                "net90": r90["net"], "mdd90": r90["mdd"], "score90": r90["score"],
                "net180": r180["net"], "mdd180": r180["mdd"], "score180": r180["score"]
            })
    COST, SLIP = base_cost, base_slip
    return pd.DataFrame(rows)

# -----------------------
# Competitiveness (simple 1–10)
# -----------------------
def clamp(x, lo, hi): return max(lo, min(hi, x))

def pf_score(pf: float) -> float:
    # saturación suave: PF 5+ ≈ 10
    return clamp(pf * 2.0, 0.0, 10.0)

def mdd_score(mdd: float) -> float:
    # entre 0–200: 0→10, 200→0 (lineal simple)
    return clamp(10.0 * (1.0 - (mdd / 200.0)), 0.0, 10.0)

def consistency_score(win_rate: float, trades: int) -> float:
    # más trades y WR↑ dan mejor; penaliza muestras pequeñas
    base = win_rate / 5.0  # 50% → 10
    adj = min(1.0, trades / 100.0)  # 100 trades → pleno
    return clamp(base * (0.5 + 0.5 * adj), 0.0, 10.0)

def robustness_score(pf: float, mdd: float) -> float:
    return clamp((pf_score(pf) * 0.6 + mdd_score(mdd) * 0.4), 0.0, 10.0)

def efficiency_score(net: float, mdd: float) -> float:
    s = score_from(net, mdd)
    # mapear score a 1–10 (heurístico)
    return clamp(s * 2.0, 0.0, 10.0)

def build_competitiveness(name: str, r: Dict) -> Dict:
    return {
        "label": name,
        "PFs": round(pf_score(r.get("pf", 0.0)), 2),
        "MDD": round(mdd_score(r.get("mdd", 0.0)), 2),
        "Cons": round(consistency_score(r.get("win_rate", 0.0), r.get("trades", 0)), 2),
        "Robo": round(robustness_score(r.get("pf", 0.0), r.get("mdd", 0.0)), 2),
        "Eff": round(efficiency_score(r.get("net", 0.0), r.get("mdd", 0.0)), 2),
    }

def save_competitiveness_summary(stamp: str, rows: List[Dict]):
    path = os.path.join(REPORT_DIR, f"competitiveness_short_{stamp}.json")
    payload = {"stamp_utc": pd.Timestamp.utcnow().isoformat(), "rows": rows}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"📊 Saved competitiveness summary (short) → {path}")

# -----------------------
# Report I/O
# -----------------------
def save_report(sources: List[str], base90: Dict, base180: Dict,
                wf: pd.DataFrame, wf_agg: Dict, hold: Dict, params: Dict,
                stress_base: pd.DataFrame):

    ensure_dirs()
    stamp = now_stamp()
    wf_csv = os.path.join(REPORT_DIR, f"walkforward_short_{stamp}.csv")
    json_path = os.path.join(REPORT_DIR, f"summary_short_{stamp}.json")

    wf_out = wf.copy()
    if "fold_start" in wf_out.columns:
        wf_out["fold_start"] = wf_out["fold_start"].astype(str)
    wf_out.to_csv(wf_csv, index=False)

    payload = {
        "stamp_utc": pd.Timestamp.utcnow().isoformat(),
        "sources": sources,
        "baseline_90d": base90,
        "baseline_180d": base180,
        "walk_forward_rows": wf_out.to_dict(orient="records"),
        "walk_forward_agg": wf_agg,
        "holdout": hold,
        "short_params_used": params,
        "stress_base": stress_base.to_dict(orient="records") if not stress_base.empty else []
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"📝 Saved report JSON (short) → {json_path}")
    print(f"🧾 Saved walk-forward CSV (short) → {wf_csv}")

    # Competitiveness quick table
    rows = [
        build_competitiveness("90d", base90),
        build_competitiveness("180d", base180),
        build_competitiveness("Holdout", hold),
    ]
    save_competitiveness_summary(stamp, rows)

# -----------------------
# Main
# -----------------------
def main():
    print("🔄 Downloading 4h data (ccxt/Binance + fallback yfinance)…")
    df_raw, sources = load_4h_stitched()
    print("Sources:", sources)
    if df_raw.empty:
        print("Not enough data (live + cache both empty)."); return

    # snapshot sandbox params (no lock change de largo)
    ensure_dirs()
    snap_blob = PARAMS_SHORT.copy()
    snap_blob["_meta"] = {"profile":"A-short","version":"sandbox","timestamp_utc": pd.Timestamp.utcnow().isoformat()}
    snap_path = os.path.join(SNAPSHOT_DIR, f"profileA_short_binance_4h_{now_stamp()}.json")
    with open(snap_path, "w") as f:
        json.dump(snap_blob, f, indent=2)
    print(f"📎 Snapshot (short sandbox) → {snap_path}")

    df = add_features(df_raw).dropna()
    if len(df) < MIN_BARS_REQ:
        print(f"Still not enough 4h bars after fallbacks. Got {len(df)} (min {MIN_BARS_REQ})"); return

    # 70/30 split
    train, test = split_train_test(df, train_frac=0.7)
    train_with, test_with = attach_probs(train, test)

    print("\n— Baseline SHORT (70% train → OOS test) —")
    base90  = backtest_short(test_with, PARAMS_SHORT, days=90)
    base180 = backtest_short(test_with, PARAMS_SHORT, days=180)
    print(f"Using short params: {PARAMS_SHORT}")
    fmt_row("90d  base (short)", base90)
    fmt_row("180d base (short)", base180)

    # Walk-forward
    print("\n— Walk-forward 6×90 (rolling, non-overlapping) —")
    wf = walk_forward(df, PARAMS_SHORT, horizon_days=90, folds=6)
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
        print(f"\nWF aggregates (short, 90d): mean net {wf_agg['mean_net']:.2f}, median PF {wf_agg['median_pf']:.2f}, "
              f"mean WR {wf_agg['mean_wr']:.2f}%, total trades {wf_agg['total_trades']}, "
              f"median MDD {wf_agg['median_mdd']:.2f}, mean score {wf_agg['mean_score']:.2f}, folds {wf_agg['folds']}")
    else:
        wf_agg = {}

    # Holdout
    print(f"\n— Holdout (short, last {HOLDOUT_DAYS} days) —")
    hold = holdout_eval(df, PARAMS_SHORT, HOLDOUT_DAYS)
    fmt_row(f"Holdout {HOLDOUT_DAYS}d (short)", hold)

    # Stress
    print("\n— Cost/slippage stress (short) —")
    stress_base = cost_slip_stress(test_with, PARAMS_SHORT)
    if not stress_base.empty:
        print(stress_base.to_string(index=False))

    # Save report + competitiveness
    save_report(sources, base90, base180, wf, wf_agg, hold, PARAMS_SHORT, stress_base)

    print("\nSandbox short run complete (no autolock). Review JSON/CSV and competitiveness file in ./reports/")

if __name__ == "__main__":
    main()