# entry_sweep_ccxt_diamante.py
# Focused Entry Sweep (threshold / ADX / trend) on ccxt/Binance
# Keeps ATR risk engine locked to your latest "Profile A — ATR v2" settings.

import os, sys, json, time, math
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, List, Optional

# === ML ===
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# === Data ===
import ccxt

np.set_printoptions(suppress=True)
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 50)

# -----------------------------
# User/Strategy Settings
# -----------------------------
EXCHANGE_ID = "binance"
SYMBOL      = "BTC/USDT"       # ccxt symbol
TF_4H       = "4h"
TF_1D       = "1d"

# How much history to load
LOOKBACK_DAYS_4H = 730
LOOKBACK_DAYS_1D = 900

# Features used by the ML signal
FEATURES = ["ema_fast","ema_slow","rsi","atr","adx4h"]

# Trading cost model
COST = 0.0002         # taker fees (roundturn applied on each entry/exit)
SLIP = 0.0001         # slippage (applied to entry/exit fills)
RISK_PERC_LOCKED = 0.007

# === Baseline (Profile A — ATR v2 locked) ===
BASE_PARAMS = {
    "threshold": 0.52,
    "adx4_min": 6,
    "adx1d_min": 0,
    "sl_atr_mul": 1.1,
    "tp1_atr_mul": 1.6,
    "tp2_atr_mul": 4.5,
    "trail_mul": 0.8,
    "partial_pct": 0.5,
    "risk_perc": RISK_PERC_LOCKED,
    "trend_mode": "fast_or_slope",  # "fast_or_slope" | "slope_up" | "none"
}

# Entry sweep space (filters/signal only)
ENTRY_GRID = {
    "threshold":  [0.49, 0.50, 0.51, 0.52],
    "adx4_min":   [0, 4, 6, 8],
    "adx1d_min":  [0, 3],
    "trend_mode": ["fast_or_slope", "slope_up", "none"],
}

# Guardrails for the entry sweep
ENTRY_GUARD = {
    "min_trades_180": 60,  # ensure enough activity
    "score180_floor": 0.90, # keep ≥ 90% of baseline 180d score
    "mdd180_cap":     1.15, # MDD must be ≤ 115% of baseline MDD
    "max_90d_ddrop":  0.15, # 90d score not worse than -15% vs baseline
}

REPORT_DIR = "./reports"

# -----------------------------
# Utils
# -----------------------------
def ts_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def fmt_row(prefix: str, r: Dict[str, Any]):
    print(f"{prefix} → Net {r['net']:.2f}, PF {r['pf']:.2f}, Win% {r['win_rate']:.2f}, "
          f"Trades {r['trades']}, MDD {r['mdd']:.2f}, Score {r['score']:.2f}")

# -----------------------------
# Data via ccxt (Binance)
# -----------------------------
def fetch_ohlcv_all(exchange, symbol: str, timeframe: str, since_ms: int, limit: int = 1000) -> pd.DataFrame:
    """Loop-fetch OHLCV from ccxt into a DataFrame."""
    all_rows = []
    ms = since_ms
    while True:
        try:
            rows = exchange.fetch_ohlcv(symbol, timeframe, since=ms, limit=limit)
        except ccxt.NetworkError as e:
            time.sleep(1.0)
            continue
        except ccxt.BaseError as e:
            print(f"ccxt error: {e}")
            break

        if not rows:
            break

        all_rows += rows
        # advance
        ms_next = rows[-1][0] + 1
        if ms_next == ms:
            break
        ms = ms_next
        # polite sleeping for rate limits
        time.sleep(exchange.rateLimit / 1000.0)

        # safety: stop near "now"
        if ms > ts_ms(now_utc() - timedelta(minutes=5)):
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert(None)
    df.set_index("time", inplace=True)
    return df.astype(float).sort_index()

def load_ccxt_data() -> Dict[str, pd.DataFrame]:
    print("🔄 Downloading 4h + 1D data (ccxt/Binance)…")
    ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    end = now_utc()
    since_4h = end - timedelta(days=LOOKBACK_DAYS_4H)
    since_1d  = end - timedelta(days=LOOKBACK_DAYS_1D)
    df4 = fetch_ohlcv_all(ex, SYMBOL, TF_4H, ts_ms(since_4h))
    df1 = fetch_ohlcv_all(ex, SYMBOL, TF_1D, ts_ms(since_1d))

    if df4.empty:
        print("Failed to download 4h data."); sys.exit(1)
    if df1.empty:
        print("Failed to download 1D data."); sys.exit(1)
    return {"4h": df4, "1d": df1}

# -----------------------------
# Features & ML
# -----------------------------
def add_features_4h(df4: pd.DataFrame) -> pd.DataFrame:
    df = df4.copy()
    df["ema_fast"] = ta.ema(df["close"], length=12)
    df["ema_slow"] = ta.ema(df["close"], length=26)
    df["rsi"]      = ta.rsi(df["close"], length=14)
    df["atr"]      = ta.atr(df["high"], df["low"], df["close"], length=14)

    adx4 = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx4 is not None and not adx4.empty:
        adx_col = next((c for c in adx4.columns if "ADX_" in c or c.lower()=="adx"), None)
        if adx_col is None:
            adx_col = adx4.columns[-1]
        df["adx4h"] = adx4[adx_col]
    else:
        df["adx4h"] = np.nan
    return df.dropna()

def add_features_1d(df1: pd.DataFrame) -> pd.DataFrame:
    df = df1.copy()
    adx1 = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx1 is not None and not adx1.empty:
        adx_col = next((c for c in adx1.columns if "ADX_" in c or c.lower()=="adx"), None)
        if adx_col is None:
            adx_col = adx1.columns[-1]
        df["adx1d"] = adx1[adx_col]
    else:
        df["adx1d"] = np.nan
    return df[["adx1d"]].dropna()

def attach_daily_to_4h(df4: pd.DataFrame, df1_adx: pd.DataFrame) -> pd.DataFrame:
    # align daily ADX to 4h bars
    d = df4.copy()
    # forward-fill daily ADX to 4h timestamps
    d["date"] = d.index.date
    daily = df1_adx.copy()
    daily["date"] = daily.index.date
    m = pd.merge(d.reset_index(), daily.reset_index()[["date","adx1d"]], on="date", how="left")
    m.set_index("time", inplace=True)
    m.drop(columns=["date"], inplace=True)
    m["adx1d"] = m["adx1d"].ffill()
    return m.dropna()

def make_target(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["target"] = np.sign(d["close"].shift(-1) - d["close"]).fillna(0).astype(int)
    return d.dropna()

def train_model(insample: pd.DataFrame):
    X = insample[FEATURES]
    y = insample["target"]
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42))
    ])
    pipe.fit(X, y)
    return pipe

def predict_prob_up(model, df: pd.DataFrame) -> np.ndarray:
    probs = model.predict_proba(df[FEATURES])
    classes = list(model.classes_)
    if 1 in classes:
        up_idx = classes.index(1)
    else:
        # fallback if classes are [-1,0] etc.
        up_idx = np.argmax(probs.mean(axis=0))
    return probs[:, up_idx]

# -----------------------------
# Backtest (long-only, ATR risk engine)
# -----------------------------
def pass_trend(mode: str, row, prev_row) -> bool:
    if mode == "none":
        return True
    if mode == "slope_up":
        if prev_row is None: return False
        return (row["ema_slow"] - prev_row["ema_slow"]) > 0
    if mode == "fast_or_slope":
        fast_ok = row["ema_fast"] > row["ema_slow"]
        if prev_row is None:
            slope_ok = False
        else:
            slope_ok = (row["ema_slow"] - prev_row["ema_slow"]) > 0
        return fast_ok or slope_ok
    return True

def run_backtest(oos: pd.DataFrame, p: Dict[str, Any], days: int) -> Dict[str, Any]:
    """
    Expects columns: open, high, low, close, atr, adx4h, adx1d, ema_slow, ema_fast, prob_up
    """
    if oos.empty:
        return {"net":0.0,"pf":0.0,"win_rate":0.0,"trades":0,"mdd":0.0,"score":0.0}

    df = oos.tail(days * 6).copy()  # 6 × 4h bars / day
    equity_closed = 10000.0
    trades: List[float] = []
    eq_curve: List[float] = []
    pos = None
    prev_row = None

    for _, r in df.iterrows():
        # entry gate
        if pos is None:
            if (r["prob_up"] > p["threshold"]
                and r["close"] > r["ema_slow"]
                and (p["adx4_min"] == 0 or r["adx4h"] >= p["adx4_min"])
                and (p["adx1d_min"] == 0 or r.get("adx1d", 0) >= p["adx1d_min"])
                and pass_trend(p.get("trend_mode","none"), r, prev_row)):

                atr0 = r["atr"]
                if np.isnan(atr0) or atr0 <= 0:
                    eq_curve.append(equity_closed); prev_row = r; continue

                entry = r["open"] * (1 + SLIP)
                stop  = entry - p["sl_atr_mul"] * atr0
                tp1   = entry + p["tp1_atr_mul"] * atr0
                tp2   = entry + p["tp2_atr_mul"] * atr0
                if stop >= entry:
                    eq_curve.append(equity_closed); prev_row = r; continue

                sz = (equity_closed * p["risk_perc"]) / (entry - stop)
                # entry fee at fill
                trades.append(-entry * sz * COST)
                equity_closed += (-entry * sz * COST)

                pos = {"e":entry,"s":stop,"t1":tp1,"t2":tp2,"sz":sz,"hm":entry,
                       "p1":False,"atr0":atr0}
        else:
            # manage open
            pos["hm"] = max(pos["hm"], r["high"])
            exit_p = None

            # partial tp1
            if (not pos["p1"]) and (r["high"] >= pos["t1"]):
                part_sz = pos["sz"] * p["partial_pct"]
                pnl = (pos["t1"] - pos["e"]) * part_sz
                fee = pos["t1"] * part_sz * COST
                trades.append(pnl - fee)
                equity_closed += (pnl - fee)
                pos["sz"] *= (1 - p["partial_pct"])
                pos["p1"] = True

            # full tp2
            if r["high"] >= pos["t2"]:
                exit_p = pos["t2"]

            # trail stop (use trail_mul on ATR from entry)
            new_stop = pos["hm"] - p["trail_mul"] * pos["atr0"]
            if new_stop > pos["s"]:
                pos["s"] = new_stop
            if r["low"] <= pos["s"]:
                exit_p = pos["s"]

            if exit_p is not None:
                pnl = (exit_p - pos["e"]) * pos["sz"]
                fee = exit_p * pos["sz"] * COST
                trades.append(pnl - fee)
                equity_closed += (pnl - fee)
                pos = None

        # mark-to-market
        unreal = 0.0
        if pos is not None:
            unreal = (r["close"] - pos["e"]) * pos["sz"]
        eq_curve.append(equity_closed + unreal)
        prev_row = r

    # metrics
    if len(trades) == 0:
        return {"net":0.0,"pf":0.0,"win_rate":0.0,"trades":0,"mdd":0.0,"score":0.0}
    arr = np.array(trades, dtype=float)
    net = float(arr.sum())
    gains = arr[arr>0]; losses = arr[arr<0]
    pf = float(gains.sum()/abs(losses.sum())) if losses.size else (np.inf if gains.size else 0.0)
    wr = float(len(gains)/len(arr)*100) if arr.size else 0.0

    eq = np.array(eq_curve, dtype=float)
    peak = np.maximum.accumulate(eq) if eq.size else np.array([0.0])
    mdd = float((peak - eq).max()) if eq.size else 0.0
    score = float(net / (mdd + 1.0))
    return {"net":net,"pf":pf,"win_rate":wr,"trades":int(len(arr)),"mdd":mdd,"score":score}

# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dir(REPORT_DIR)
    data = load_ccxt_data()

    # Build features
    df4 = add_features_4h(data["4h"])
    df1_adx = add_features_1d(data["1d"])
    df4 = attach_daily_to_4h(df4, df1_adx)

    # Target & split
    df4 = make_target(df4)
    split = int(len(df4) * 0.70)
    insample = df4.iloc[:split].copy()
    oos      = df4.iloc[split:].copy()
    if len(oos) < 300:
        print("Not enough OOS bars."); sys.exit(1)

    # Train ML
    model = train_model(insample)
    oos["prob_up"] = predict_prob_up(model, oos)

    # Baseline
    base = BASE_PARAMS.copy()
    print("\n— Baseline (current set) —")
    r90  = run_backtest(oos, base, 90)
    r180 = run_backtest(oos, base, 180)
    fmt_row("90d ", r90)
    fmt_row("180d", r180)

    # Focused Entry Sweep (threshold/ADX/trend) with guardrails
    print("\nFocused entry sweep (threshold / ADX / trend)…")
    cands = []
    for thr in ENTRY_GRID["threshold"]:
        for a4 in ENTRY_GRID["adx4_min"]:
            for a1 in ENTRY_GRID["adx1d_min"]:
                for tm in ENTRY_GRID["trend_mode"]:
                    cand = base.copy()
                    cand.update({"threshold":thr, "adx4_min":a4, "adx1d_min":a1, "trend_mode":tm})
                    rr90  = run_backtest(oos, cand, 90)
                    rr180 = run_backtest(oos, cand, 180)

                    status = "OK"
                    if rr180["trades"] < ENTRY_GUARD["min_trades_180"]:
                        status = "FAIL_TRADES180"
                    if rr180["score"] < r180["score"] * ENTRY_GUARD["score180_floor"]:
                        status = "FAIL_SCORE180"
                    if rr180["mdd"] > r180["mdd"] * ENTRY_GUARD["mdd180_cap"]:
                        status = "FAIL_MDD180"
                    if rr90["score"] < r90["score"] * (1.0 - ENTRY_GUARD["max_90d_ddrop"]):
                        status = "FAIL_SCORE90"

                    cands.append({
                        "threshold": thr, "adx4_min": a4, "adx1d_min": a1, "trend_mode": tm,
                        "net90": rr90["net"], "mdd90": rr90["mdd"], "score90": rr90["score"], "trades90": rr90["trades"],
                        "net180": rr180["net"], "mdd180": rr180["mdd"], "score180": rr180["score"], "trades180": rr180["trades"],
                        "status": status
                    })

    dfc = pd.DataFrame(cands)
    ok = dfc[dfc["status"]=="OK"].copy()
    if ok.empty:
        print("\nEntry sweep: no candidate passed guardrails. Keep baseline.")
        picked = base.copy()
    else:
        ok = ok.sort_values(by=["score180","score90","trades180"], ascending=False)
        print("\n— Top by 180d score (OK only) —")
        cols = ["threshold","adx4_min","adx1d_min","trend_mode","net90","mdd90","score90","trades90",
                "net180","mdd180","score180","trades180","status"]
        print(ok.head(10)[cols].to_string(index=False))
        best = ok.iloc[0]
        picked = base.copy()
        picked.update({
            "threshold": float(best["threshold"]),
            "adx4_min":  int(best["adx4_min"]),
            "adx1d_min": int(best["adx1d_min"]),
            "trend_mode": str(best["trend_mode"]),
        })

    print("\n— Baseline vs Entry-sweep pick —")
    fmt_row("Baseline 90d", r90)
    fmt_row("Picked   90d", run_backtest(oos, picked, 90))
    fmt_row("Baseline 180d", r180)
    fmt_row("Picked   180d", run_backtest(oos, picked, 180))

    # Persist best set
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(REPORT_DIR, f"entry_sweep_pick_{stamp}.json")
    with open(out_json, "w") as f:
        json.dump(picked, f, indent=2)
    print(f"\nSaved best set → {out_json}")

if __name__ == "__main__":
    main()