# runner_profileA_binance_4h.py
# Loads params from JSON, fetches Binance data via ccxt, trains the ML signal,
# runs backtests (90/180d), optional mini-grid tuning with guardrails,
# cost/slippage stress, persists history, and prints "lessons learned".

import os, json, time, math, argparse, warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ------------------------- Config -------------------------
EXCHANGE_ID   = "binance"
SYMBOL        = "BTC/USDT"
TIMEFRAME_4H  = "4h"
TIMEFRAME_1D  = "1d"

COST      = 0.0002
SLIP      = 0.0001
SEED      = 42
INIT_EQ   = 10_000.0

FEATURES  = ["ema_fast","ema_slow","rsi","atr","adx4h","slope"]
REPORTS_DIR = "./reports"
PROFILES_DIR = "./profiles"
HISTORY_CSV  = os.path.join(REPORTS_DIR, "history_profileA_binance_4h.csv")

# ------------------------- Utils -------------------------
def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(PROFILES_DIR, exist_ok=True)

def load_params(json_path: str, default_params: dict) -> dict:
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return default_params

def save_params(json_path: str, params: dict):
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)

def save_run_summary(summary: dict, stamp: str):
    path = os.path.join(REPORTS_DIR, f"run_summary_{stamp}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

def append_history(stamp: str, summary: dict):
    row = {
        "timestamp": stamp,
        "threshold": summary["params"]["threshold"],
        "adx4_min": summary["params"]["adx4_min"],
        "adx1d_min": summary["params"]["adx1d_min"],
        "trend_mode": summary["params"]["trend_mode"],
        "sl_atr_mul": summary["params"]["sl_atr_mul"],
        "tp1_atr_mul": summary["params"]["tp1_atr_mul"],
        "tp2_atr_mul": summary["params"]["tp2_atr_mul"],
        "trail_mul": summary["params"]["trail_mul"],
        "partial_pct": summary["params"]["partial_pct"],
        "risk_perc": summary["params"]["risk_perc"],
    }
    for H in (90, 180):
        m = summary["horizons"].get(str(H), {})
        row[f"net_{H}"]   = m.get("net", 0.0)
        row[f"pf_{H}"]    = m.get("pf", 0.0)
        row[f"wr_{H}"]    = m.get("win_rate", 0.0)
        row[f"tr_{H}"]    = m.get("trades", 0)
        row[f"mdd_{H}"]   = m.get("mdd", 0.0)
        row[f"score_{H}"] = m.get("score", 0.0)
    df = pd.DataFrame([row])
    if os.path.exists(HISTORY_CSV):
        df.to_csv(HISTORY_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_CSV, index=False)

def lessons_learned():
    if not os.path.exists(HISTORY_CSV):
        print("\n(No history yet → lessons will appear from next run.)")
        return
    df = pd.read_csv(HISTORY_CSV)
    if len(df) < 2:
        print("\n(Only one run in history → lessons will appear from next run.)")
        return
    last, prev = df.iloc[-1], df.iloc[-2]
    def delta(a,b): return float(a) - float(b)
    notes = []
    for H in (90, 180):
        dn = delta(last[f"net_{H}"],   prev[f"net_{H}"])
        ds = delta(last[f"score_{H}"], prev[f"score_{H}"])
        dm = delta(last[f"mdd_{H}"],   prev[f"mdd_{H}"])
        dtr= delta(last[f"tr_{H}"],    prev[f"tr_{H}"])
        notes.append(f"{H}d → ΔNet {dn:.2f}, ΔScore {ds:.2f}, ΔMDD {dm:.2f}, ΔTrades {dtr:.0f}")

    print("\n◆ Lessons learned (vs previous run)")
    print("  • ATR risk + trailing + partial TP remain core; tweak entries carefully.")
    print("  • 90d is noisy; steer by 180d score under guardrails.")
    print("  • If Net↑ and MDD≈flat → keep; if Net↑ but MDD↑↑ → raise sl_atr_mul or threshold.")
    for n in notes:
        print("  • " + n)

# ------------------------- Data: ccxt fetch -------------------------
def fetch_ccxt(symbol: str, timeframe: str, since_ms: int, until_ms: int, limit=1000):
    import ccxt
    ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    out = []
    since = since_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        out.extend(batch)
        last_ts = batch[-1][0]
        since = last_ts + 1
        time.sleep(0.2)
        if last_ts >= until_ms or len(batch) < limit:
            break
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df.set_index("timestamp", inplace=True)
    return df

def download_data(days_back_4h=800, days_back_1d=800):
    end = datetime.now(timezone.utc)
    start_4h = end - timedelta(days=days_back_4h)
    start_1d = end - timedelta(days=days_back_1d)
    print("🔄 Downloading 4h + 1D data (ccxt/Binance)…")
    df4 = fetch_ccxt(SYMBOL, TIMEFRAME_4H, int(start_4h.timestamp()*1000), int(end.timestamp()*1000))
    df1 = fetch_ccxt(SYMBOL, TIMEFRAME_1D, int(start_1d.timestamp()*1000), int(end.timestamp()*1000))
    if df4.empty:
        raise RuntimeError("4h data download empty.")
    if df1.empty:
        df1 = pd.DataFrame()
    return df4, df1

# ------------------------- Features & Model -------------------------
def add_features_4h(df4: pd.DataFrame) -> pd.DataFrame:
    df = df4.copy()
    df["ema_fast"] = ta.ema(df["close"], length=12)
    df["ema_slow"] = ta.ema(df["close"], length=26)
    df["rsi"]      = ta.rsi(df["close"], length=14)
    df["atr"]      = ta.atr(df["high"], df["low"], df["close"], length=14)
    adx4 = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx4 is None or adx4.empty:
        df["adx4h"] = np.nan
    else:
        col = next((c for c in adx4.columns if "ADX" in c.upper()), None)
        df["adx4h"] = adx4[col] if col else np.nan
    df["slope"] = df["ema_slow"].diff()
    return df.dropna()

def add_adx_daily(df4: pd.DataFrame, df1: pd.DataFrame) -> pd.DataFrame:
    if df1.empty:
        df4["adx1d"] = np.nan
        return df4
    d1 = df1.copy()
    adx1 = ta.adx(d1["high"], d1["low"], d1["close"], length=14)
    if adx1 is None or adx1.empty:
        d1["adx1d"] = np.nan
    else:
        col = next((c for c in adx1.columns if "ADX" in c.upper()), None)
        d1["adx1d"] = adx1[col] if col else np.nan
    d1 = d1[["adx1d"]].copy()
    d1.index = pd.to_datetime(d1.index)
    df4 = df4.join(d1["adx1d"].reindex(df4.index, method="ffill"))
    return df4

def make_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    diff = df["close"].shift(-1) - df["close"]
    df["target"] = np.where(diff > 0, 1, -1)
    return df.dropna()

def train_rf(insample: pd.DataFrame):
    X = insample[FEATURES]; y = insample["target"]
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=SEED))
    ])
    pipe.fit(X, y)
    return pipe

def add_prob_up(model, oos: pd.DataFrame) -> pd.DataFrame:
    probs = model.predict_proba(oos[FEATURES])
    classes = list(model.classes_)
    up_idx = classes.index(1)
    oos = oos.copy()
    oos["prob_up"] = probs[:, up_idx]
    return oos

# ------------------------- Trend gate -------------------------
def pass_trend(mode: str, row) -> bool:
    if mode == "none":
        return True
    if mode == "slope_up":
        return pd.notna(row["slope"]) and row["slope"] > 0
    if mode == "fast_or_slope":
        cond_fast  = row["close"] > row["ema_slow"]
        cond_slope = pd.notna(row["slope"]) and row["slope"] > 0
        return bool(cond_fast or cond_slope)
    return True

# ------------------------- Backtest -------------------------
def backtest_long(df: pd.DataFrame, p: dict, days: int):
    slice_ = df.tail(days * 6).copy()  # 6 bars/day at 4h
    equity_closed = INIT_EQ
    trades = []
    pos = None
    eq_curve = []

    for _, r in slice_.iterrows():
        # open
        if pos is None:
            if (r["prob_up"] > p["threshold"]
                and (r["adx4h"] >= p["adx4_min"] if pd.notna(r["adx4h"]) else True)
                and (pd.isna(r["adx1d"]) or r["adx1d"] >= p["adx1d_min"])
                and pass_trend(p.get("trend_mode","slope_up"), r)):
                atr0 = r["atr"]
                if atr0 is None or np.isnan(atr0) or atr0 <= 0:
                    eq_curve.append(equity_closed); continue
                entry = r["open"] * (1 + SLIP)
                stop  = entry - p["sl_atr_mul"] * atr0
                tp1   = entry + p["tp1_atr_mul"] * atr0
                tp2   = entry + p["tp2_atr_mul"] * atr0
                if stop >= entry:
                    eq_curve.append(equity_closed); continue
                sz = (equity_closed * p["risk_perc"]) / (entry - stop)
                entry_cost = -entry * sz * COST
                trades.append(entry_cost); equity_closed += entry_cost
                pos = {"e": entry, "s": stop, "t1": tp1, "t2": tp2,
                       "sz": sz, "hm": entry, "p1": False, "atr0": atr0}

        else:
            pos["hm"] = max(pos["hm"], r["high"])
            exit_p = None

            if (not pos["p1"]) and (r["high"] >= pos["t1"]):
                part = pos["sz"] * p["partial_pct"]
                pnl  = (pos["t1"] - pos["e"]) * part
                cost = pos["t1"] * part * COST
                trades.append(pnl - cost); equity_closed += (pnl - cost)
                pos["sz"] *= (1 - p["partial_pct"])
                pos["p1"] = True

            if r["high"] >= pos["t2"]:
                exit_p = pos["t2"]

            new_stop = pos["hm"] - p["trail_mul"] * pos["atr0"]
            if new_stop > pos["s"]:
                pos["s"] = new_stop
            if r["low"] <= pos["s"]:
                exit_p = pos["s"]

            if exit_p is not None:
                pnl  = (exit_p - pos["e"]) * pos["sz"]
                cost = exit_p * pos["sz"] * COST
                trades.append(pnl - cost); equity_closed += (pnl - cost)
                pos = None

        unreal = 0.0
        if pos is not None:
            unreal = (r["close"] - pos["e"]) * pos["sz"]
        eq_curve.append(equity_closed + unreal)

    if len(trades) == 0:
        return {"net": 0.0, "pf": 0.0, "win_rate": 0.0, "trades": 0, "mdd": 0.0, "score": 0.0}

    arr = np.array(trades, dtype=float)
    net = float(arr.sum())
    gains = arr[arr > 0]
    losses = arr[arr < 0]
    pf = float(gains.sum() / abs(losses.sum())) if losses.size else (np.inf if gains.size else 0.0)
    wr = float(len(gains) / len(arr) * 100)
    eq = np.array(eq_curve, dtype=float)
    peak = np.maximum.accumulate(eq) if eq.size else np.array([0.0])
    mdd = float((peak - eq).max()) if eq.size else 0.0
    score = float(net / (mdd + 1))
    return {"net": net, "pf": pf, "win_rate": wr, "trades": int(len(arr)), "mdd": mdd, "score": score}

# ------------------------- Mini-grid tuner (guardrailed) -------------------------
def mini_grid_tighten(oos: pd.DataFrame, P: dict, base90: dict, base180: dict):
    # Small neighborhood around current params
    thr_list   = [round(P["threshold"] + d, 2) for d in (-0.02, -0.01, 0.00, 0.01)]  # e.g. 0.48–0.53
    thr_list   = [t for t in thr_list if 0.45 <= t <= 0.60]
    sl_list    = [round(x,2) for x in (P["sl_atr_mul"], P["sl_atr_mul"]+0.1, P["sl_atr_mul"]+0.2)]
    trail_list = [round(x,2) for x in (P["trail_mul"], 1.0, P["trail_mul"]+0.2)]
    risk_list  = sorted(set([round(x,4) for x in (P["risk_perc"], 0.005, 0.006, 0.007)]))
    part_list  = sorted(set([round(x,2) for x in (P["partial_pct"], 0.4, 0.5)]))

    best = None
    rows = []
    for thr in thr_list:
        for slm in sl_list:
            for trm in trail_list:
                for rp in risk_list:
                    for pp in part_list:
                        cand = P.copy()
                        cand.update({
                            "threshold": thr, "sl_atr_mul": slm,
                            "trail_mul": trm, "risk_perc": rp, "partial_pct": pp
                        })
                        r90  = backtest_long(oos, cand, 90)
                        r180 = backtest_long(oos, cand, 180)
                        status = "OK"
                        # Guardrails:
                        # 1) 90d score must not drop by >10%
                        if base90["score"] > 0 and (r90["score"] < 0.9 * base90["score"]):
                            status = "DEGRADE_90"
                        # 2) 180d trades floor
                        if r180["trades"] < 140:
                            status = (status + ",DEGRADE_180") if status != "OK" else "DEGRADE_180"

                        rows.append({
                            "threshold": thr, "sl_atr_mul": slm, "trail_mul": trm,
                            "risk_perc": rp, "partial_pct": pp,
                            "net90": r90["net"], "mdd90": r90["mdd"], "score90": r90["score"], "trades90": r90["trades"],
                            "net180": r180["net"], "mdd180": r180["mdd"], "score180": r180["score"], "trades180": r180["trades"],
                            "status": status
                        })
                        if status == "OK":
                            if best is None or r180["score"] > best["r180"]["score"]:
                                best = {"params": cand, "r90": r90, "r180": r180}

    df = pd.DataFrame(rows).sort_values("score180", ascending=False)
    ok_df = df[df["status"]=="OK"].copy()

    print("\n— Mini-grid (tighten risk) —")
    if ok_df.empty:
        print("No candidate passed guardrails. Keeping baseline.")
        return None, df
    print("Top 8 by 180d score (OK only):")
    print(ok_df.head(8).round(3).to_string(index=False))

    print("\n— Picked (guardrailed) —")
    print(json.dumps(best["params"], indent=2))
    print("\n— Baseline vs Pick —")
    print(f"90d  base: Net {base90['net']:.2f}, MDD {base90['mdd']:.2f}, Score {base90['score']:.2f}, Trades {base90['trades']}")
    print(f"90d  pick: Net {best['r90']['net']:.2f}, MDD {best['r90']['mdd']:.2f}, Score {best['r90']['score']:.2f}, Trades {best['r90']['trades']}")
    print(f"180d base: Net {base180['net']:.2f}, MDD {base180['mdd']:.2f}, Score {base180['score']:.2f}, Trades {base180['trades']}")
    print(f"180d pick: Net {best['r180']['net']:.2f}, MDD {best['r180']['mdd']:.2f}, Score {best['r180']['score']:.2f}, Trades {best['r180']['trades']}")

    # save sweep
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ok_df.to_csv(os.path.join(REPORTS_DIR, f"minigrid_ok_{stamp}.csv"), index=False)
    df.to_csv(os.path.join(REPORTS_DIR, f"minigrid_all_{stamp}.csv"), index=False)
    return best, ok_df

# ------------------------- Pipeline -------------------------
def run_once(params_path: str, horizons=(90,180), minigrid=False, autolock=False):
    ensure_dirs()
    DEFAULT = {
        "threshold": 0.50, "adx4_min": 6, "adx1d_min": 0, "trend_mode": "slope_up",
        "sl_atr_mul": 1.1, "tp1_atr_mul": 1.6, "tp2_atr_mul": 4.5, "trail_mul": 0.8,
        "partial_pct": 0.5, "risk_perc": 0.007
    }
    P = load_params(params_path, DEFAULT)

    # data & features
    df4, df1 = download_data(days_back_4h=800, days_back_1d=800)
    df = add_features_4h(df4)
    df = add_adx_daily(df, df1)
    df = make_target(df)

    # split
    split = int(len(df)*0.7)
    insample, oos = df.iloc[:split].copy(), df.iloc[split:].copy()

    # train + probs
    model = train_rf(insample)
    oos   = add_prob_up(model, oos)

    # baseline
    results = {}
    for H in horizons:
        results[str(H)] = backtest_long(oos, P, H)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {"timestamp": stamp, "params": P, "horizons": results}
    save_run_summary(summary, stamp)
    append_history(stamp, summary)

    print("\n— Baseline (loaded from JSON) —")
    for H in horizons:
        r = results[str(H)]
        print(f"{H}d → Net {r['net']:.2f}, PF {r['pf']:.2f}, Win% {r['win_rate']:.2f}, "
              f"Trades {r['trades']}, MDD {r['mdd']:.2f}, Score {r['score']:.2f}")

    # stress
    stress = []
    for cm in (0.5,1.0,1.5):
        for sm in (0.5,1.0,1.5):
            global COST, SLIP
            oldC, oldS = COST, SLIP
            COST = 0.0002 * cm
            SLIP = 0.0001 * sm
            r90 = backtest_long(oos, P, 90)
            r180= backtest_long(oos, P, 180)
            stress.append({"cost_mult":cm,"slip_mult":sm,
                           "net90":r90["net"],"mdd90":r90["mdd"],"score90":r90["score"],
                           "net180":r180["net"],"mdd180":r180["mdd"],"score180":r180["score"]})
            COST, SLIP = oldC, oldS
    df_stress = pd.DataFrame(stress)
    print("\n— Cost/slippage stress (quick) —")
    print(df_stress.round(3).to_string(index=False))

    # Optional mini-grid tighten step
    if minigrid:
        base90, base180 = results["90"], results["180"]
        best, _ = mini_grid_tighten(oos, P, base90, base180)
        if best and autolock:
            save_params(params_path, best["params"])
            print(f"\n✅ Auto-locked params to {params_path}")

    lessons_learned()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str,
                        default=os.path.join(PROFILES_DIR, "profileA_binance_4h.json"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[90,180])
    parser.add_argument("--minigrid", action="store_true", help="run mini-grid tighten with guardrails")
    parser.add_argument("--autolock", action="store_true", help="write best mini-grid params back to profile JSON")
    args = parser.parse_args()
    run_once(args.profile, horizons=tuple(args.horizons), minigrid=args.minigrid, autolock=args.autolock)

if __name__ == "__main__":
    main()