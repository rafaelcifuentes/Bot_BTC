# diamante_walkforward_and_sensitivity.py
import sys, json, warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------- Settings ----------------
SYMBOL      = "BTC-USD"
PERIOD_4H   = "730d"
INTERVAL_4H = "4h"
PERIOD_1D   = "800d"
INTERVAL_1D = "1d"

FEATURES = ["ema_fast","ema_slow","rsi","atr","adx4h","adx1d"]

COST       = 0.0002
SLIP       = 0.0001
RISK_PERC  = 0.005  # fixed risk sizing for this script

# ATR engine (kept stable)
ENGINE = dict(
    sl_atr_mul  = 1.2,
    tp1_atr_mul = 1.6,
    tp2_atr_mul = 4.5,
    trail_mul   = 1.0,
    partial_pct = 0.4,
)

# Baseline filters
BASELINE = dict(
    threshold     = 0.52,
    adx4_min      = 8,
    adx1d_min     = 3,
    require_slope = False,
    cooldown_tp2  = 0,
    cooldown_stop = 2,
)

HORIZONS = [90, 180]  # days to report

# ---------------- Data & Features ----------------
def download_data():
    print("🔄 Downloading 4h + 1D data…")
    df4 = yf.download(SYMBOL, period=PERIOD_4H, interval=INTERVAL_4H, progress=False, auto_adjust=False)
    if df4 is None or df4.empty:
        sys.exit("Failed to download 4h data.")

    if isinstance(df4.columns, pd.MultiIndex):
        df4.columns = df4.columns.get_level_values(0)
    df4.columns = df4.columns.str.lower()
    df4.index = pd.to_datetime(df4.index).tz_localize(None)

    dfd = yf.download(SYMBOL, period=PERIOD_1D, interval=INTERVAL_1D, progress=False, auto_adjust=False)
    if dfd is None or dfd.empty:
        sys.exit("Failed to download 1d data.")
    if isinstance(dfd.columns, pd.MultiIndex):
        dfd.columns = dfd.columns.get_level_values(0)
    dfd.columns = dfd.columns.str.lower()
    dfd.index = pd.to_datetime(dfd.index).tz_localize(None)

    # Daily ADX
    adx_d = ta.adx(dfd["high"], dfd["low"], dfd["close"], length=14)
    if adx_d is not None and not adx_d.empty:
        col = next((c for c in adx_d.columns if "adx" in c.lower()), None)
        dfd["adx1d"] = adx_d[col] if col else np.nan
    else:
        dfd["adx1d"] = np.nan

    # Merge onto 4h
    df4["adx1d"] = dfd["adx1d"].reindex(df4.index, method="ffill")

    # 4h indicators
    df4["ema_fast"] = ta.ema(df4["close"], length=12)
    df4["ema_slow"] = ta.ema(df4["close"], length=26)
    df4["rsi"]      = ta.rsi(df4["close"], length=14)
    df4["atr"]      = ta.atr(df4["high"], df4["low"], df4["close"], length=14)
    adx4 = ta.adx(df4["high"], df4["low"], df4["close"], length=14)
    if adx4 is not None and not adx4.empty:
        col4 = next((c for c in adx4.columns if "adx" in c.lower()), None)
        df4["adx4h"] = adx4[col4] if col4 else 0.0
    else:
        df4["adx4h"] = 0.0

    # slope flag
    df4["ema_slow_prev"] = df4["ema_slow"].shift(1)
    df4["slope_up"] = (df4["ema_slow"] > df4["ema_slow_prev"]).astype(int)

    # target
    df4["target"] = np.sign(df4["close"].shift(-1) - df4["close"]).fillna(0.0).astype(int)
    df4.dropna(inplace=True)
    return df4

# ---------------- Model ----------------
def train_model(df_train: pd.DataFrame):
    X = df_train[FEATURES]
    y = df_train["target"]
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))
    ])
    pipe.fit(X, y)
    return pipe

def infer_prob_up(model, df: pd.DataFrame) -> np.ndarray:
    probs = model.predict_proba(df[FEATURES])
    classes = list(model.classes_)
    if 1 in classes:
        return probs[:, classes.index(1)]
    return np.full(len(df), 0.5, dtype=float)

# ---------------- ATR Backtest (long-only) ----------------
def backtest_long(df: pd.DataFrame, p: dict, days: int):
    # requires: open, high, low, close, atr, adx4h, adx1d, ema_slow, slope_up, prob_up
    n = int(days * 6)
    chunk = df.tail(n).copy()

    eq_closed = 10000.0
    eq_curve  = []
    pos = None
    trades_pnl = []
    closed_trades = 0
    cooldown = 0
    last_exit_tp2 = False

    for _, r in chunk.iterrows():
        if cooldown > 0:
            cooldown -= 1

        # manage open
        if pos is not None:
            pos["hm"] = max(pos["hm"], r["high"])
            exit_price = None
            exit_reason = None

            # partial TP1
            if (not pos["p1"]) and (r["high"] >= pos["t1"]):
                part_sz = pos["sz"] * p["partial_pct"]
                pnl  = (pos["t1"] - pos["e"]) * part_sz
                cost = pos["t1"] * part_sz * COST
                trades_pnl.append(pnl - cost)
                eq_closed += (pnl - cost)
                pos["sz"] *= (1 - p["partial_pct"])
                pos["p1"] = True

            # TP2
            if r["high"] >= pos["t2"]:
                exit_price = pos["t2"]
                exit_reason = "tp2"

            # trail
            new_stop = pos["hm"] - p["trail_mul"] * pos["atr0"]
            if new_stop > pos["s"]:
                pos["s"] = new_stop
            if r["low"] <= pos["s"] and exit_price is None:
                exit_price = pos["s"]
                exit_reason = "stop"

            if exit_price is not None:
                pnl  = (exit_price - pos["e"]) * pos["sz"]
                cost = exit_price * pos["sz"] * COST
                trades_pnl.append(pnl - cost)
                eq_closed += (pnl - cost)
                closed_trades += 1
                if exit_reason == "tp2" and p.get("cooldown_tp2", 0) > 0:
                    cooldown = p["cooldown_tp2"]
                elif exit_reason != "tp2" and p.get("cooldown_stop", 0) > 0:
                    cooldown = p["cooldown_stop"]
                pos = None

        # try entry
        if pos is None and cooldown == 0:
            if not (r["prob_up"] > p["threshold"]):
                eq_curve.append(eq_closed); continue
            if not (r["close"] > r["ema_slow"]):
                eq_curve.append(eq_closed); continue
            if not (r["adx4h"] >= p["adx4_min"]):
                eq_curve.append(eq_closed); continue
            if not (pd.notna(r["adx1d"]) and r["adx1d"] >= p["adx1d_min"]):
                eq_curve.append(eq_closed); continue
            if p.get("require_slope", False) and r["slope_up"] != 1:
                eq_curve.append(eq_closed); continue

            atr0 = r["atr"]
            if atr0 <= 0:
                eq_curve.append(eq_closed); continue

            entry = r["open"] * (1 + SLIP)
            stop  = entry - p["sl_atr_mul"] * atr0
            tp1   = entry + p["tp1_atr_mul"] * atr0
            tp2   = entry + p["tp2_atr_mul"] * atr0
            if stop >= entry:
                eq_curve.append(eq_closed); continue

            sz = (eq_closed * RISK_PERC) / (entry - stop)
            entry_cost = -entry * sz * COST
            trades_pnl.append(entry_cost)
            eq_closed += entry_cost

            pos = {"e": entry, "s": stop, "t1": tp1, "t2": tp2,
                   "sz": sz, "hm": entry, "p1": False, "atr0": atr0}

        # mark-to-market
        unreal = 0.0
        if pos is not None:
            unreal = (r["close"] - pos["e"]) * pos["sz"]
        eq_curve.append(eq_closed + unreal)

    if not trades_pnl:
        return dict(net=0.0, pf=0.0, win_rate=0.0, trades=0, mdd=0.0, score=0.0)

    arr = np.array(trades_pnl, dtype=float)
    net = float(arr.sum())
    gains  = arr[arr > 0]
    losses = arr[arr < 0]
    pf = float(gains.sum() / abs(losses.sum())) if losses.size else np.inf
    wr = float(len(gains) / len(arr) * 100.0)

    eq = np.array(eq_curve, dtype=float)
    peak = np.maximum.accumulate(eq) if eq.size else np.array([0.0])
    mdd = float((peak - eq).max()) if eq.size else 0.0
    score = float(net / (mdd + 1.0))
    return dict(net=net, pf=pf, win_rate=wr, trades=int(len(arr)), mdd=mdd, score=score)

# ---------------- Helpers ----------------
def train_and_prob(df_all: pd.DataFrame):
    split = int(len(df_all) * 0.7)
    df_train = df_all.iloc[:split].copy()
    df_oos   = df_all.iloc[split:].copy()
    model = train_model(df_train)
    df_oos["prob_up"] = infer_prob_up(model, df_oos)
    return df_oos

def eval_params(df_oos, params):
    res = {}
    merged = {**ENGINE, **params}
    for d in HORIZONS:
        res[d] = backtest_long(df_oos, merged, d)
    return res

def print_row(prefix, r):
    print(f"{prefix} → Net {r['net']:.2f}, PF {r['pf']:.2f}, Win% {r['win_rate']:.2f}, "
          f"Trades {r['trades']}, MDD {r['mdd']:.2f}, Score {r['score']:.2f}")

def funnel_report(df_oos, params, days=180):
    chunk = df_oos.tail(int(days*6)).copy()
    N = len(chunk)
    thr = params["threshold"]
    a4  = params["adx4_min"]
    a1d = params["adx1d_min"]
    slope_on = params.get("require_slope", False)

    step1 = chunk[chunk["prob_up"] > thr]
    step2 = step1[step1["close"] > step1["ema_slow"]]
    step3 = step2[step2["adx4h"] >= a4]
    step4 = step3[step3["adx1d"].notna() & (step3["adx1d"] >= a1d)]
    step5 = step4[step4["slope_up"] == 1] if slope_on else step4

    print("\n— Funnel (last 180d bars) —")
    print(f"Total bars: {N}")
    print(f"prob_up > {thr:.2f}: {len(step1)} ({len(step1)/N*100:.1f}%)")
    print(f"close > ema_slow: {len(step2)} ({len(step2)/N*100:.1f}%)")
    print(f"adx4h ≥ {a4}: {len(step3)} ({len(step3)/N*100:.1f}%)")
    print(f"adx1d ≥ {a1d}: {len(step4)} ({len(step4)/N*100:.1f}%)")
    if slope_on:
        print(f"slope_up = 1: {len(step5)} ({len(step5)/N*100:.1f}%)")
    else:
        print("slope_up filter: OFF")

# ---------------- Main ----------------
def main():
    df0 = download_data()
    df_oos = train_and_prob(df0)

    # Baseline
    base_res = eval_params(df_oos, BASELINE)
    print("\n— Baseline (current set) —\n")
    for d in HORIZONS:
        print_row(f"{d}d", base_res[d])

    # Distribution & funnel diagnostics
    last180 = df_oos.tail(180*6)
    print("\n— prob_up distribution in last 180d —")
    print(last180["prob_up"].describe().round(6).to_string())

    funnel_report(df_oos, BASELINE, days=180)
    loose = {**BASELINE, "threshold": 0.50}
    funnel_report(df_oos, loose, days=180)

    # Adaptive guardrails
    base90 = base_res[90]["score"]
    base90_tr = base_res[90]["trades"]
    base180_tr = base_res[180]["trades"]
    enforce_90 = (base90_tr >= 25) and (base90 > 0)
    min_trades_180 = max(25, int(0.5 * max(1, base180_tr)))

    # Grid
    thresholds    = [0.50, 0.51, 0.52, 0.54]
    adx4_vals     = [6, 8, 10]
    adx1d_vals    = [0, 3, 5]
    slope_opts    = [False, True]
    cooldowns_tp2 = [0, 2]

    records = []
    for thr in thresholds:
        for a4 in adx4_vals:
            for a1d in adx1d_vals:
                for slope in slope_opts:
                    for cd2 in cooldowns_tp2:
                        cand = dict(
                            threshold=thr,
                            adx4_min=a4,
                            adx1d_min=a1d,
                            require_slope=slope,
                            cooldown_tp2=cd2,
                            cooldown_stop=BASELINE["cooldown_stop"],
                        )
                        res = eval_params(df_oos, cand)
                        r90, r180 = res[90], res[180]
                        status = []
                        if r180["trades"] < min_trades_180:
                            status.append("LOW_TRADES")
                        if enforce_90 and (r90["score"] < 0.9 * base90):
                            status.append("DEGRADE_90")
                        records.append({
                            "threshold": thr, "adx4_min": a4, "adx1d_min": a1d,
                            "require_slope": slope, "cooldown_tp2": cd2,
                            "net90": r90["net"], "pf90": r90["pf"], "wr90": r90["win_rate"],
                            "mdd90": r90["mdd"], "score90": r90["score"], "trades90": r90["trades"],
                            "net180": r180["net"], "pf180": r180["pf"], "wr180": r180["win_rate"],
                            "mdd180": r180["mdd"], "score180": r180["score"], "trades180": r180["trades"],
                            "status": "OK" if not status else ",".join(status)
                        })

    df = pd.DataFrame(records)
    ok = df[df["status"] == "OK"].copy()
    if ok.empty:
        print("\nNo candidate passed adaptive guardrails. Keep baseline.\n")
        return

    ok = ok.sort_values(["score180","score90"], ascending=False)
    top = ok.head(10).copy()

    print("\n— Top 10 by 180d score (after guardrails) —")
    cols = ["threshold","adx4_min","adx1d_min","require_slope","cooldown_tp2",
            "net90","mdd90","score90","trades90","net180","mdd180","score180","trades180","status"]
    print(top[cols].to_string(index=False, justify="center",
                              float_format=lambda x: f"{x:.3f}"))

    # Pick best & recompute full metrics (fix for KeyError: 'pf')
    best = ok.iloc[0].to_dict()
    best_params = dict(
        threshold=float(best["threshold"]),
        adx4_min=int(best["adx4_min"]),
        adx1d_min=int(best["adx1d_min"]),
        require_slope=bool(best["require_slope"]),
        cooldown_tp2=int(best["cooldown_tp2"]),
        cooldown_stop=BASELINE["cooldown_stop"],
    )

    print("\n— Selected best set — [OK (adaptive guardrails)]")
    pretty = {**best_params, **ENGINE, "risk_perc": RISK_PERC}
    print(json.dumps(pretty, indent=2))

    # Recompute full metrics for printing (includes pf & win_rate)
    new_full = eval_params(df_oos, best_params)

    print("\n— 90d / 180d Comparison —")
    for d in HORIZONS:
        print_row(f"Baseline {d}d", base_res[d])
        print_row(f"New set  {d}d", new_full[d])

    # Save JSON
    with open("minigrid_gate_pick.json", "w") as f:
        json.dump(pretty, f, indent=2)

if __name__ == "__main__":
    main()
