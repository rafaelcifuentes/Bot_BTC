# swing_4h_forward_diamond.py
# Diamante 4h — RF prob + gestión ATR con TP1/TP2/Parcial
# Incluye: SAFE vs LEAKY, walk-forward robusto, freeze/max_bars, CSVs, y sweep de threshold.

import os, time, json, argparse, math
from datetime import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import ccxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------- Parámetros por defecto ----------
BEST_P = {
    "sl_atr_mul":  1.2883725950523068,
    "tp1_atr_mul": 0.746647669156696,
    "tp2_atr_mul": 6.37765361639344,
    "partial_pct": 0.7553906921260284,
    "threshold":   0.5639752144560473,
}
SYMBOL    = "BTC-USD"
PERIOD    = "730d"
INTERVAL  = "4h"
FEATURES  = ["ema_fast", "ema_slow", "rsi", "atr"]
COST      = 0.0002  # comisión proporcional en salida final
SLIP      = 0.0001  # deslizamiento proporcional en entrada
RISK_PERC = 0.01    # % del equity a arriesgar por trade
EQUITY0   = 10000.0

# ---------- Utils ----------
def _normalize_utc_naive_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Asegura índice naive en UTC (sin tz)."""
    if getattr(idx, "tz", None) is None:
        return pd.to_datetime(idx).tz_localize("UTC").tz_localize(None)
    else:
        return pd.to_datetime(idx).tz_convert("UTC").tz_localize(None)

def _fetch_ccxt(exchange_id: str = "binanceus", symbol_ccxt: str | None = None,
                timeframe: str = "4h", days: int = 730) -> pd.DataFrame:
    """Descarga OHLCV vía ccxt con paginación 'limit=1500' (comportamiento restaurado)."""
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    if symbol_ccxt is None:
        symbol_ccxt = "BTC/USD" if exchange_id == "binanceus" else "BTC/USDT"
    since = int((pd.Timestamp.utcnow() - pd.Timedelta(days=days)).timestamp() * 1000)
    all_rows, limit_req = [], 1500
    while True:
        ohlcv = ex.fetch_ohlcv(symbol_ccxt, timeframe=timeframe, since=since, limit=limit_req)
        if not ohlcv:
            break
        all_rows += ohlcv
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < limit_req:
            break
        time.sleep(ex.rateLimit / 1000)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df.index = _normalize_utc_naive_index(df.index)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

def _parse_horizons(s: str | None):
    if not s: return (30,60,90)
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())

def _parse_sweep(s: str):
    # ejemplo: "0.56,0.58,0.60"  o "0.56:0.64:0.01" (ini:fin:paso)
    s = s.strip()
    if ":" in s:
        a,b,st = (float(x) for x in s.split(":"))
        arr = []
        v = a
        while v <= b + 1e-12:
            arr.append(round(v, 10))
            v += st
        return arr
    else:
        return [float(x.strip()) for x in s.split(",") if x.strip()]

# ---------- Carga + features ----------
def load_and_feat(symbol=SYMBOL, period=PERIOD, interval=INTERVAL,
                  skip_yf=False, freeze_end: str|None=None, max_bars: int|None=None) -> pd.DataFrame:
    df = pd.DataFrame()
    exchange_id = os.getenv("EXCHANGE", "binanceus").strip().lower()
    if not skip_yf:
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
            if df is None or df.empty:
                df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False, actions=False,
                                               prepost=False, back_adjust=False)
        except Exception as e:
            print(f"⚠️ yfinance lanzó excepción: {e}")
    if skip_yf or df is None or df.empty:
        base = symbol.replace("-USD", "/USD") if exchange_id == "binanceus" else symbol.replace("-USD", "/USDT")
        print(f"⚠️ yfinance no se usó (--skip_yf). Intentando ccxt/{exchange_id} con {base}…")
        try:
            days = int(period.rstrip("d"))
        except Exception:
            days = 730
        df = _fetch_ccxt(exchange_id=exchange_id, symbol_ccxt=base, timeframe=interval, days=days)
        if df.empty:
            raise RuntimeError(f"No se pudo obtener datos con ccxt/{exchange_id}.")
    # normaliza columnas
    if "open" not in [c.lower() for c in df.columns]:
        if hasattr(df.columns, "get_level_values"):
            df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df.index = _normalize_utc_naive_index(df.index)
    df = df[["open","high","low","close","volume"]].dropna()

    # freeze / cap (sobre crudas)
    raw_start, raw_end = df.index.min(), df.index.max()
    if freeze_end:
        fe = pd.Timestamp(freeze_end).tz_localize("UTC").tz_localize(None)
        df = df.loc[:fe]
    if max_bars:
        df = df.tail(int(max_bars))
    print(f"Rango bruto: {raw_start} → {raw_end}  | velas_brutas={len(df)}")

    # features
    df["ema_fast"] = ta.ema(df["close"], length=12)
    df["ema_slow"] = ta.ema(df["close"], length=26)
    df["rsi"]      = ta.rsi(df["close"], length=14)
    df["atr"]      = ta.atr(df["high"], df["low"], df["close"], length=14)
    df = df.dropna()
    print(f"Rango post-features: {df.index.min()} → {df.index.max()}  | velas={len(df)}")
    return df

# ---------- Modelado ----------
def _prepare_xy(df: pd.DataFrame, shift_features: bool):
    df_ = df.copy()
    # target: up/down del próximo cierre
    delta = df_["close"].shift(-1) - df_["close"]
    target = np.sign(delta)
    # features (shift para variante segura)
    X = df_[FEATURES].shift(1) if shift_features else df_[FEATURES]
    mask = target.notna() & (target != 0)
    # asegurar no NaN en X
    mask = mask & X.notna().all(axis=1)
    X = X.loc[mask].astype(float)
    y = target.loc[mask].astype(int)
    # ambas clases?
    classes = set(np.unique(y))
    if not ({-1, 1} <= classes):
        counts = {int(k): int((y == k).sum()) for k in classes}
        raise ValueError(f"Dataset sin ambas clases. Presentes: {sorted(classes)} conteos: {counts}")
    return X, y

def train_model_from_xy(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            class_weight="balanced", random_state=42,
            n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=-1
        )),
    ])
    pipe.fit(X, y)
    return pipe

def _predict_probas(model: Pipeline, df: pd.DataFrame, shift_features: bool) -> pd.DataFrame:
    X_all = df[FEATURES].shift(1) if shift_features else df[FEATURES]
    X_all = X_all.dropna()
    probs = model.predict_proba(X_all)
    classes = list(model.classes_)
    up_idx  = classes.index(1)
    dn_idx  = classes.index(-1)
    out = pd.DataFrame({
        "p_up": probs[:, up_idx],
        "p_dn": probs[:, dn_idx],
    }, index=X_all.index)
    return out

# ---------- Backtest ----------
def _simulate_period(df: pd.DataFrame, probas: pd.DataFrame, days: int, p: dict, long_only=False):
    # usamos solo el 'tail(n)' del conjunto de test para simular
    n = int(days * 6)  # 6 velas de 4h por día
    period_idx = probas.index.intersection(df.index)
    period_idx = period_idx.sort_values()
    period = df.loc[period_idx].tail(n).copy()
    probas = probas.loc[period.index].copy()
    if period.empty:
        return {"days": days, "net": 0.0, "pf": np.nan, "win_rate": 0.0, "trades": 0, "mdd": 0.0}

    equity = EQUITY0
    peak   = EQUITY0
    mdd    = 0.0
    trades = []
    pos = None

    for ts, r in period.iterrows():
        p_up = probas.loc[ts, "p_up"]
        p_dn = probas.loc[ts, "p_dn"]

        if pos is None:
            long_sig  = (p_up > p["threshold"]) and (r["close"] > r["ema_slow"])
            short_sig = (p_dn > p["threshold"]) and (r["close"] < r["ema_slow"]) and (not long_only)

            if not (long_sig or short_sig):
                # track mdd sin cambios
                peak = max(peak, equity)
                mdd = min(mdd, (equity - peak) / peak)
                continue

            d = 1 if long_sig else -1
            entry = r["open"] * (1 + SLIP * d)
            atr0  = r["atr"]
            stop  = entry - d * p["sl_atr_mul"] * atr0
            tp1   = entry + d * p["tp1_atr_mul"] * atr0
            tp2   = entry + d * p["tp2_atr_mul"] * atr0
            if d * (stop - entry) >= 0:
                peak = max(peak, equity)
                mdd = min(mdd, (equity - peak) / peak)
                continue
            risk_per_unit = abs(entry - stop)
            if risk_per_unit <= 0:
                peak = max(peak, equity)
                mdd = min(mdd, (equity - peak) / peak)
                continue
            size = (equity * RISK_PERC) / risk_per_unit
            pos = {"d": d, "e": entry, "s": stop, "t1": tp1, "t2": tp2, "sz": size,
                   "hm": entry, "p1": False, "atr0": atr0}
            peak = max(peak, equity)
            mdd = min(mdd, (equity - peak) / peak)
        else:
            d, e = pos["d"], pos["e"]
            pos["hm"] = max(pos["hm"], r["high"]) if d == 1 else min(pos["hm"], r["low"])
            exit_p = None

            # parcial en TP1
            if not pos["p1"]:
                hit_tp1 = (d == 1 and r["high"] >= pos["t1"]) or (d == -1 and r["low"] <= pos["t1"])
                if hit_tp1:
                    pnl = (pos["t1"] - e) * d * (pos["sz"] * p["partial_pct"])
                    equity += pnl
                    trades.append(pnl)
                    pos["sz"] *= (1 - p["partial_pct"])
                    pos["p1"] = True
                    peak = max(peak, equity)
                    mdd = min(mdd, (equity - peak) / peak)

            # tp2
            hit_tp2 = (d == 1 and r["high"] >= pos["t2"]) or (d == -1 and r["low"] <= pos["t2"])
            if hit_tp2:
                exit_p = pos["t2"]

            # trailing stop con ATR fijo
            new_stop = pos["hm"] - d * p["sl_atr_mul"] * pos["atr0"]
            if (d == 1 and new_stop > pos["s"]) or (d == -1 and new_stop < pos["s"]):
                pos["s"] = new_stop

            # salida por stop
            stop_hit = (d == 1 and r["low"] <= pos["s"]) or (d == -1 and r["high"] >= pos["s"])
            if stop_hit:
                exit_p = pos["s"]

            if exit_p is not None:
                pnl = (exit_p - e) * d * pos["sz"] - exit_p * pos["sz"] * COST
                equity += pnl
                trades.append(pnl)
                pos = None
                peak = max(peak, equity)
                mdd = min(mdd, (equity - peak) / peak)
            else:
                peak = max(peak, equity)
                mdd = min(mdd, (equity - peak) / peak)

    arr = np.array(trades, dtype=float)
    if arr.size == 0:
        net = 0.0; pf = np.nan; wr = 0.0; ntr = 0
    else:
        net = float(arr.sum())
        gains = arr[arr > 0].sum()
        losses = -arr[arr < 0].sum()
        pf = float(gains / losses) if losses > 0 else np.inf
        wr = float((arr > 0).mean() * 100.0)
        ntr = int(arr.size)
    return {"days": days, "net": net, "pf": pf, "win_rate": wr, "trades": ntr, "mdd": float(mdd)}

def _run_core(df: pd.DataFrame, horizons, params, split=0.7, long_only=False, label="SAFE"):
    cut = int(len(df) * split)
    train = df.iloc[:cut]
    # prepara X,y
    shift_features = (label == "SAFE")
    X, y = _prepare_xy(train, shift_features=shift_features)
    model = train_model_from_xy(X, y)
    # probas para TODO df y luego nos quedamos con test tail(n)
    probas_all = _predict_probas(model, df, shift_features=shift_features)
    probas_all = probas_all.loc[probas_all.index.intersection(df.index)]
    probas_test = probas_all.loc[df.index[cut:]]
    results = [_simulate_period(df.loc[probas_test.index], probas_test, d, params, long_only=long_only) for d in horizons]
    return pd.DataFrame(results).set_index("days")

def _buyhold_net(df: pd.DataFrame, idx: pd.Index, days: int):
    n = int(days * 6)
    sl = df.loc[idx].tail(n)
    if sl.empty:
        return 0.0
    first = sl["open"].iloc[0]
    last  = sl["close"].iloc[-1]
    roi = (last / first - 1.0)
    return float(roi * EQUITY0)

# ---------- Walk-forward ----------
def _walk_forward(df: pd.DataFrame, k: int, horizons, params, shift_features=True, long_only=False):
    # splits temporales iguales
    idx = df.index
    n = len(idx)
    if k < 2:
        raise ValueError("--walk_k debe ser >=2")
    fold_edges = [int(n * i / k) for i in range(1, k+1)]
    rows = []
    for fold, edge in enumerate(fold_edges, start=1):
        train_idx = idx[:edge]
        test_idx  = idx[edge:]
        if len(train_idx) < 100 or len(test_idx) < 50:
            continue
        X, y = _prepare_xy(df.loc[train_idx], shift_features=shift_features)
        model = train_model_from_xy(X, y)
        probas_all = _predict_probas(model, df, shift_features=shift_features)
        # nos aseguramos de usar la intersección segura de índices para evitar KeyError
        test_idx_i = probas_all.index.intersection(test_idx)
        if len(test_idx_i) < 10:
            continue
        for d in horizons:
            n_tail = int(d * 6)
            idx_tail = test_idx_i[-n_tail:] if len(test_idx_i) >= n_tail else test_idx_i
            probas = probas_all.loc[idx_tail]
            metrics = _simulate_period(df, probas, d, params, long_only=long_only)
            rows.append({"fold": fold, **metrics})
    if not rows:
        return pd.DataFrame(columns=["fold","days","net","pf","win_rate","trades","mdd"]).set_index(["fold","days"])
    wf_df = pd.DataFrame(rows).set_index(["fold","days"]).sort_index()
    return wf_df

def _wf_summary(wf_df: pd.DataFrame):
    if wf_df.empty:
        return pd.DataFrame(columns=["days","net_sum","pf_w","win_rate_w","mdd_min"]).set_index("days")
    def _agg(g):
        w = g["trades"].clip(lower=1).astype(float)
        pf_w = (g["pf"] * w).sum() / w.sum()
        wr_w = (g["win_rate"] * w).sum() / w.sum()
        mdd_min = g["mdd"].min()
        net_sum = g["net"].sum()
        return pd.Series({"net_sum": net_sum, "pf_w": pf_w, "win_rate_w": wr_w, "mdd_min": mdd_min})
    # pandas future warning: excluimos columnas de grouping
    out = (
        wf_df.reset_index()
             .groupby("days", as_index=False)
             .apply(lambda g: _agg(g))
             .reset_index(drop=True)
             .set_index("days")
    )
    return out

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=SYMBOL)
    ap.add_argument("--period", default=PERIOD)
    ap.add_argument("--interval", default=INTERVAL)
    ap.add_argument("--skip_yf", action="store_true")
    ap.add_argument("--freeze_end", type=str, default=None)
    ap.add_argument("--max_bars", type=int, default=None)
    ap.add_argument("--horizons", type=str, default="30,60,90")
    ap.add_argument("--split", type=float, default=0.7)
    ap.add_argument("--long_only", action="store_true")
    ap.add_argument("--no_intrabarpredict", action="store_true",
                    help="Activa comparativa SIN shift(1) (leaky) además de la segura.")
    ap.add_argument("--walk_k", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default=None)
    # tuning de parámetros
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--sl_atr_mul", type=float, default=None)
    ap.add_argument("--tp1_atr_mul", type=float, default=None)
    ap.add_argument("--tp2_atr_mul", type=float, default=None)
    ap.add_argument("--partial_pct", type=float, default=None)
    ap.add_argument("--best_p_json", type=str, default=None)
    # NUEVO: sweep de threshold
    ap.add_argument("--sweep_threshold", type=str, default=None,
                    help='Ej: "0.56,0.58,0.60" o "0.56:0.64:0.01" (ini:fin:paso). SAFE-only.')
    args = ap.parse_args()

    horizons = _parse_horizons(args.horizons)

    # parámetros efectivos
    params = BEST_P.copy()
    if args.best_p_json:
        try:
            params.update(json.loads(args.best_p_json))
        except Exception as e:
            print("⚠️ best_p_json inválido:", e)
    for k in ["threshold","sl_atr_mul","tp1_atr_mul","tp2_atr_mul","partial_pct"]:
        v = getattr(args, k)
        if v is not None:
            params[k] = v

    # Datos
    df = load_and_feat(args.symbol, args.period, args.interval,
                       skip_yf=args.skip_yf, freeze_end=args.freeze_end, max_bars=args.max_bars)

    # ----- Sweep de threshold (SAFE-only) -----
    if args.sweep_threshold:
        grid = _parse_sweep(args.sweep_threshold)
        print(f"\n== Sweep de threshold (SAFE) ==\nGrid: {grid}")
        rows = []
        # Entrenamos 1 sola vez para SAFE con features shift(1) (entreno con split)
        cut = int(len(df)*args.split)
        Xs, ys = _prepare_xy(df.iloc[:cut], shift_features=True)
        model = train_model_from_xy(Xs, ys)
        probas_all = _predict_probas(model, df, shift_features=True)
        probas_test = probas_all.loc[df.index[cut:]]
        for th in grid:
            p_local = params.copy()
            p_local["threshold"] = float(th)
            for d in horizons:
                m = _simulate_period(df.loc[probas_test.index], probas_test, d, p_local, long_only=args.long_only)
                rows.append({"threshold": th, **m})
        sweep_df = pd.DataFrame(rows)
        # pivot por days con métricas clave
        sweep_out = []
        for th, g in sweep_df.groupby("threshold"):
            rec = {"threshold": th}
            for d in sorted(g["days"].unique()):
                gi = g[g["days"]==d].iloc[0]
                rec.update({
                    f"pf_{d}d":      gi["pf"],
                    f"wr_{d}d":      gi["win_rate"],
                    f"mdd_{d}d":     gi["mdd"],
                    f"net_{d}d":     gi["net"],
                    f"trades_{d}d":  gi["trades"],
                })
            sweep_out.append(rec)
        sweep_tbl = pd.DataFrame(sweep_out).sort_values("threshold")
        print("\n== Resultado sweep (SAFE) ==")
        with pd.option_context("display.float_format", lambda x: f"{x:,.3f}"):
            print(sweep_tbl)

        # guardar CSV
        ts = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("reports", exist_ok=True)
        path = f"reports/swing4h_sweep_threshold_{ts}.csv"
        sweep_tbl.to_csv(path, index=False)
        print("CSV sweep ->", path)
        # seguimos a reportes base también (por si piden ambos)

    # ----- corrida SAFE -----
    print("\n== Variante SEGURA (features shift(1)) ==")
    res_safe = _run_core(df, horizons, params, split=args.split, long_only=args.long_only, label="SAFE")
    with pd.option_context("display.float_format", lambda x: f"{x:,.3f}"):
        print(res_safe)

    # Buy & Hold (sobre ventana de test 60d aprox -> calculamos para el máximo de horizons)
    cut = int(len(df)*args.split)
    bh_days = max(horizons)
    bh_net = _buyhold_net(df, df.index[cut:], bh_days)
    print(f"\nBuy&Hold en ~{bh_days*4/24:.1f} días de test: net={bh_net:,.2f} USD")

    # guardados CSV (principal + extendido)
    enriched = res_safe.copy()
    if args.out_csv:
        out_base = args.out_csv or "reports/swing4h_metrics.csv"
        os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
        enriched.to_csv(out_base, index=True)
        enriched.assign(roi_pct=100*enriched["net"]/EQUITY0).to_csv(out_base.replace(".csv","_plus.csv"), index=True)
        print("CSV ->", out_base)
        print("CSV+ ->", out_base.replace(".csv","_plus.csv"))

    # ----- comparativa LEAKY opcional -----
    if args.no_intrabarpredict:
        print("\n== Variante SIN shift(1) (leaky) (sin shift(1)) ==")
        res_leaky = _run_core(df, horizons, params, split=args.split, long_only=args.long_only, label="LEAKY")
        with pd.option_context("display.float_format", lambda x: f"{x:,.3f}"):
            print(res_leaky)

        # comparativa lado a lado
        comp = pd.DataFrame({
            "net_safe":     res_safe["net"],
            "pf_safe":      res_safe["pf"],
            "win_rate_safe":res_safe["win_rate"],
            "trades_safe":  res_safe["trades"],
            "mdd_safe":     res_safe["mdd"],
            "net_leaky":    res_leaky["net"],
            "pf_leaky":     res_leaky["pf"],
            "win_rate_leaky":res_leaky["win_rate"],
            "trades_leaky": res_leaky["trades"],
            "mdd_leaky":    res_leaky["mdd"],
        })
        print("\n=== Comparativa SAFE vs LEAKY (lado a lado) ===")
        with pd.option_context("display.float_format", lambda x: f"{x:,.3f}"):
            print(comp)
        ts = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("reports", exist_ok=True)
        comp.to_csv(f"reports/swing4h_compare_safe_vs_leaky_{ts}.csv")
        print("CSV comparativa ->", f"reports/swing4h_compare_safe_vs_leaky_{ts}.csv")

    # ----- Walk-forward opcional -----
    if args.walk_k and args.walk_k >= 2:
        print("\n=== Walk-forward (folds x horizontes) — SEGURA ===")
        wf_safe = _walk_forward(df, args.walk_k, horizons, params, shift_features=True, long_only=args.long_only)
        with pd.option_context("display.float_format", lambda x: f"{x:,.3f}"):
            print(wf_safe)
        print("\n=== Resumen WF (ponderado por #trades) — SEGURA ===")
        summary_safe = _wf_summary(wf_safe)
        with pd.option_context("display.float_format", lambda x: f"{x:,.6f}"):
            print(summary_safe)
        ts = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("reports", exist_ok=True)
        wf_safe.to_csv(f"reports/swing4h_wf_secure_{ts}.csv")
        summary_safe.to_csv(f"reports/swing4h_wf_secure_summary_{ts}.csv")
        print("CSV WF ->", f"reports/swing4h_wf_secure_{ts}.csv")
        print("CSV WF summary ->", f"reports/swing4h_wf_secure_summary_{ts}.csv")

        if args.no_intrabarpredict:
            print("\n=== Walk-forward (folds x horizontes) — SIN shift(1) ===")
            wf_leak = _walk_forward(df, args.walk_k, horizons, params, shift_features=False, long_only=args.long_only)
            with pd.option_context("display.float_format", lambda x: f"{x:,.3f}"):
                print(wf_leak)
            print("\n=== Resumen WF (ponderado por #trades) — SIN shift(1) ===")
            summary_leak = _wf_summary(wf_leak)
            with pd.option_context("display.float_format", lambda x: f"{x:,.6f}"):
                print(summary_leak)
            wf_leak.to_csv(f"reports/swing4h_wf_leaky_{ts}.csv")
            summary_leak.to_csv(f"reports/swing4h_wf_leaky_summary_{ts}.csv")
            print("CSV WF ->", f"reports/swing4h_wf_leaky_{ts}.csv")
            print("CSV WF summary ->", f"reports/swing4h_wf_leaky_summary_{ts}.csv")


if __name__ == "__main__":
    main()