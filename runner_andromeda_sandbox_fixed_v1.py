# runner_andromeda_sandbox_fixed_v1.py
# ─────────────────────────────────────────────────────────────────────────────
# Sandbox/Prueba — Andromeda (LONG+SHORT) con parámetros FIJOS por porcentaje
# - No toca V2/V3 ni archivos de perfiles/weights.
# - Carga BTC 4h (ccxt + fallbacks + yfinance), entrena SOLO en train (70%),
#   opera SOLO en test (30%) estilo walk-forward.
# - Señal: prob_up de un RandomForest sencillo; LONG si > THRESHOLD, SHORT si < 1 - THRESHOLD.
# - Gestión de trade: SL/TP por porcentaje sobre el precio de entrada, trailing por porcentaje.
# - Sizing: risk_perc sobre equity y distancia hasta el stop.
# - 1 sola posición a la vez (conflict-guard).
# - Export opcional de trades OOS: --export-trades (CSV).
#
# Parámetros (por defecto):
#   THRESHOLD = 0.55
#   SL_PCT    = 0.025
#   TP_PCT    = 0.05
#   TRAIL_PCT = 0.01
#   RISK_PERC = 0.01
#   COST      = 0.0002
#   SLIP      = 0.0001
#
# También imprime un bloque de "Benchmark hints" para comparar con:
#   Net ≈ 235,772.66 | PF ≈ 1.99 | WR ≈ 57.2% | Trades ≈ 1208  (solo como referencia).
# ─────────────────────────────────────────────────────────────────────────────
import os, json, time, argparse, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Optional deps
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
# Config / Paths
# -----------------------
SYMBOL = "BTC/USDT"
YF_SYMBOL = "BTC-USD"
TIMEFRAME = "4h"
MAX_BARS = 3000
SINCE_DAYS = 720
MIN_BARS_REQ = 800

DEFAULTS = dict(
    THRESHOLD=0.55,
    SL_PCT=0.025,
    TP_PCT=0.05,
    TRAIL_PCT=0.01,
    RISK_PERC=0.01,
    COST=0.0002,
    SLIP=0.0001,
    START_EQUITY=10_000.0,
)

REPORT_DIR = "./reports"
CACHE_DIR = "./cache"
CACHE_FILE = os.path.join(CACHE_DIR, "btc_4h_ohlcv.csv")

BENCHMARK_HINTS = {
    "net": 235_772.66,
    "pf": 1.99,
    "wr": 57.2,
    "trades": 1208
}

SEED = 42

# -----------------------
# Utils
# -----------------------
def ensure_dirs():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def ts_ms(days_back: int) -> int:
    return int((pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)).timestamp() * 1000)

def now_stamp() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

def dedup_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.index.duplicated(keep="last")].sort_index()

def safe_pf(trades: np.ndarray) -> float:
    if trades.size == 0:
        return 0.0
    gains = trades[trades > 0].sum()
    losses = trades[trades < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
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

def parse_freeze_end(s: str) -> pd.Timestamp:
    """
    Convierte string a timestamp "naive" en UTC (sin tzinfo) para indexar el DataFrame.
    Si no pasas argumento, toma el último índice disponible.
    """
    t = pd.Timestamp(s, tz="UTC").tz_convert(None)
    return t

# -----------------------
# Data (ccxt + fallbacks + yfinance)
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
            try:
                time.sleep((ex.rateLimit or 200) / 1000.0)
            except Exception:
                pass
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

def load_4h_stitched() -> Tuple[pd.DataFrame, List[str]]:
    sources, dfs = [], []
    for ex_id, syms in EX_SYMBOLS_ALL.items():
        try:
            d = fetch_ccxt_any(ex_id, syms, TIMEFRAME, SINCE_DAYS)
        except Exception:
            d = pd.DataFrame()
        sources.append(f"{ex_id}: {'ok' if not d.empty else 'empty'}")
        if not d.empty:
            dfs.append(d)
    try:
        d_yf = fetch_yf(YF_SYMBOL, period_days=SINCE_DAYS)
    except Exception:
        d_yf = pd.DataFrame()
    sources.append("yfinance: ok" if not d_yf.empty else "yfinance: empty")
    if not d_yf.empty:
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
# Features + modelo simple
# -----------------------
FEATURES = ["ema_fast","ema_slow","rsi","atr","adx4h"]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ta.ema(out["close"], length=12)
    out["ema_slow"] = ta.ema(out["close"], length=26)
    out["rsi"] = ta.rsi(out["close"], length=14)
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    adx4 = ta.adx(out["high"], out["low"], out["close"], length=14)
    out["adx4h"] = adx4[[c for c in adx4.columns if c.lower().startswith("adx")][0]] if (adx4 is not None and not adx4.empty) else 0.0
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

# -----------------------
# Backtest FIXED % (SL/TP/Trail)
# -----------------------
def run_backtest_percent(df: pd.DataFrame, params: Dict, start_equity: float,
                         cost: float, slip: float, export_trades: bool=False) -> Dict:
    """
    LONG: SL = entry*(1-SL_PCT), TP = entry*(1+TP_PCT)
    SHORT: SL = entry*(1+SL_PCT), TP = entry*(1-TP_PCT)
    Trailing: distancia fija = entry*TRAIL_PCT (sube/baja con high/low).
    """
    eq_closed = float(start_equity)
    trades = []
    eq_curve = []
    pos = None
    trade_rows = []

    th = float(params["THRESHOLD"])
    sl_pct = float(params["SL_PCT"])
    tp_pct = float(params["TP_PCT"])
    tr_pct = float(params["TRAIL_PCT"])
    risk = float(params["RISK_PERC"])

    for ts, r in df.iterrows():
        # Cierre
        if pos is not None:
            exit_price = None
            if pos["dir"] == "long":
                # trailing
                pos["hm"] = max(pos["hm"], r["high"])
                trail_dist = pos["e"] * tr_pct
                new_stop = pos["hm"] - trail_dist
                if new_stop > pos["s"]:
                    pos["s"] = new_stop
                # ¿tocó TP o SL en la vela?
                hit_tp = r["high"] >= pos["tp"]
                hit_sl = r["low"]  <= pos["s"]
                # si ambos, asumir conservador: SL primero
                if hit_sl:
                    exit_price = pos["s"]
                elif hit_tp:
                    exit_price = pos["tp"]
                if exit_price is not None:
                    fee = exit_price * pos["sz"] * cost
                    pnl = (exit_price - pos["e"]) * pos["sz"] - fee
                    trades.append(pnl); eq_closed += pnl
                    if export_trades:
                        trade_rows.append([ts, "LONG", pos["e"], exit_price, pos["sz"], pnl, eq_closed])
                    pos = None
            else:
                # short
                pos["lm"] = min(pos["lm"], r["low"])
                trail_dist = pos["e"] * tr_pct
                new_stop = pos["lm"] + trail_dist
                if new_stop < pos["s"]:
                    pos["s"] = new_stop
                hit_tp = r["low"]  <= pos["tp"]
                hit_sl = r["high"] >= pos["s"]
                if hit_sl:
                    exit_price = pos["s"]
                elif hit_tp:
                    exit_price = pos["tp"]
                if exit_price is not None:
                    fee = exit_price * pos["sz"] * cost
                    pnl = (pos["e"] - exit_price) * pos["sz"] - fee
                    trades.append(pnl); eq_closed += pnl
                    if export_trades:
                        trade_rows.append([ts, "SHORT", pos["e"], exit_price, pos["sz"], pnl, eq_closed])
                    pos = None

        # Apertura (1 sola posición a la vez)
        if pos is None:
            pu = r["prob_up"]
            go_long  = (pu > th) and (r["close"] > r["ema_slow"])
            go_short = (pu < (1 - th)) and (r["close"] < r["ema_slow"])

            if go_long or go_short:
                # prioridad por "más convicción"
                sc_long = pu - th
                sc_short = (1 - th) - pu
                direction = "long" if (go_long and sc_long >= sc_short) else ("short" if go_short else None)
                if direction is not None:
                    if direction == "long":
                        entry = float(r["open"]) * (1 + slip)
                        stop  = entry * (1 - sl_pct)
                        tp    = entry * (1 + tp_pct)
                        if stop < entry:
                            risk_per_unit = entry - stop
                            sz = (eq_closed * risk) / max(risk_per_unit, 1e-12)
                            entry_fee = entry * sz * cost
                            eq_closed -= entry_fee
                            pos = {"dir":"long","e":entry,"s":stop,"tp":tp,"sz":sz,"hm":entry}
                            if export_trades:
                                trade_rows.append([ts, "LONG-OPEN", entry, np.nan, sz, -entry_fee, eq_closed])
                    else:
                        entry = float(r["open"]) * (1 - slip)
                        stop  = entry * (1 + sl_pct)
                        tp    = entry * (1 - tp_pct)
                        if stop > entry:
                            risk_per_unit = stop - entry
                            sz = (eq_closed * risk) / max(risk_per_unit, 1e-12)
                            entry_fee = entry * sz * cost
                            eq_closed -= entry_fee
                            pos = {"dir":"short","e":entry,"s":stop,"tp":tp,"sz":sz,"lm":entry}
                            if export_trades:
                                trade_rows.append([ts, "SHORT-OPEN", entry, np.nan, sz, -entry_fee, eq_closed])

        # Equity mark-to-market
        unreal = 0.0
        if pos is not None:
            if pos["dir"] == "long":
                unreal = (r["close"] - pos["e"]) * pos["sz"]
            else:
                unreal = (pos["e"] - r["close"]) * pos["sz"]
        eq_curve.append(eq_closed + unreal)

    arr = np.array(trades, dtype=float)
    net = float(arr.sum()) if arr.size else 0.0
    pf = safe_pf(arr) if arr.size else 0.0
    wr = float((arr > 0).sum() / len(arr) * 100.0) if arr.size else 0.0
    eq = np.array(eq_curve, dtype=float)
    mdd = float((np.maximum.accumulate(eq) - eq).max()) if eq.size else 0.0
    score = float(net / (mdd + 1e-9)) if mdd > 0 else net
    res = {
        "net": net, "pf": pf, "wr": wr, "trades": int(len(arr)),
        "mdd": mdd, "score": score, "final_equity": float(eq[-1] if eq.size else start_equity)
    }
    if export_trades and trade_rows:
        stamp = now_stamp()
        path = os.path.join(REPORT_DIR, f"sandbox_trades_oos_{stamp}.csv")
        df_t = pd.DataFrame(trade_rows, columns=["time","side","entry","exit","size","pnl","equity_after"])
        df_t.to_csv(path, index=False)
        res["trades_csv"] = path
    return res

# -----------------------
# CLI
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Sandbox Andromeda (LONG+SHORT) con % fijos de SL/TP/Trail")
    ap.add_argument("--freeze-end", type=str, default=None,
                    help='UTC end timestamp (e.g. "2025-08-12 12:05"). Si no, usa el último dato.')
    ap.add_argument("--start-equity", type=float, default=DEFAULTS["START_EQUITY"])
    ap.add_argument("--threshold", type=float, default=DEFAULTS["THRESHOLD"])
    ap.add_argument("--sl-pct", type=float, default=DEFAULTS["SL_PCT"])
    ap.add_argument("--tp-pct", type=float, default=DEFAULTS["TP_PCT"])
    ap.add_argument("--trail-pct", type=float, default=DEFAULTS["TRAIL_PCT"])
    ap.add_argument("--risk-perc", type=float, default=DEFAULTS["RISK_PERC"])
    ap.add_argument("--cost", type=float, default=DEFAULTS["COST"])
    ap.add_argument("--slip", type=float, default=DEFAULTS["SLIP"])
    ap.add_argument("--export-trades", action="store_true")
    return ap.parse_args()

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    ensure_dirs()

    # Carga datos
    print("🔄 Loading 4h data (ccxt + fallbacks + yfinance)…")
    df_raw, sources = load_4h_stitched()
    if df_raw.empty:
        print("No data."); return
    print("✨ Adding features…")
    df = add_features(df_raw).dropna()
    if len(df) < MIN_BARS_REQ:
        print(f"Not enough bars: {len(df)} < {MIN_BARS_REQ}")
        return

    # Corte de ventana por freeze-end (opcional)
    if args.freeze_end:
        t_end = parse_freeze_end(args.freeze_end)
        df = df.loc[:t_end].copy()

    # Split 70/30 (temporal)
    cut = int(len(df) * 0.70)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    # Modelo / prob_up
    print("🧠 Training model on 70% / scoring 30% OOS…")
    model = train_model(train)
    test["prob_up"] = predict_prob_up(model, test)

    # Parámetros fijos
    fixed = dict(
        THRESHOLD=args.threshold,
        SL_PCT=args.sl_pct,
        TP_PCT=args.tp_pct,
        TRAIL_PCT=args.trail_pct,
        RISK_PERC=args.risk_perc
    )

    # Backtest
    res = run_backtest_percent(
        test, fixed, start_equity=args.start_equity,
        cost=args.cost, slip=args.slip, export_trades=args.export_trades
    )

    # Output
    print("\n— Sandbox Andromeda (Fixed %) —")
    print("Params:", {k: (round(v,4) if isinstance(v,float) else v) for k,v in fixed.items()},
          f"| cost={args.cost}, slip={args.slip}, start={args.start_equity:,.2f}")
    print(f"Net Profit     : {res['net']:,.2f}")
    print(f"Profit Factor  : {res['pf']:.2f}")
    print(f"Win Rate       : {res['wr']:.2f}%")
    print(f"Trades         : {res['trades']}")
    print(f"Max DD         : {res['mdd']:.2f}")
    print(f"Score (Net/MDD): {res['score']:.2f}")
    print(f"Final Equity   : {res['final_equity']:,.2f}")
    if "trades_csv" in res:
        print(f"🧾 Saved trades OOS → {res['trades_csv']}")

    # Hints vs benchmark de referencia (no es objetivo, es para comparar)
    print("\n— Benchmark hints (referencia) —")
    d_net = res["net"] - BENCHMARK_HINTS["net"]
    d_pf  = res["pf"]  - BENCHMARK_HINTS["pf"]
    d_wr  = res["wr"]  - BENCHMARK_HINTS["wr"]
    d_trd = res["trades"] - BENCHMARK_HINTS["trades"]
    print(f"Target ~ Net {BENCHMARK_HINTS['net']:,.2f}, PF {BENCHMARK_HINTS['pf']:.2f}, WR {BENCHMARK_HINTS['wr']:.1f}%, Trades {BENCHMARK_HINTS['trades']}")
    print(f"Δ vs target → Net {d_net:,.2f} | PF {d_pf:+.2f} | WR {d_wr:+.2f}pp | Trades {d_trd:+d}")
    print("⚠️ Nota: estos targets son una referencia; los resultados reales dependen del periodo, datos y costos.")

if __name__ == "__main__":
    main()