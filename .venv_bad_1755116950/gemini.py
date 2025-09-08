# swing_4h_ml_rf_with_shorts_refactored.py
# Step 4.3: More trades (threshold 0.52, gap 0.02) + triple‑barrier + auto‑retrain bundle
"""
Versión «diamond‑black» – Paso 4.3 (más operaciones, manteniendo edge)
=====================================================================

Ajustes para subir el número de trades (partiendo de tu OOS positivo: 21 trades, PF≈1.17, MDD≈‑7.9%):
- **THRESHOLD** ↓ a **0.52** (antes 0.55).
- **MIN_PROB_GAP** ↓ a **0.02** (antes 0.05).
- Mantiene target **triple‑barrier** y feature set ampliado.
- Señales **lag=1** y **backtest bar‑a‑bar** sin lookahead; sizing por riesgo.
- **Bundle** con metadatos (features + versión): se **reentrena automático** si cambian.

CLI
---
Puedes sobreescribir parámetros por línea de comandos sin tocar el código.

```bash
python swing_4h_ml_rf_with_shorts_refactored.py --symbol BTC-USD --split 0.7 --retrain \
  --threshold 0.52 --min-prob-gap 0.02

# CSV local con split por fecha
python swing_4h_ml_rf_with_shorts_refactored.py --csv btc_4h.csv --split-date 2024-01-01 --retrain \
  --threshold 0.52 --min-prob-gap 0.02
```
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Parámetros base
# -----------------------------
CODE_VERSION = "v4.3"

DEFAULT_PARAMS = {
    # Señales/gestión
    "THRESHOLD": 0.52,
    "MIN_PROB_GAP": 0.02,
    "SL_PCT": 0.025,
    "TP_PCT": 0.05,
    "TRAIL_PCT": 0.01,
    "RISK_PERC": 0.01,          # 1% del equity por trade
    "COST": 0.0002,             # 2 pb ida+vuelta
    "SLIPPAGE": 0.0001,         # 1 pb por lado

    # Target triple‑barrier
    "H_BARS": 6,                # horizonte máx. (6 velas 4h ≈ 1 día)
    "TP_ATR_MULT": 1.5,
    "SL_ATR_MULT": 1.0,

    # Filtros contexto (puedes dejarlos por defecto)
    "USE_ADX_D1": True,
    "ADX_D1_MIN": 20.0,
    "USE_ATR_NORM": True,
    "ATR_NORM_WIN": 14,
    "ATR_NORM_MIN": 0.8,
    "ATR_NORM_MAX": 1.8,
    "ALLOW_HOURS": None,        # e.g. [4,8,12,16,20] en UTC

    # Features opcionales
    "USE_CYCLICAL_HOUR": True,
}

MODEL_BUNDLE_PATH = Path("rf_swing4h_bundle.joblib")

# Lista base de features (se completará tras ingeniería)
BASE_FEATURES = [
    "atr", "rsi", "ema_50", "ema_200", "adx4",  # clásicos 4h
    "atr_norm", "adx_d1",                         # contexto
    # retornos/volatilidad
    "ret_1", "ret_3", "ret_6", "vola_10",
    # estructura y tendencia
    "dist_ema50", "dist_ema200", "ema_ratio", "ema50_slope", "ema200_slope",
    # bandas y rango
    "bb_width", "range_frac",
    # volumen
    "obv_z",
]

# Campos cíclicos (opcionales)
CYCLICAL_FEATURES = ["hour_sin", "hour_cos"]

# -----------------------------
# Utilidades de datos
# -----------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def load_data(symbol: Optional[str] = None, csv_path: Optional[str] = None,
              period: str = "720d", interval: str = "4h") -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]

    if csv_path:
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
        df = pd.read_csv(p, header=None)
        if df.shape[1] < 6:
            raise ValueError("CSV debe tener >=6 columnas: timestamp open high low close volume")
        df.columns = ["timestamp", *required, *[f"extra_{i}" for i in range(df.shape[1]-6)]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        return df[required].astype(float)

    if not symbol:
        raise ValueError("Debes proporcionar --symbol o --csv")

    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("yfinance devolvió un DataFrame vacío")
    df.index = pd.to_datetime(df.index, utc=True)
    return _flatten_columns(df)[required]


# -----------------------------
# Ingeniería de features (4h) + contexto diario
# -----------------------------

def _safe_get(d: pd.DataFrame, names: Iterable[str]) -> Optional[pd.Series]:
    for n in names:
        if n in d.columns:
            return d[n]
    return None


def add_features(df4h: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df4h.copy()

    # Indicadores 4h con nombres estables
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    adx4 = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx4 is not None and not adx4.empty:
        col_adx4 = _safe_get(adx4, [c for c in adx4.columns if "ADX_" in c or "adx" in c.lower()])
        df["adx4"] = col_adx4 if col_adx4 is not None else np.nan
    else:
        df["adx4"] = np.nan
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_200"] = ta.ema(df["close"], length=200)

    # Retornos y volatilidad
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)
    df["vola_10"] = df["ret_1"].rolling(10).std()

    # Distancias/tendencia
    df["dist_ema50"] = df["close"] / df["ema_50"] - 1
    df["dist_ema200"] = df["close"] / df["ema_200"] - 1
    df["ema_ratio"] = df["ema_50"] / df["ema_200"] - 1
    df["ema50_slope"] = df["ema_50"].pct_change(5)
    df["ema200_slope"] = df["ema_200"].pct_change(5)

    # Bandas de Bollinger (ancho relativo)
    bb = ta.bbands(df["close"], length=20)
    if bb is not None and not bb.empty:
        up = _safe_get(bb, [c for c in bb.columns if "BBU_" in c])
        lo = _safe_get(bb, [c for c in bb.columns if "BBL_" in c])
        if up is not None and lo is not None:
            df["bb_width"] = (up - lo) / df["close"]

    # Rango relativo
    df["range_frac"] = (df["high"] - df["low"]) / df["close"]

    # OBV z‑score
    obv = ta.obv(df["close"], df["volume"]) if "volume" in df else None
    if obv is not None:
        df["obv_z"] = (obv - obv.rolling(50).mean()) / obv.rolling(50).std()

    # ATR normalizado 4h
    if p.get("USE_ATR_NORM", True):
        win = int(p.get("ATR_NORM_WIN", 14))
        df["atr_norm"] = df["atr"] / df["atr"].rolling(win).mean()

    # ADX diario previo (D1)
    if p.get("USE_ADX_D1", True):
        daily = df4h[["open", "high", "low", "close", "volume"]].resample("1D").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        adx_d = ta.adx(daily["high"], daily["low"], daily["close"], length=14)
        if adx_d is not None and not adx_d.empty:
            col_adx_d = _safe_get(adx_d, [c for c in adx_d.columns if "ADX_" in c or "adx" in c.lower()])
            series = col_adx_d.shift(1) if col_adx_d is not None else pd.Series(index=daily.index, dtype=float)
        else:
            series = pd.Series(index=daily.index, dtype=float)
        series.index = series.index.normalize()
        df["date"] = df.index.normalize()
        df["adx_d1"] = series.reindex(df["date"]).values
        df.drop(columns=["date"], inplace=True)

    # Hora (cíclica)
    if p.get("USE_CYCLICAL_HOUR", True):
        h = df.index.hour
        df["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * h / 24.0)

    df.dropna(inplace=True)
    return df


# -----------------------------
# Target: triple‑barrier (multiclase)
# -----------------------------

def label_triple_barrier(df: pd.DataFrame, H: int, tp_mult: float, sl_mult: float) -> pd.Series:
    close = df["close"].values
    atr = df["atr"].values
    up = close * (1 + tp_mult * (atr / close))
    dn = close * (1 - sl_mult * (atr / close))

    lbl = np.zeros(len(df), dtype=int)
    high = df["high"].values
    low = df["low"].values

    for i in range(len(df) - 1):
        end = min(len(df), i + 1 + H)
        hit_up = np.where(high[i+1:end] >= up[i])[0]
        hit_dn = np.where(low[i+1:end] <= dn[i])[0]
        t_up = (hit_up[0] if hit_up.size else np.inf)
        t_dn = (hit_dn[0] if hit_dn.size else np.inf)
        if t_up < t_dn:
            lbl[i] = 1
        elif t_dn < t_up:
            lbl[i] = -1
        else:
            lbl[i] = 0
    lbl[-H:] = 0
    return pd.Series(lbl, index=df.index)


# -----------------------------
# Modelo (bundle con metadatos)
# -----------------------------

def _save_bundle(model: RandomForestClassifier, features: List[str]):
    joblib.dump({"version": CODE_VERSION, "features": list(features), "model": model}, MODEL_BUNDLE_PATH)


def _load_bundle() -> Optional[dict]:
    if MODEL_BUNDLE_PATH.exists():
        try:
            return joblib.load(MODEL_BUNDLE_PATH)
        except Exception:
            return None
    return None


def _features_match(bundle: dict, features_now: List[str]) -> bool:
    try:
        return list(bundle.get("features", [])) == list(features_now)
    except Exception:
        return False


def train_or_load(train_df: pd.DataFrame, features: List[str], force_retrain: bool=False) -> tuple[RandomForestClassifier, List[str]]:
    X = train_df[features]
    y = train_df["target"].astype(int)

    if not force_retrain:
        bundle = _load_bundle()
        if bundle and _features_match(bundle, features):
            return bundle["model"], bundle["features"]

    model = RandomForestClassifier(
        n_estimators=400, max_depth=12, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
    )
    model.fit(X, y)
    _save_bundle(model, features)
    return model, features


# -----------------------------
# Filtros de contexto
# -----------------------------

def _build_allow_trade(df: pd.DataFrame, p: dict) -> pd.Series:
    conds = []
    if p.get("USE_ADX_D1", True) and "adx_d1" in df.columns:
        conds.append(df["adx_d1"] >= float(p.get("ADX_D1_MIN", 20)))
    if p.get("USE_ATR_NORM", True) and "atr_norm" in df.columns:
        a_min = float(p.get("ATR_NORM_MIN", 0.8))
        a_max = float(p.get("ATR_NORM_MAX", 1.8))
        conds.append(df["atr_norm"].between(a_min, a_max))
    allow_hours = p.get("ALLOW_HOURS")
    if allow_hours is not None:
        # soporta lista de ints
        allow_set = set(int(h) for h in allow_hours)
        conds.append(pd.Series(df.index.hour).isin(allow_set).set_axis(df.index))
    if not conds:
        return pd.Series(True, index=df.index)
    out = conds[0]
    for c in conds[1:]:
        out = out & c
    return out.fillna(False)


# -----------------------------
# Señales (multiclase) sin lookahead
# -----------------------------

def predict_signals(df_test: pd.DataFrame, model: RandomForestClassifier, features: List[str], p: dict) -> pd.DataFrame:
    X = df_test[features]
    proba = model.predict_proba(X)
    classes = list(model.classes_)

    def p_of(cls):
        if cls in classes:
            return pd.Series(proba[:, classes.index(cls)], index=df_test.index)
        return pd.Series(np.full(len(df_test), 1/3), index=df_test.index)

    p_up = p_of(1)
    p_dn = p_of(-1)

    out = df_test.copy()
    out["p_up"], out["p_dn"] = p_up, p_dn

    thr = float(p.get("THRESHOLD", 0.52))
    gap = float(p.get("MIN_PROB_GAP", 0.02))

    long_raw = (p_up >= thr) & ((p_up - p_dn) >= gap)
    short_raw = (p_dn >= thr) & ((p_dn - p_up) >= gap)

    # Filtros de contexto
    out["allow_trade"] = _build_allow_trade(out, p)

    # Lag=1 enmascarado
    out["long_sig"]  = long_raw.shift(1).fillna(False) & out["allow_trade"].shift(1).fillna(False)
    out["short_sig"] = short_raw.shift(1).fillna(False) & out["allow_trade"].shift(1).fillna(False)

    return out


# -----------------------------
# Backtest bar‑a‑bar
# -----------------------------
@dataclass
class Position:
    direction: int   # 1 long, -1 short
    entry: float
    sl: float
    tp: float
    trail: float
    units: float
    max_fav: float


def backtest_bar_by_bar(df: pd.DataFrame, p: dict) -> dict:
    equity = 100_000.0
    init_eq = equity
    peak = equity

    pos: Position | None = None
    trades: list[float] = []

    eq_curve: list[float] = []
    dd_curve: list[float] = []

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        row = df.iloc[i]

        # Gestión de posición
        if pos is not None:
            if pos.direction == 1:
                pos.max_fav = max(pos.max_fav, row["high"])
                pos.trail = max(pos.trail, pos.max_fav * (1 - p["TRAIL_PCT"]))
                stop_level = min(pos.sl, pos.trail)
                stop_hit = row["low"] <= stop_level
                tp_hit = row["high"] >= pos.tp
                exit_price = None
                if stop_hit and tp_hit:
                    exit_price = stop_level * (1 - p["SLIPPAGE"])  # peor caso
                elif stop_hit:
                    exit_price = stop_level * (1 - p["SLIPPAGE"])  # venta
                elif tp_hit:
                    exit_price = pos.tp * (1 - p["SLIPPAGE"])      # venta
                if exit_price is not None:
                    pnl = (exit_price - pos.entry) * pos.direction * pos.units
                    cost = (pos.entry + exit_price) * pos.units * p["COST"]
                    equity += pnl - cost
                    trades.append(pnl - cost)
                    pos = None
            else:
                pos.max_fav = min(pos.max_fav, row["low"])
                pos.trail = min(pos.trail, pos.max_fav * (1 + p["TRAIL_PCT"]))
                stop_level = max(pos.sl, pos.trail)
                stop_hit = row["high"] >= stop_level
                tp_hit = row["low"] <= pos.tp
                exit_price = None
                if stop_hit and tp_hit:
                    exit_price = stop_level * (1 + p["SLIPPAGE"])  # peor caso
                elif stop_hit:
                    exit_price = stop_level * (1 + p["SLIPPAGE"])  # compra
                elif tp_hit:
                    exit_price = pos.tp * (1 + p["SLIPPAGE"])      # compra
                if exit_price is not None:
                    pnl = (exit_price - pos.entry) * pos.direction * pos.units
                    cost = (pos.entry + exit_price) * pos.units * p["COST"]
                    equity += pnl - cost
                    trades.append(pnl - cost)
                    pos = None

        # Entradas (lag=1)
        if pos is None:
            if prev["long_sig"]:
                entry = row["open"] * (1 + p["SLIPPAGE"])  # compra
                sl = entry * (1 - p["SL_PCT"]) ; tp = entry * (1 + p["TP_PCT"]) ; trail = entry * (1 - p["TRAIL_PCT"])
                risk_per_unit = max(entry - sl, 1e-12)
                units = (equity * p["RISK_PERC"]) / risk_per_unit
                pos = Position(1, entry, sl, tp, trail, units, max_fav=row["high"])
            elif prev["short_sig"]:
                entry = row["open"] * (1 - p["SLIPPAGE"])  # venta corta
                sl = entry * (1 + p["SL_PCT"]) ; tp = entry * (1 - p["TP_PCT"]) ; trail = entry * (1 + p["TRAIL_PCT"])
                risk_per_unit = max(sl - entry, 1e-12)
                units = (equity * p["RISK_PERC"]) / risk_per_unit
                pos = Position(-1, entry, sl, tp, trail, units, max_fav=row["low"])

        peak = max(peak, equity)
        dd_curve.append((equity - peak) / peak if peak > 0 else 0.0)
        eq_curve.append(equity)

    pnl_arr = np.array(trades)
    gross_profit = pnl_arr[pnl_arr > 0].sum() if len(pnl_arr) else 0.0
    gross_loss = -pnl_arr[pnl_arr < 0].sum() if len(pnl_arr) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

    return {
        "equity_curve": np.array(eq_curve),
        "drawdown_curve": np.array(dd_curve),
        "net_profit": equity - init_eq,
        "win_rate": (pnl_arr > 0).mean() * 100 if len(pnl_arr) else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown": float(min(dd_curve)) if dd_curve else 0.0,
        "trades": int(len(pnl_arr)),
    }


# -----------------------------
# Gráficos
# -----------------------------

def plot_equity_drawdown(dates: pd.Index, equity: np.ndarray, drawdown: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
    ax1.plot(dates[1:1+len(equity)], equity, label="Equity ($)")
    ax1.set_title("Curva de Equity (Out-of-sample)")
    ax1.legend()

    ax2.fill_between(dates[1:1+len(drawdown)], drawdown * 100, 0, alpha=0.3)
    ax2.set_title("Drawdown (%)")
    ax2.set_ylim(min(drawdown) * 100 * 1.1, 0)

    plt.tight_layout()
    plt.show()


# -----------------------------
# CLI principal
# -----------------------------

def _parse_hours_list(arg: Optional[str]) -> Optional[List[int]]:
    if not arg:
        return None
    try:
        return [int(x) for x in str(arg).split(',') if str(x).strip() != '']
    except Exception:
        raise SystemExit("--allow-hours debe ser una lista separada por comas, p. ej. 4,8,12,16,20")


def main():
    ap = argparse.ArgumentParser(description="Swing 4h ML RF – Paso 4.3: más trades (threshold/gap)")
    ap.add_argument("--symbol", default="BTC-USD", help="Símbolo yfinance (ej: BTC-USD)")
    ap.add_argument("--csv", help="Ruta CSV local respaldo (opcional)")
    ap.add_argument("--split", type=float, default=0.7, help="Proporción train/test (0-1)")
    ap.add_argument("--split-date", help="Fecha corte YYYY-MM-DD para split temporal")
    ap.add_argument("--retrain", action="store_true", help="Fuerza reentrenar el modelo")
    # overrides opcionales
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--min-prob-gap", type=float)
    ap.add_argument("--adx-d1-min", type=float)
    ap.add_argument("--atr-norm-min", type=float)
    ap.add_argument("--atr-norm-max", type=float)
    ap.add_argument("--allow-hours", type=str)
    args = ap.parse_args()

    # Params
    p = DEFAULT_PARAMS.copy()
    if args.threshold is not None:      p["THRESHOLD"] = float(args.threshold)
    if args.min_prob_gap is not None:   p["MIN_PROB_GAP"] = float(args.min_prob_gap)
    if args.adx_d1_min is not None:     p["ADX_D1_MIN"] = float(args.adx_d1_min)
    if args.atr_norm_min is not None:   p["ATR_NORM_MIN"] = float(args.atr_norm_min)
    if args.atr_norm_max is not None:   p["ATR_NORM_MAX"] = float(args.atr_norm_max)
    ah_list = _parse_hours_list(args.allow_hours)
    if ah_list is not None:
        p["ALLOW_HOURS"] = ah_list

    # 1) Datos + features + contexto
    raw = load_data(args.symbol if not args.csv else None, args.csv)
    feats = add_features(raw, p)

    # 2) Target (triple‑barrier)
    feats["target"] = label_triple_barrier(feats, int(p["H_BARS"]), float(p["TP_ATR_MULT"]), float(p["SL_ATR_MULT"]))

    # 3) Split temporal
    if args.split_date:
        cutoff = pd.to_datetime(args.split_date, utc=True)
        train = feats[feats.index < cutoff]
        test = feats[feats.index >= cutoff]
    else:
        idx = int(len(feats) * float(args.split))
        train, test = feats.iloc[:idx], feats.iloc[idx:]

    if len(train) < 400 or len(test) < 150:
        raise SystemExit("Split produce segmentos demasiado pequeños; ajusta --split o --split-date")

    # 4) Selección final de features (descarta columnas no calculadas o constantes)
    all_feats = BASE_FEATURES + (CYCLICAL_FEATURES if p.get("USE_CYCLICAL_HOUR", True) else [])
    cols = [c for c in all_feats if c in feats.columns and feats[c].std() > 0]

    # 5) Entrenar/cargar
    model, feats_used = train_or_load(train.assign(target=train["target"]), cols, force_retrain=args.retrain)

    # 6) Predicciones OOS + señales (lag=1) + filtros
    test_sig = predict_signals(test, model, feats_used, p)

    # 7) Backtest bar‑a‑bar
    res = backtest_bar_by_bar(test_sig, p)

    # 8) Resultados
    print("\n=== RESULTADOS OUT-OF-SAMPLE ===")
    print(f"Trades           : {res['trades']}")
    print(f"Net Profit ($)   : {res['net_profit']:,.2f}")
    print(f"Win Rate (%)     : {res['win_rate']:.2f}")
    print(f"Profit Factor    : {res['profit_factor']:.2f}")
    print(f"Max Drawdown (%) : {res['max_drawdown'] * 100:.2f}")

    # 9) Gráfico
    plot_equity_drawdown(test_sig.index, res["equity_curve"], res["drawdown_curve"])


if __name__ == "__main__":
    main()
