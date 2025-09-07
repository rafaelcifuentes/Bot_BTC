#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corazón weights generator (v0.2)

Consumes: heart_rules.yaml v0.2 + inputs (OHLC 4h, signals diamante/perla, optional clusters)
Produces:  corazon/weights.csv and corazon/lq.csv (UTC, 4h)

Author: ChatGPT (GPT-5 Thinking)

Usage example:
  python corazon_weights_generator.py \
    --rules corazon/heart_rules.yaml \
    --ohlc data/ohlc/4h/BTC-USD.csv \
    --diamante signals/diamante.csv \
    --perla signals/perla.csv \
    --clusters corazon/clusters.csv \
    --out_weights corazon/weights.csv \
    --out_lq corazon/lq.csv

Notes:
- Expects UTC timestamps. Will attempt to localize to UTC if naive.
- signals/*.csv columns expected:
    diamante.csv: timestamp,sD,w_diamante_raw,retD_btc
    perla.csv:    timestamp,sP,w_perla_raw,retP_btc
- clusters.csv (optional) columns (percent distances & scores):
    timestamp,lq_up_dist_pct,lq_dn_dist_pct,lq_up_score,lq_dn_score
- Corr-gate requires retD_btc/retP_btc present. If missing → corr-gate skipped.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# ---------------------- Utils ----------------------

def _ensure_utc_index(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    if ts_col in df.columns:
        idx = pd.to_datetime(df[ts_col], utc=True)
        df = df.drop(columns=[ts_col])
        df.index = idx
    elif df.index.name != ts_col:
        df.index = pd.to_datetime(df.index, utc=True)
    # sort & drop dups
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

def _resample_4h(df: pd.DataFrame, how: str = "ffill") -> pd.DataFrame:
    # For OHLC we'll resample separately; for signals simple ffill is enough
    if how == "ffill":
        return df.resample("4H").last().ffill()
    return df.resample("4H").last()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def true_range(h, l, c_prev):
    return np.maximum.reduce([h - l, np.abs(h - c_prev), np.abs(l - c_prev)])

def atr_wilder(df: pd.DataFrame, length: int = 14) -> pd.Series:
    # Assumes df has columns High, Low, Close
    h, l, c = df["High"], df["Low"], df["Close"]
    c_prev = c.shift(1)
    tr = true_range(h, l, c_prev)
    atr = pd.Series(tr, index=df.index).ewm(alpha=1/length, adjust=False).mean()
    return atr

def adx_wilder(df: pd.DataFrame, length: int = 14) -> pd.Series:
    # Classic Wilder's ADX (approx). df columns: High, Low, Close
    H, L, C = df["High"], df["Low"], df["Close"]
    up_move = H.diff()
    down_move = -L.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(H, L, C.shift(1))
    atr = pd.Series(tr, index=df.index).ewm(alpha=1/length, adjust=False).mean()

    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean() / atr.replace(0, np.nan)
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (np.abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan))
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    adx.index = df.index
    return adx

def pct_rank(series: pd.Series, window: int) -> pd.Series:
    def _rank_last(x: pd.Series) -> float:
        r = x.rank(pct=True)
        return r.iloc[-1]
    return series.rolling(window).apply(lambda x: _rank_last(pd.Series(x)), raw=False)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ---------------------- LQ logic ----------------------

class LQHysteresis:
    def __init__(self, hysteresis_steps: int = 2):
        self.hyst = hysteresis_steps
        self.state = "NORMAL"
        self.counter = 0

    def update(self, want_state: str) -> str:
        if want_state == self.state:
            self.counter = 0
            return self.state
        # different
        if self.counter + 1 >= self.hyst:
            self.state = want_state
            self.counter = 0
        else:
            self.counter += 1
        return self.state

# ---------------------- Regime State Machine ----------------------

class RegimeSM:
    def __init__(self, dwell_bars: int, max_delta_weight: float, rules: dict):
        self.state = "AMARILLO"
        self.dwell = dwell_bars
        self.min_dwell = dwell_bars
        self.max_delta = max_delta_weight
        self.rules = rules  # dict with thresholds

    def classify(self, adx_val: float, ema_slope: float, atr_pct_val: float) -> str:
        adx_thr = self.rules.get("adx_thr", 20)
        vol_pctl_low = self.rules.get("vol_pctl_low", 0.40)
        slope_thr = self.rules.get("ema50_slope_thr", 0.0)

        if (pd.notna(adx_val) and adx_val >= adx_thr) and (abs(ema_slope) > slope_thr) and (atr_pct_val >= vol_pctl_low):
            desired = "VERDE"
        elif (pd.notna(adx_val) and adx_val < (adx_thr - 2)) and (atr_pct_val < vol_pctl_low):
            desired = "ROJO"
        else:
            desired = "AMARILLO"
        return desired

    def update_state(self, desired: str) -> str:
        if desired != self.state:
            if self.dwell <= 0:
                self.state = desired
                self.dwell = self.min_dwell
            else:
                self.dwell -= 1
        else:
            self.dwell = max(0, self.dwell - 1)
        return self.state

# ---------------------- Core Generator ----------------------

def generate_weights(
    rules_path: Path,
    ohlc_path: Path,
    diamante_path: Path,
    perla_path: Path,
    clusters_path: Path | None,
    out_weights_path: Path,
    out_lq_path: Path,
):
    # Load rules
    rules = yaml.safe_load(Path(rules_path).read_text(encoding="utf-8"))
    tz = rules.get("meta", {}).get("timezone", "UTC")

    # Regime config
    reg_cfg = rules.get("regime", {})
    dwell_bars = reg_cfg.get("dwell_bars", 6)
    max_delta = reg_cfg.get("max_delta_weight", 0.20)
    sm = RegimeSM(dwell_bars, max_delta, {
        "adx_thr": reg_cfg.get("adx_thr", reg_cfg.get("adx1d_min", 20)),
        "ema50_slope_thr": reg_cfg.get("ema50_slope_thr", 0.0),
        "vol_pctl_low": reg_cfg.get("vol_pctl_low", 0.40),
    })

    # State for LQ
    lq_cfg = rules.get("lq", {})
    lq_enabled = bool(lq_cfg.get("enable", True))
    lq_mult_soft = float(lq_cfg.get("mult_soft_veto", 0.70))
    lq_hyst_steps = int(lq_cfg.get("hysteresis_steps", 2))
    lq_hyst = LQHysteresis(lq_hyst_steps)
    dist_close_atr = float(lq_cfg.get("dist_close_atr", 0.6))
    min_cluster_score = float(lq_cfg.get("min_cluster_score", 0.7))

    # Corr gate
    cg = rules.get("corr_gate", {})
    corr_lb = int(cg.get("lookback_bars", 60))
    corr_thr = float(cg.get("threshold", 0.35))
    corr_max_pen = float(cg.get("max_penalty", 0.30))
    perf_days = int(cg.get("perf_window_days", 30))

    # Weight maps
    ws = rules.get("weights_states", {
        "verde": {"diamante": 0.80, "perla": 0.20},
        "amarillo": {"diamante": 0.50, "perla": 0.50},
        "rojo": {"diamante": 0.20, "perla": 0.80},
    })

    # Read OHLC (expects 4h candles)
    ohlc = pd.read_csv(ohlc_path)
    ohlc = _ensure_utc_index(ohlc)
    # normalize columns
    def _get_any(df, colnames):
        for name in colnames:
            for c in df.columns:
                if c.lower() == name.lower():
                    return c
        return None

    high_c = _get_any(ohlc, ["high", "High", "HIGH"])
    low_c  = _get_any(ohlc, ["low", "Low", "LOW"])
    close_c= _get_any(ohlc, ["close", "Close", "CLOSE"])
    if not all([high_c, low_c, close_c]):
        raise ValueError("OHLC CSV must have High/Low/Close columns")

    ohlc = ohlc.rename(columns={high_c: "High", low_c: "Low", close_c: "Close"})
    ohlc = _resample_4h(ohlc[["High","Low","Close"]], how="ffill")

    # Indicators
    adx = adx_wilder(ohlc, length=int(reg_cfg.get("adx1d_len", 14)))
    atr = atr_wilder(ohlc, length=14)
    atr_pct = pct_rank(atr, window=int(reg_cfg.get("atr_pct_window", 100))).clip(0,1)
    ema50 = ema(ohlc["Close"], span=int(reg_cfg.get("ema_fast", 50)))
    ema50_slope = ema50.diff() / ohlc["Close"].replace(0,np.nan)

    # Read signals
    d = _ensure_utc_index(pd.read_csv(diamante_path))
    p = _ensure_utc_index(pd.read_csv(perla_path))

    # Align to 4H
    d = _resample_4h(d)
    p = _resample_4h(p)

    # Optional clusters
    clusters = None
    if clusters_path and Path(clusters_path).exists():
        clusters = _ensure_utc_index(pd.read_csv(clusters_path))
        clusters = _resample_4h(clusters)

    # Prepare unified frame
    df = pd.concat([ohlc[["Close"]], adx.rename("ADX"), atr.rename("ATR"), atr_pct.rename("ATR_pct"),
                    ema50.rename("EMA50"), ema50_slope.rename("EMA50_slope"),
                    d.add_prefix("D_"), p.add_prefix("P_")], axis=1).dropna()

    # Output containers
    rows_w = []
    rows_lq = []

    # Rolling performance for corr gate
    have_rets = ("D_retD_btc" in df.columns) and ("P_retP_btc" in df.columns)
    perf_window_days = perf_days

    # Iterate chronologically
    last_wD, last_wP = 0.0, 0.0
    for ts, row in df.iterrows():
        # 1) Regime classification
        state_desired = sm.classify(row["ADX"], row["EMA50_slope"], row["ATR_pct"])
        state = sm.update_state(state_desired)

        st_key = state.lower()
        base_wD = ws.get(st_key, ws["amarillo"])["diamante"]
        base_wP = ws.get(st_key, ws["amarillo"])["perla"]

        # 2) LQ soft-veto (optional)
        lq_flag = "NORMAL"
        if lq_enabled and clusters is not None and ts in clusters.index:
            cl = clusters.loc[ts]
            sD = float(row.get("D_sD", 0.0))
            sP = float(row.get("P_sP", 0.0))
            bias = np.sign(sD + sP)
            if bias >= 0:  # longs → danger from down cluster
                dist = float(cl.get("lq_dn_dist_pct", np.nan))
                score = float(cl.get("lq_dn_score", np.nan))
            else:         # shorts → danger from up cluster
                dist = float(cl.get("lq_up_dist_pct", np.nan))
                score = float(cl.get("lq_up_score", np.nan))

            if np.isfinite(dist) and np.isfinite(score):
                if (dist <= dist_close_atr) and (score >= min_cluster_score):
                    lq_flag = "HIGH_RISK"

        lq_flag = lq_hyst.update(lq_flag)
        lq_mult = lq_mult_soft if lq_flag == "HIGH_RISK" else 1.0

        wD = base_wD * lq_mult
        wP = base_wP * lq_mult

        # 3) Correlation gate (optional)
        if have_rets and corr_lb > 0:
            # Rolling corr over last corr_lb bars (4h bars)
            # Use integer position for speed
            try:
                pos = df.index.get_loc(ts)
                lb_start = max(0, pos - corr_lb + 1)
                window_idx = df.index[lb_start:pos+1]
                rD = df.loc[window_idx, "D_retD_btc"].dropna()
                rP = df.loc[window_idx, "P_retP_btc"].dropna()
                aligned = pd.concat([rD, rP], axis=1).dropna()
                if len(aligned) > 5:
                    corr = aligned.corr().iloc[0,1]
                    # Performance window (sum of returns) over last perf_window_days
                    start_perf = ts - pd.Timedelta(days=perf_window_days)
                    perfD = df.loc[(df.index>start_perf) & (df.index<=ts), "D_retD_btc"].sum()
                    perfP = df.loc[(df.index>start_perf) & (df.index<=ts), "P_retP_btc"].sum()
                    if pd.notna(corr) and corr > corr_thr:
                        slope = (corr - corr_thr) / (1 - corr_thr)
                        penalty = min(0.5 * slope, corr_max_pen)
                        if perfD < perfP:
                            wD *= (1 - penalty)
                        else:
                            wP *= (1 - penalty)
            except Exception:
                pass

        # 4) Smooth weights per-bar (max delta)
        def smooth(target, last, max_delta):
            return last + np.clip(target - last, -max_delta, +max_delta)

        wD_s = smooth(wD, last_wD, max_delta)
        wP_s = smooth(wP, last_wP, max_delta)

        last_wD, last_wP = wD_s, wP_s

        rows_w.append({"timestamp": ts, "w_diamante": round(float(wD_s), 6),
                       "w_perla": round(float(wP_s), 6)})
        rows_lq.append({"timestamp": ts, "lq_flag": lq_flag})

    weights_df = pd.DataFrame(rows_w).set_index("timestamp")
    lq_df = pd.DataFrame(rows_lq).set_index("timestamp")

    out_weights_path.parent.mkdir(parents=True, exist_ok=True)
    out_lq_path.parent.mkdir(parents=True, exist_ok=True)
    weights_df.to_csv(out_weights_path, date_format="%Y-%m-%d %H:%M:%S%z")
    lq_df.to_csv(out_lq_path, date_format="%Y-%m-%d %H:%M:%S%z")

    print(f"[OK] Pesos escritos en: {out_weights_path}")
    print(f"[OK] LQ flags escritas en: {out_lq_path}")


def main():
    ap = argparse.ArgumentParser(description="Corazón weights generator v0.2")
    ap.add_argument("--rules", type=str, default="corazon/heart_rules.yaml")
    ap.add_argument("--ohlc", type=str, required=True, help="OHLC 4h CSV (UTC) with columns: timestamp,Open,High,Low,Close (case-insensitive)")
    ap.add_argument("--diamante", type=str, required=True, help="signals/diamante.csv")
    ap.add_argument("--perla", type=str, required=True, help="signals/perla.csv")
    ap.add_argument("--clusters", type=str, default=None, help="Optional clusters CSV")
    ap.add_argument("--out_weights", type=str, default="corazon/weights.csv")
    ap.add_argument("--out_lq", type=str, default="corazon/lq.csv")
    args = ap.parse_args()

    generate_weights(
        rules_path=Path(args.rules),
        ohlc_path=Path(args.ohlc),
        diamante_path=Path(args.diamante),
        perla_path=Path(args.perla),
        clusters_path=Path(args.clusters) if args.clusters else None,
        out_weights_path=Path(args.out_weights),
        out_lq_path=Path(args.out_lq),
    )

if __name__ == "__main__":
    main()