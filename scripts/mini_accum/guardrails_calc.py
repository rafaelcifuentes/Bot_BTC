#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, json, math
from datetime import timedelta
import pandas as pd

def _load_equity(path: str) -> pd.Series:
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else ("ts" if "ts" in df.columns else df.columns[0])
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    eq = None
    for c in ("equity_btc", "equity"):
        if c in df.columns:
            eq = df[c]; break
    if eq is None:
        raise ValueError(f"No encuentro columna equity en {path}")
    s = pd.Series(eq.values, index=ts).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s

def _load_flips(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else ("ts" if "ts" in df.columns else df.columns[0])
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.assign(ts=ts)
    if "executed" not in df.columns:
        if "action" in df.columns:
            df = df.rename(columns={"action": "executed"})
        else:
            raise ValueError(f"No encuentro columna executed/action en {path}")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    return df

def _window_30d_common(a: pd.Series, b: pd.Series, days: int = 30, end_override: str | None = None):
    if a.empty or b.empty:
        return None, None
    end = min(a.index.max(), b.index.max())
    if end_override:
        eo = pd.to_datetime(end_override, utc=True, errors="coerce")
        if eo is not None and not pd.isna(eo):
            if eo.tzinfo is None:
                eo = eo.tz_localize("UTC")
            end = min(end, eo)
    start = end - timedelta(days=days)
    return start, end

def _mdd_mag(s: pd.Series) -> float | None:
    if s.size < 2:
        return None
    dd = (s / s.cummax() - 1.0).min()  # negativo
    return float(-dd)  # magnitud positiva

def _ann_roi(s: pd.Series) -> float | None:
    if s.size < 2:
        return None
    days = (s.index[-1] - s.index[0]).days
    if days <= 0:
        return None
    ratio = float(s.iloc[-1] / s.iloc[0])
    if ratio <= 0:
        return None
    return math.pow(ratio, 365.0 / days) - 1.0

def _fpy(flips_df: pd.DataFrame, start, end, actions: set[str]) -> float:
    if flips_df.empty or start is None or end is None:
        return 0.0
    m = flips_df[(flips_df["ts"] >= start) & (flips_df["ts"] <= end)]
    if m.empty:
        return 0.0
    n = int((m["executed"].fillna("").isin(actions)).sum())
    days = max(1, (end - start).days)
    return n * 365.0 / days

def main():
    ap = argparse.ArgumentParser(description="Guardrails calculator (30d window).")
    ap.add_argument("--base-eq", required=True)
    ap.add_argument("--cand-eq", required=True)
    ap.add_argument("--base-flips", required=True)
    ap.add_argument("--cand-flips", required=True)
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--end", type=str, default=None, help="Override window end date (YYYY-MM-DD, UTC). If not set, uses min(ts_max_base, ts_max_cand).")
    ap.add_argument("--mdd-max-delta", type=float, default=0.0)        # ΔMDD_mag ≤ 0
    ap.add_argument("--fpy-max-delta", type=float, default=2.0)        # ΔFPY ≤ +2/a
    ap.add_argument("--roi-min-delta-annual", type=float, default=-0.04)  # ΔROI_ann ≥ -4%
    ap.add_argument("--json", action="store_true", help="Emit JSON only")
    args = ap.parse_args()

    base = _load_equity(args.base_eq)
    cand = _load_equity(args.cand_eq)
    start, end = _window_30d_common(base, cand, args.window_days, args.end)

    # Métricas equity
    mdd_base = mdd_cand = roi_base = roi_cand = None
    if start is not None:
        b = base[(base.index >= start) & (base.index <= end)]
        c = cand[(cand.index >= start) & (cand.index <= end)]
        mdd_base = _mdd_mag(b)
        mdd_cand = _mdd_mag(c)
        roi_base = _ann_roi(b)
        roi_cand = _ann_roi(c)

    # Flips/FPY (cand cuenta SELL_SLTP)
    base_flips = _load_flips(args.base_flips)
    cand_flips = _load_flips(args.cand_flips)
    fpy_base = _fpy(base_flips, start, end, {"BUY", "SELL"})
    fpy_cand = _fpy(cand_flips, start, end, {"BUY", "SELL", "SELL_SLTP"})

    # Deltas (signo “bueno”: ΔMDD≤0, ΔFPY≤+2, ΔROI≥-4%)
    d_mdd = (mdd_cand - mdd_base) if (mdd_cand is not None and mdd_base is not None) else None
    d_fpy = fpy_cand - fpy_base
    d_roi = (roi_cand - roi_base) if (roi_cand is not None and roi_base is not None) else None

    violations, warns = [], []
    if d_mdd is None: warns.append("ΔMDD=N/A")
    elif d_mdd > args.mdd_max_delta: violations.append("ΔMDD")
    if d_roi is None: warns.append("ΔROI_anual=N/A")
    elif d_roi < args.roi_min_delta_annual: violations.append("ΔROI_anual")
    if d_fpy > args.fpy_max_delta: violations.append("ΔFPY")

    out = {
        "window": {"start": None if start is None else start.isoformat(),
                   "end":   None if end   is None else end.isoformat(),
                   "days":  None if start is None else (end - start).days},
        "metrics": {"mdd_base": mdd_base, "mdd_cand": mdd_cand, "d_mdd": d_mdd,
                    "roi_base": roi_base, "roi_cand": roi_cand, "d_roi": d_roi,
                    "fpy_base": fpy_base, "fpy_cand": fpy_cand, "d_fpy": d_fpy},
        "warns": warns, "violations": violations, "pass": len(violations) == 0
    }

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"[DEBUG] Window={out['window']}")
        print(f"[GUARDRAILS] ΔMDD={out['metrics']['d_mdd']} | ΔFPY={out['metrics']['d_fpy']:.2f}/a | ΔROI_anual={out['metrics']['d_roi']}")
        for w in warns: print(f"[WARN] {w}")
        print("[PASS] Guardrails dentro de umbrales" if out["pass"] else "[VIOLATION] " + ",".join(violations))
    sys.exit(0 if out["pass"] else 2)

if __name__ == "__main__":
    main()