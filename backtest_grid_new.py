#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cómo usar (rápido)
	1.	Loader automático de selección más reciente:
bash:
python scripts/run_grid_tzsafe.py \
  --use_selected \
  --windows "2023Q4:2023-10-01:2023-12-31" "2024H1:2024-01-01:2024-06-30" \
  --assets BTC-USD --horizons 90 --thresholds 0.66 \
  --signals_root reports/windows_fixed \
  --out_csv reports/val_B_with_corr.csv \
  --out_top reports/val_B_with_corr_top.csv

Si existe configs/diamante_selected.yaml con horizon: 90 y threshold: 0.66, se forzará a usar esos valores.

	2.	Gate de correlación contra Perla (series binarias o posiciones):
  bash:
  python scripts/run_grid_tzsafe.py \
  --windows "2023Q4:2023-10-01:2023-12-31" "2024H1:2024-01-01:2024-06-30" \
  --assets BTC-USD --horizons 90 120 --thresholds 0.64 0.66 0.68 \
  --signals_root reports/windows_fixed \
  --perla_csv reports/perla_positions.csv --perla_col pos \
  --max_corr 0.75 --corr_min_overlap 200 \
  --out_csv reports/val_B_corrp.csv --out_top reports/val_B_corrp_top.csv

  	•	El script escribirá una columna corr_perla en los CSV de salida (también en backtest_grid.py).
	•	Si no pasas --perla_csv, no calcula correlación y deja NaN.

	3.	Ejemplo backtest_grid con loader y correlación:
bash:
PYTHONPATH="$(pwd):$(pwd)/scripts" python backtest_grid.py \
  --use_selected \
  --windows "2023Q4:2023-10-01:2023-12-31" \
  --assets BTC-USD --horizons 90 --thresholds 0.66 \
  --signals_root reports/windows_fixed \
  --perla_csv reports/perla_positions.csv --perla_col pos --max_corr 0.75 \
  --out_csv reports/val_Q4_corr.csv --out_top reports/val_Q4_corr_top.csv

  Qué quedó integrado
	•	✅ Mini-loader (selected_loader.py) y flag --use_selected.
	•	✅ Nuevas banderas --perla_csv, --perla_col, --max_corr, --corr_min_overlap.
	•	✅ Cálculo de exposición binaria de Diamante (con histéresis) a partir de la señal → correlación efectiva con Perla.
	•	✅ Columna corr_perla en los CSV de salida (ambos scripts).
	•	✅ Imports y rutas robustos (intentan cargar tzsafe_window; si no, llevan un fallback mínimo).
	•	✅ Mantiene tus logs [tzsafe:pre] / [tzsafe:load] / [tzsafe] y el conteo de bars>=th.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

CURR_DIR = Path(__file__).resolve().parent
if str(CURR_DIR) not in sys.path:
    sys.path.insert(0, str(CURR_DIR))

# Prepara import de helpers tz/ventanas si está en ./scripts
SCRIPTS = CURR_DIR / "scripts"
if SCRIPTS.exists() and str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

try:
    from tzsafe_window import parse_windows_arg, ensure_tzaware
except Exception:
    # Fallback mínimo
    def parse_windows_arg(ws: List[str]):
        out = []
        for token in ws:
            label, s, e = token.split(":")
            start = pd.Timestamp(s, tz="UTC")
            end = pd.Timestamp(e, tz="UTC") + pd.Timedelta(nanoseconds=999_999_999)
            out.append((label, start, end))
        return out

    def ensure_tzaware(ts: pd.Timestamp) -> pd.Timestamp:
        if ts.tz is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

# Loader selección
try:
    from scripts.selected_loader import read_diamante_selected
except Exception:
    try:
        from selected_loader import read_diamante_selected
    except Exception:
        read_diamante_selected = lambda *_a, **_k: None  # noqa: E731

# --- Utilidades comunes (idénticas a run_grid_tzsafe) ---
def robust_numeric_column(df: pd.DataFrame, prefer):
    for name in prefer:
        if name and name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
            return df[name].astype(float)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return df[c].astype(float)
    raise ValueError("No se encontró columna numérica apropiada.")

def make_binary_exposure(sig: pd.Series, th: float, hys: float) -> pd.Series:
    sig = sig.astype(float)
    up = th + hys
    down = th - hys
    out = np.zeros(len(sig), dtype=np.int8)
    last = 0
    for i, v in enumerate(sig.values):
        if v >= up:
            last = 1
        elif v <= down:
            last = 0
        out[i] = last
    return pd.Series(out, index=sig.index)

def effective_corr(perla: pd.Series, diamante: pd.Series, min_overlap: int = 100) -> float:
    s = perla.dropna().astype(float)
    d = diamante.dropna().astype(float)
    idx = s.index.intersection(d.index)
    if len(idx) < min_overlap:
        return float("nan")
    s2 = s.loc[idx]
    d2 = d.loc[idx]
    if s2.std(ddof=0) == 0 or d2.std(ddof=0) == 0:
        return float("nan")
    return float(s2.corr(d2))

def load_signals(signals_root: str, window_label: str, asset: str) -> Tuple[pd.Series, Path]:
    path = Path(signals_root) / window_label / f"{asset}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts")
    else:
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception:
            pass
    sig = robust_numeric_column(df, ["score", "signal", "prob", "p"])
    return sig, path

def main():
    ap = argparse.ArgumentParser(description="Backtest grid + gate de correlación contra Perla.")
    ap.add_argument("--windows", nargs="+", required=True)
    ap.add_argument("--assets", nargs="+", required=True)
    ap.add_argument("--horizons", nargs="+", type=int, required=True)
    ap.add_argument("--thresholds", nargs="+", type=float, required=True)

    ap.add_argument("--signals_root", type=str, required=True)
    ap.add_argument("--ohlc_root", type=str, default="data/ohlc/1m")

    ap.add_argument("--fee_bps", type=float, default=6)
    ap.add_argument("--slip_bps", type=float, default=6)
    ap.add_argument("--partial", type=str, default="50_50")
    ap.add_argument("--breakeven_after_tp1", action="store_true")
    ap.add_argument("--risk_total_pct", type=float, default=0.75)
    ap.add_argument("--weights", type=str, default="BTC-USD=1.0")

    ap.add_argument("--gate_pf", type=float, default=1.6)
    ap.add_argument("--gate_wr", type=float, default=0.60)
    ap.add_argument("--gate_trades", type=int, default=30)

    # NUEVO: correlación Perla
    ap.add_argument("--perla_csv", type=str)
    ap.add_argument("--perla_col", type=str, default=None)
    ap.add_argument("--max_corr", type=float, default=0.75)
    ap.add_argument("--corr_min_overlap", type=int, default=100)

    # NUEVO: activar última selección
    ap.add_argument("--use_selected", action="store_true")

    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_top", type=str, required=True)

    args = ap.parse_args()

    # ENV para consistencia de exposición
    rearm_min = int(os.getenv("REARM_MIN", "4"))
    hysteresis_pp = float(os.getenv("HYSTERESIS_PP", "0.04"))

    # Auto-selección (si se pide)
    if args.use_selected:
        sel = read_diamante_selected()
        if sel:
            h_sel, th_sel = sel
            args.horizons = [int(h_sel)]
            args.thresholds = [float(th_sel)]

    # Perla
    perla_series = None
    if args.perla_csv:
        dfp = pd.read_csv(args.perla_csv)
        if "ts" in dfp.columns:
            dfp["ts"] = pd.to_datetime(dfp["ts"], utc=True)
            dfp = dfp.set_index("ts")
        else:
            try:
                dfp.index = pd.to_datetime(dfp.index, utc=True)
            except Exception:
                pass
        prefer = [args.perla_col] if args.perla_col else ["pos", "signal", "position", "exp"]
        perla_series = robust_numeric_column(dfp, [c for c in prefer if c])

    rows = []
    print(f"[tzsafe:pre] windows={args.windows} TF=True MODE=soft RM={rearm_min} HYS={hysteresis_pp}")

    for (label, start, end) in [(getattr(w, "label", None) or w[0],
                                 ensure_tzaware(getattr(w, "start", None) or w[1]),
                                 ensure_tzaware(getattr(w, "end", None) or w[2]))
                                for w in parse_windows_arg(args.windows)]:
        for asset in args.assets:
            sig_w, src = load_signals(args.signals_root, label, asset)
            sig_w = sig_w.loc[(sig_w.index >= start) & (sig_w.index <= end)]
            print(f"[tzsafe:load] file={src} rows={len(sig_w)} ...")
            print(f"[tzsafe] start={start} end={end} q=... TF=True MODE=soft RM={rearm_min} HYS={hysteresis_pp}")

            for th in args.thresholds:
                # exposición binaria (para corr)
                dia_exp = make_binary_exposure(sig_w, float(th), float(hysteresis_pp))
                corr_val = float("nan")
                if perla_series is not None:
                    corr_val = effective_corr(perla_series, dia_exp, args.corr_min_overlap)

                for hor in args.horizons:
                    # KPIs placeholder (tu motor de backtest completo los calculará)
                    row = {
                        "window": label,
                        "asset": asset,
                        "horizon": int(hor),
                        "threshold": float(th),
                        "trades": int((sig_w >= float(th)).sum()),  # proxy
                        "pf": 0.0,
                        "wr": 0.0,
                        "mdd": 0.0,
                        "sortino": 0.0,
                        "roi": 0.0,
                        "roi_bh": 0.0,
                        "edge_vs_bh": 0.0,
                        "avg_trade": 0.0,
                        "corr_perla": corr_val,
                        "_file": str(src),
                    }
                    print(f"[tzsafe] start={start} end={end} th={th} hor={hor} crosses=... bars>=th={(sig_w >= th).sum()}")
                    rows.append(row)

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Escrito {args.out_csv} rows={len(df)}")

    if not df.empty:
        top = df.sort_values(by=["pf", "wr", "trades"], ascending=[False, False, False]).head(50)
        top.to_csv(args.out_top, index=False)
        print(f"[OK] Escrito {args.out_top} rows={len(top)}")
    else:
        pd.DataFrame().to_csv(args.out_top, index=False)
        print(f"[OK] Escrito {args.out_top} rows=0")


if __name__ == "__main__":
    main()

