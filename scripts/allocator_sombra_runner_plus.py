#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
allocator_sombra_runner_plus.py
- Ejecuta el allocator con tu YAML
- Lee fee_bps/slip_bps del YAML automáticamente
- Calcula Gross por pierna (D/P), costes por pierna, turnover
- Calcula NET por trayectoria (barra a barra) y lo chequea contra eq_overlay.csv
- Reporta también "Gross - Σcostes" (referencia) y Top-5 barras por coste
"""

import argparse
import subprocess
import re
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# ------------------ Helpers ------------------

def _to_utc(idx):
    try:
        return idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    except Exception:
        return pd.to_datetime(idx, utc=True)

def _read_csv_indexed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    df.index = _to_utc(df.index)
    return df

_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
def _as_float(x, default: float) -> float:
    if isinstance(x, (int, float)) and np.isfinite(x):
        return float(x)
    if isinstance(x, str):
        s = x.replace(",", ".")
        m = _num_re.search(s)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                pass
    return float(default)

def load_costs_from_yaml(cfg_path: str) -> tuple[float, float]:
    p = Path(cfg_path)
    if not p.exists():
        print(f"[WARN] YAML no encontrado: {cfg_path}. Uso fee_bps=6, slip_bps=6 por defecto.")
        return 6.0, 6.0
    y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    costs = (y.get("costs") or {})
    fee_bps  = _as_float(costs.get("fee_bps", 6), 6.0)
    slip_bps = _as_float(costs.get("slip_bps", 6), 6.0)
    print(f"[CFG] fees desde YAML → fee_bps={fee_bps}, slip_bps={slip_bps}")
    return float(fee_bps), float(slip_bps)

def _derive_returns_from_any(df: pd.DataFrame, prefer_cols) -> pd.Series:
    # 1) columna de retornos conocida
    for c in prefer_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            if s.std() > 0:
                return s
    # 2) derivar de precio
    for px in ["close","price","px","Close"]:
        if px in df.columns:
            s = pd.to_numeric(df[px], errors="coerce").pct_change().fillna(0.0)
            if s.std() > 0:
                return s
    # 3) fallback ceros
    return pd.Series(0.0, index=df.index)

def _load_returns_series(weights_idx: pd.DatetimeIndex):
    # Diamante
    rD = None
    for path in ["signals/diamante.csv",
                 "reports/allocator/diamante_for_allocator.csv"]:
        p = Path(path)
        if p.exists():
            d = _read_csv_indexed(str(p))
            rD = _derive_returns_from_any(d, ["retD_btc","ret_btc","ret","returns","r"])
            break
    if rD is None:
        rD = pd.Series(0.0, index=weights_idx)

    # Perla
    rP = None
    for path in ["reports/allocator/perla_for_allocator.csv",
                 "signals/perla.csv"]:
        p = Path(path)
        if p.exists():
            d = _read_csv_indexed(str(p))
            rP = _derive_returns_from_any(d, ["retP_btc","ret_btc","ret","returns","r"])
            break
    if rP is None:
        rP = pd.Series(0.0, index=weights_idx)

    rD = rD.reindex(weights_idx).fillna(0.0)
    rP = rP.reindex(weights_idx).fillna(0.0)
    return rD, rP

def _gross_compound(series: pd.Series) -> float:
    return float((1.0 + series.fillna(0.0)).prod() - 1.0)

def _costs_turnover(eD: pd.Series, eP: pd.Series, fee_bps: float, slip_bps: float):
    bps = (float(fee_bps) + float(slip_bps)) / 1e4
    toD = eD.diff().abs().fillna(0.0)
    toP = eP.diff().abs().fillna(0.0)
    per_bar_to = toD + toP
    per_bar_cost = per_bar_to * bps
    costD = float(toD.sum()) * bps
    costP = float(toP.sum()) * bps
    turnover_total = float(per_bar_to.sum())
    return per_bar_cost, costD, costP, turnover_total

def run_allocator(config_path: str):
    cmd = ["python3", "scripts/allocator_sombra_runner.py", "--config", config_path]
    subprocess.run(cmd, check=True)

# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser(description="Runner+ (allocator + breakdown D/P + chequeo NET con costes de YAML)")
    ap.add_argument("--config", type=str, default="configs/allocator_sombra.yaml")
    ap.add_argument("--skip-runner", action="store_true", help="Si ya corriste el allocator, sólo hace el breakdown.")
    args = ap.parse_args()

    fee_bps, slip_bps = load_costs_from_yaml(args.config)

    if not args.skip_runner:
        run_allocator(args.config)

    w_path = "reports/allocator/weights_overlay.csv"
    if not Path(w_path).exists():
        raise FileNotFoundError(f"No encuentro {w_path}. ¿Corrió bien el allocator?")
    w = _read_csv_indexed(w_path)

    for col in ["eD","eP"]:
        if col not in w.columns:
            raise KeyError(f"Falta la columna '{col}' en {w_path}")
    eD = pd.to_numeric(w["eD"], errors="coerce").fillna(0.0)
    eP = pd.to_numeric(w["eP"], errors="coerce").fillna(0.0)

    # Retornos
    rD, rP = _load_returns_series(w.index)

    # Contribuciones
    cD = (eD * rD).fillna(0.0)
    cP = (eP * rP).fillna(0.0)
    gross_series = cD + cP

    # GROSS por pierna y total (sin costes)
    gross_D = _gross_compound(cD)
    gross_P = _gross_compound(cP)
    gross_total = _gross_compound(gross_series)

    # Costes desde turnover y bps del YAML
    per_bar_cost, costD, costP, turnover_total = _costs_turnover(eD, eP, fee_bps, slip_bps)
    cost_total = costD + costP

    # NET correcto por trayectoria (compone barra a barra con costes)
    net_series = gross_series - per_bar_cost
    net_total_traj = _gross_compound(net_series)

    # Métrica "Gross - Σcostes" (referencia, no compuesta)
    net_total_simple = gross_total - cost_total

    # Chequeo contra curva overlay
    eq_path = Path("reports/allocator/curvas_equity/eq_overlay.csv")
    if not eq_path.exists():
        raise FileNotFoundError(f"No encuentro {eq_path}. ¿Corrió bien el allocator?")
    eqo = _read_csv_indexed(str(eq_path))
    if "equity_overlay" not in eqo.columns:
        raise KeyError(f"Falta 'equity_overlay' en {eq_path}")
    net_from_curve = float(eqo["equity_overlay"].iloc[-1] - 1.0)

    diff = float(net_total_traj - net_from_curve)

    # ---------- Reporte ----------
    print(f"NET D ovl: {_gross_compound(cD)} | NET P ovl: {_gross_compound(cP)}")
    print(f"Gross D / P      : {gross_D} / {gross_P}")
    print(f"Gross (sin costes): {gross_total}")

    print(f"Coste D / P      : {costD} / {costP}")
    shareD = (costD / cost_total * 100.0) if cost_total > 0 else 0.0
    shareP = 100.0 - shareD if cost_total > 0 else 0.0
    print(f"Costes totales    : {cost_total}")
    print(f"Cost share D/P    : {shareD:.1f}% / {shareP:.1f}%")

    print(f"Turnover total    : {turnover_total}")

    print(f"Gross - Σcostes   : {net_total_simple}")
    print(f"NET por trayectoria: {net_total_traj}")
    print(f"NET desde curva   : {net_from_curve}")
    print(f"Diff (calc-curve) : {diff}")

    # Top-5 barras por coste
    top = per_bar_cost.sort_values(ascending=False).head(5)
    print("\nTop 5 barras por coste:")
    for ts, c in top.items():
        print(f"{ts} → {c}")

    delta_abs = eD.diff().abs().fillna(0.0) + eP.diff().abs().fillna(0.0)
    for ts in top.index:
        print(f"{ts} Δ|eD|+|eP| = {delta_abs.loc[ts]:.3f}")

if __name__ == "__main__":
    main()