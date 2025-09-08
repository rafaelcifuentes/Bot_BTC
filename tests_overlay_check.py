#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests_overlay_check.py — Reconciliación de PnL overlay con los outputs del allocator.
- Lee fee/slip desde YAML (fallback 6/6 bps)
- Normaliza a UTC y resamplea a 4h como el allocator
- Aplica costes barra-a-barra (compuesto) y compara con eq_overlay.csv
"""
# --- Breakdown útil (pegar al final de tests_overlay_check.py) ---
import numpy as np
import pandas as pd
import yaml
y = yaml.safe_load(open("configs/allocator_sombra.yaml"))
FEE_BPS  = float(y["costs"]["fee_bps"])
SLIP_BPS = float(y["costs"]["slip_bps"])
bps = (FEE_BPS + SLIP_BPS) / 1e-4
from pathlib import Path


# Rutas
CFG_PATH = Path("configs/allocator_sombra.yaml")
W_PATH   = Path("reports/allocator/weights_overlay.csv")
D_PATH   = Path("signals/diamante.csv")
P_PATH   = Path("reports/allocator/perla_for_allocator.csv")
EQO_PATH = Path("reports/allocator/curvas_equity/eq_overlay.csv")

def _to_utc_index(df: pd.DataFrame, ts: str = "timestamp") -> pd.DataFrame:
    """Asegura índice UTC y ordenado por tiempo."""
    if ts in df.columns:
        idx = pd.to_datetime(df[ts], utc=True)
        df = df.drop(columns=[ts])
        df.index = idx
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    # Localiza/convierte a UTC si hace falta
    try:
        df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    except Exception:
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()

def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Mismo resample del allocator: barras 4h, last+ffill."""
    return df.resample("4h").last().ffill()


# 1) Costes desde YAML (fallback a 6/6 bps)
fee_bps, slip_bps = 6.0, 6.0
if CFG_PATH.exists():
    try:
        y = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8")) or {}
        fee_bps  = float(str(y.get("costs", {}).get("fee_bps", fee_bps)).replace(",", "."))
        slip_bps = float(str(y.get("costs", {}).get("slip_bps", slip_bps)).replace(",", "."))
    except Exception:
        pass
bps = (fee_bps + slip_bps) / 1e4

# 2) Cargar CSVs requeridos
for pth in (W_PATH, D_PATH, P_PATH, EQO_PATH):
    if not pth.exists():
        raise FileNotFoundError(f"No encontré {pth}")

w   = _to_utc_index(pd.read_csv(W_PATH,   parse_dates=["timestamp"]))
d   = _to_utc_index(pd.read_csv(D_PATH,   parse_dates=["timestamp"]))
p   = _to_utc_index(pd.read_csv(P_PATH,   parse_dates=["timestamp"]))
eqo = _to_utc_index(pd.read_csv(EQO_PATH, parse_dates=["timestamp"]))

# 3) Resamplear señales a 4h como hace el allocator
d4 = _resample_4h(d)
p4 = _resample_4h(p)

# 4) Alinear todo al timeline de weights (lo ejecutado realmente)
idx = w.index
rD = pd.to_numeric(d4.get("retD_btc"), errors="coerce").reindex(idx).fillna(0.0)
rP = pd.to_numeric(p4.get("retP_btc"), errors="coerce").reindex(idx).fillna(0.0)
eD = pd.to_numeric(w["eD"], errors="coerce").fillna(0.0)
eP = pd.to_numeric(w["eP"], errors="coerce").fillna(0.0)

# 5) PnL bruto y costes
gross_series = eD * rD + eP * rP
to_series    = w[["eD", "eP"]].diff().abs().sum(axis=1).fillna(0.0)
cost_series  = to_series * bps

# 6) Composición correcta: COSTES BARRA-A-BARRA
net_series   = gross_series - cost_series
gross        = float((1.0 + gross_series).prod() - 1.0)
cost_total   = float(cost_series.sum())
net_overlay  = float((1.0 + net_series).prod() - 1.0)
gross_D = float(((1 + eD*rD).prod() - 1))
gross_P = float(((1 + eP*rP).prod() - 1))

# Coste por pata y por barra
turn_D = w["eD"].diff().abs().fillna(0.0)
turn_P = w["eP"].diff().abs().fillna(0.0)
cost_D  = turn_D * bps
cost_P  = turn_P * bps

print("Coste D / P      :", float(cost_D.sum()), "/", float(cost_P.sum()))

# Top 5 barras que más costaron
top_cost = (cost_series.sort_values(ascending=False).head(5))
print("\nTop 5 barras por coste:")
for ts, c in top_cost.items():
    print(ts, "→", float(c))

# Comparación 'gross - sum(costes)' vs trayectoria (didáctico)
gross_minus_sum = (1 + gross_series).prod() - 1 - cost_series.sum()
net_path = (1 + gross_series - cost_series).prod() - 1
print("\nGross - Σcostes  :", float(gross_minus_sum))
print("NET por trayectoria:", float(net_path))
print("Gross D / P      :", gross_D, "/", gross_P)
print("Gross (sin costes):", gross)
print("Costes totales    :", cost_total)
print("NET overlay calc  :", net_overlay)
print("Turnover total    :", float(to_series.sum()))

# 7) Sanity check contra la curva overlay del allocator
eq_overlay = pd.to_numeric(eqo["equity_overlay"], errors="coerce").reindex(idx).ffill()
net_from_curve = float(eq_overlay.iloc[-1] - 1.0)
diff = float(net_overlay - net_from_curve)
print("NET desde curva   :", net_from_curve)
print("Diff (calc-curve) :", diff)
assert abs(diff) < 1e-12, f"Desajuste inesperado vs curva: {diff}"
# Si quieres ser estricto, descomenta:
# assert abs(diff) < 1e-8, f"Desajuste inesperado vs curva: {diff}"
# Cost share por pata
cost_share_D = float(cost_D.sum() / cost_series.sum())
cost_share_P = float(cost_P.sum() / cost_series.sum())
print(f"Cost share D/P    : {cost_share_D:.1%} / {cost_share_P:.1%}")

# ΔE total en top-5 barras de coste (para ver el salto de exposición)
dE_total = w[["eD","eP"]].diff().abs().sum(axis=1).loc[top_cost.index]
for ts, de in dE_total.items():
    print(f"{ts} Δ|eD|+|eP| = {float(de):.3f}")