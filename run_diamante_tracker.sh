# === run_diamante_tracker.sh ===
#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${OUTDIR:-reports}"
mkdir -p "$OUTDIR"

python - <<'PY'
import pandas as pd
from pathlib import Path

outdir = Path("reports")

# Archivos que el plan de la semana va generando
candidatos = [
    "diamante_day2_summary.csv",
    "diamante_day3_microgrid_best.csv",
    "diamante_day4_costs_summary.csv",
    "diamante_day5_wf_summary.csv",   # se añadirá cuando exista
]

partes = []
for name in candidatos:
    p = outdir / name
    if p.exists():
        df = pd.read_csv(p)
        df.insert(0, "block", name)
        partes.append(df)

if not partes:
    print("⚠️  No encontré archivos de resumen para trackear.")
else:
    tracker = pd.concat(partes, ignore_index=True)
    out = outdir / "diamante_week1_tracker.csv"
    tracker.to_csv(out, index=False)
    print("[OK] Tracker actualizado →", out)
    # Muestra corta
    print("\nVista rápida:")
    with pd.option_context('display.width', 140, 'display.max_columns', None):
        print(tracker.tail(10).to_string(index=False))
PY