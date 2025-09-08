# === run_diamante_day5.sh ===
#!/usr/bin/env bash
set -euo pipefail

# =========================
# Defaults (override via env)
# =========================
EXCHANGE="${EXCHANGE:-binanceus}"
FREEZE="${FREEZE:-2025-08-05 00:00}"
PERIOD="${PERIOD:-730d}"
HORIZONS="${HORIZONS:-30,60,90}"
THRESH="${THRESH:-0.60}"     # umbral base
PY_BIN="${PY_BIN:-python}"
PY_SCRIPT="${PY_SCRIPT:-swing_4h_forward_diamond.py}"
OUTDIR="${OUTDIR:-reports}"
WF_LIST="${WF_LIST:-3,5}"     # Walk-k a evaluar (coma-sep)

mkdir -p "$OUTDIR"

banner() {
  echo "[INFO] D5 WF | EXCHANGE=${EXCHANGE} | FREEZE=\"${FREEZE}\" | PERIOD=${PERIOD} | HORIZONS=${HORIZONS} | THRESH=${THRESH} | WF_LIST=${WF_LIST}"
}

run_wf_one() {
  local symbol="$1"
  local tag="$2"
  local k="$3"
  local out_csv="${OUTDIR}/diamante_${tag}_wf${k}_week1.csv"

  echo "[RUN] ${symbol} | walk_k=${k} → ${out_csv}"
  EXCHANGE="${EXCHANGE}" \
  ${PY_BIN} "${PY_SCRIPT}" --skip_yf \
    --symbol "${symbol}" \
    --period "${PERIOD}" \
    --horizons "${HORIZONS}" \
    --freeze_end "${FREEZE}" \
    --threshold "${THRESH}" \
    --walk_k "${k}" \
    --out_csv "${out_csv}"
}

summarize_wf() {
  ${PY_BIN} - <<'PY'
import pandas as pd
from pathlib import Path
import re

outdir = Path("reports")
files = sorted(outdir.glob("diamante_*_wf*_week1.csv"))

rows = []
for p in files:
    m = re.match(r"diamante_(\w+)_wf(\d+)_week1\.csv", p.name)
    if not m:
        continue
    asset, k = m.group(1).upper(), int(m.group(2))
    df = pd.read_csv(p)
    d60 = df[df["days"]==60]
    if d60.empty:
        d60 = df[df["days"]==90]
    if d60.empty:
        continue
    r = d60.iloc[0].to_dict()
    rows.append({
        "asset": asset,
        "walk_k": k,
        "pf_60d": r.get("pf"),
        "wr_60d": r.get("win_rate"),
        "trades_60d": r.get("trades"),
        "mdd_60d": r.get("mdd"),
        "net_60d": r.get("net"),
        "src": p.name
    })

wf = pd.DataFrame(rows).sort_values(["asset","walk_k"])
if wf.empty:
    print("⚠️  No se encontraron CSV de D5 WF para resumir.")
else:
    # Gate estabilidad: PF>1.5 y WR>60% (score simple)
    wf["gate_pf>1.5"] = wf["pf_60d"] > 1.5
    wf["gate_wr>60%"] = wf["wr_60d"] > 60.0
    wf["gate_ok"] = wf["gate_pf>1.5"] & wf["gate_wr>60%"]
    # Mejor K por activo (por PF; tie-break por MDD menor)
    best = wf.sort_values(["asset","pf_60d","mdd_60d"], ascending=[True, False, True]) \
             .groupby("asset", as_index=False).head(1)
    wf.to_csv(outdir / "diamante_day5_wf_summary.csv", index=False)
    best.to_csv(outdir / "diamante_day5_wf_best.csv", index=False)
    print("=== D5 WF — resumen 60d ===")
    print(wf.to_string(index=False))
    print("\n=== Mejor por activo (PF) ===")
    print(best.to_string(index=False))
    print("[OK] Guardado → reports/diamante_day5_wf_{summary,best}.csv")

# Opcional: construir tracker semanal si existen previos
track_parts = []
for fname in [
    "diamante_day2_summary.csv",
    "diamante_day3_microgrid_best.csv",
    "diamante_day4_costs_summary.csv",
    "diamante_day5_wf_summary.csv",
]:
    p = outdir / fname
    if p.exists():
        df = pd.read_csv(p)
        df.insert(0, "block", fname)
        track_parts.append(df)

if track_parts:
    tr = pd.concat(track_parts, ignore_index=True)
    tr.to_csv(outdir / "diamante_week1_tracker.csv", index=False)
    print("[OK] Tracker actualizado → reports/diamante_week1_tracker.csv")
else:
    print("ℹ️  Tracker no creado: no se hallaron bloques previos.")
PY
}

main() {
  banner
  IFS=',' read -ra KS <<< "${WF_LIST}"

  for k in "${KS[@]}"; do
    run_wf_one "BTC-USD" "btc" "$k"
    run_wf_one "ETH-USD" "eth" "$k"
    run_wf_one "SOL-USD" "sol" "$k"
  done

  echo
  echo "== Archivos generados (WF) =="
  ls -lh "${OUTDIR}"/diamante_*_wf*_week1.csv || true
  echo

  summarize_wf
}

main "$@"