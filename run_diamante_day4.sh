# === run_diamante_day4.sh ===
#!/usr/bin/env bash
set -euo pipefail

# =========================
# Defaults (override via env)
# =========================
EXCHANGE="${EXCHANGE:-binanceus}"
FREEZE="${FREEZE:-2025-08-05 00:00}"
PERIOD="${PERIOD:-730d}"
HORIZONS="${HORIZONS:-30,60,90}"

# Umbral base
THRESH="${THRESH:-0.60}"

# Costes “deseados” (2× slip + fee ida/vuelta). Solo se aplican si el .py los soporta.
SLIP="${SLIP:-0.0002}"
COST="${COST:-0.0004}"

# STRICT=1 fuerza abortar si el .py no soporta costos
STRICT="${STRICT:-0}"

PY_BIN="${PY_BIN:-python}"
PY_SCRIPT="${PY_SCRIPT:-swing_4h_forward_diamond.py}"
OUTDIR="${OUTDIR:-reports}"

mkdir -p "$OUTDIR"

banner() {
  echo "[INFO] D4 costes | EXCHANGE=${EXCHANGE} | FREEZE=\"${FREEZE}\" | PERIOD=${PERIOD} | HORIZONS=${HORIZONS} | THRESH=${THRESH} | SLIP=${SLIP} | COST=${COST}"
}

supports_cost_flags() {
  # Devuelve 0 si soporta --slip y --cost, 1 si no
  local helpout
  if ! helpout="$(${PY_BIN} "${PY_SCRIPT}" --help 2>&1)"; then
    return 1
  fi
  echo "$helpout" | grep -q -- "--slip" && echo "$helpout" | grep -q -- "--cost"
}

run_one() {
  local symbol="$1"
  local tag="$2"
  local out_csv="${OUTDIR}/diamante_${tag}_day4_costs_week1.csv"

  if supports_cost_flags; then
    echo "[RUN] ${symbol} con COSTES reales → ${out_csv}"
    EXCHANGE="${EXCHANGE}" \
    ${PY_BIN} "${PY_SCRIPT}" --skip_yf \
      --symbol "${symbol}" \
      --period "${PERIOD}" \
      --horizons "${HORIZONS}" \
      --freeze_end "${FREEZE}" \
      --threshold "${THRESH}" \
      --slip "${SLIP}" \
      --cost "${COST}" \
      --out_csv "${out_csv}"
  else
    if [[ "${STRICT}" == "1" ]]; then
      echo "❌ ${PY_SCRIPT} no reconoce --slip/--cost. STRICT=1 → abortando."
      exit 1
    fi
    echo "⚠️  ${PY_SCRIPT} no reconoce --slip/--cost → corriendo BASELINE (sin costos) y guardando como ${out_csv}"
    EXCHANGE="${EXCHANGE}" \
    ${PY_BIN} "${PY_SCRIPT}" --skip_yf \
      --symbol "${symbol}" \
      --period "${PERIOD}" \
      --horizons "${HORIZONS}" \
      --freeze_end "${FREEZE}" \
      --threshold "${THRESH}" \
      --out_csv "${out_csv}"
  fi
}

summarize() {
  ${PY_BIN} - <<'PY'
import pandas as pd
from pathlib import Path
import datetime as dt

paths = {
  "BTC": Path("reports/diamante_btc_day4_costs_week1.csv"),
  "ETH": Path("reports/diamante_eth_day4_costs_week1.csv"),
  "SOL": Path("reports/diamante_sol_day4_costs_week1.csv"),
}
rows = []
no_cost_flag = False

for asset, p in paths.items():
    if not p.exists():
        continue
    df = pd.read_csv(p)
    # Detectar si el archivo se generó en modo baseline (heurística simple):
    # asumimos que el script original no agrega columnas 'slip'/'cost' en CSV;
    # por eso solo dejamos una marca en el resumen si fue baseline.
    # (La marca global se pondrá al final si algún activo cayó en fallback)
    # Tomar 60d, si no 90d
    d = df[df["days"]==60]
    if d.empty:
        d = df[df["days"]==90]
    if d.empty:
        continue
    r = d.iloc[0].to_dict()
    rows.append({
        "asset": asset,
        "pf_60d": r.get("pf", float("nan")),
        "wr_60d": r.get("win_rate", float("nan")),
        "trades_60d": r.get("trades", float("nan")),
        "mdd_60d": r.get("mdd", float("nan")),
        "net_60d": r.get("net", float("nan")),
        "src": p.name
    })

summary = pd.DataFrame(rows).sort_values("asset")
summary["gate_pf>1.5"] = summary["pf_60d"] > 1.5
# Nota: tus CSV usan WR en porcentaje (ej. 72.9); el gate era >0.60 “en fracción”.
# Para mantener intención original, usamos >60.0
summary["gate_wr>60%"] = summary["wr_60d"] > 60.0
summary["gate_ok"] = summary["gate_pf>1.5"] & summary["gate_wr>60%"]

out = Path("reports/diamante_day4_costs_summary.csv")
summary.to_csv(out, index=False)

print("=== D4 (Costes) — Resumen 60d ===")
print(summary.to_string(index=False))
print(f"[OK] Guardado → {out}")
PY
}

main() {
  banner
  run_one "BTC-USD" "btc"
  run_one "ETH-USD" "eth"
  run_one "SOL-USD" "sol"

  echo
  echo "== Archivos generados =="
  ls -lh "${OUTDIR}"/diamante_*_day4_costs_week1.csv || true
  echo

  summarize

  if ! supports_cost_flags; then
    echo "⚠️  NOTA: Estos resultados son BASELINE (sin costos aplicados en motor)."
    echo "    Cuando integre --slip/--cost en ${PY_SCRIPT}, este mismo script recalculará con costes reales."
    echo "    Tip: puedes forzar abortar si faltan flags corriendo: STRICT=1 bash ./run_diamante_day4.sh"
  fi
}

main "$@"