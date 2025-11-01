#!/usr/bin/env zsh
set -euo pipefail

ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
VENV="$ROOT/.venv"

# Preset CORE (TOP Santo Grial)
PRESET_LINK="$ROOT/configs/mini_accum/presets/PROD_TOP.yaml"
PRESET_SRC="$ROOT/configs/mini_accum/presets/CORE_2025.yaml"

# Rango OOS H1-2025
START="2025-01-01"
END="2025-06-30"
SUFFIX="OOS_2025H1_core_check_from_source"

# 0) Entorno
if [[ -f "$VENV/bin/activate" ]]; then
  . "$VENV/bin/activate"
fi

# 1) Asegura alias de preset TOP → CORE_2025
if [[ ! -L "$PRESET_LINK" ]]; then
  ln -sfn "$(basename "$PRESET_SRC")" "$PRESET_LINK"
fi

# 2) (Info) Hash de datasets candidatos (por auditoría)
for CAND in "$ROOT/data/ohlc/4h/BTC-USD.csv" "$ROOT/data/ohlc/1d/BTC-USD.csv"; do
  if [[ -s "$CAND" ]]; then
    h=$(shasum -a 256 "$CAND" | awk '{print $1}')
    echo "[DATA HASH] $(realpath "$CAND") -> $h"
  fi
done

# 3) Ejecuta el motor (Cocinero) con CORE_2025 en 2025H1
python -m mini_accum.cli \
  --config "$PRESET_LINK" \
  --start "$START" --end "$END" \
  --suffix "$SUFFIX"

# 4) Localiza KPI recién generado
KPI_CSV="$(ls -1t "$ROOT"/reports/mini_accum/*_kpis__${SUFFIX}.csv 2>/dev/null | head -n1 || true)"
if [[ -z "${KPI_CSV:-}" || ! -s "$KPI_CSV" ]]; then
  KPI_CSV="$(find "$ROOT/reports/mini_accum" -type f -name "*_kpis__${SUFFIX}.csv" -print -quit 2>/dev/null || true)"
fi
[[ -n "${KPI_CSV:-}" && -s "$KPI_CSV" ]] || { echo "[ERR] No salió KPI CSV para ${SUFFIX}"; exit 2; }
echo "[KPI] $KPI_CSV"

# 5) Extrae sats_mult y mdd_vs_hodl de forma robusta
read SATS MDDRATIO FLIPS <<EOF
$(python - "$KPI_CSV" << 'PY'
import sys, csv
kpi_csv = sys.argv[1]
row={}
with open(kpi_csv, newline='') as f:
    r = csv.DictReader(f)
    row = next(r, {})

# sats_mult (aliases)
sats = ""
for key in ("sats_mult","net_btc_ratio","netBTC","btc_mult","mult","sats"):
    if key in row and row[key] and row[key].strip():
        sats = row[key].strip()
        break

# mdd_vs_hodl (aliases + reconstrucción mm/mh)
mdd = (row.get("mdd_vs_hodl") or row.get("mdd_vs_hodl_ratio") or "").strip()
if not mdd:
    mm = (row.get("mdd_model") or row.get("mdd_model_usd") or
          row.get("mdd_model_btc") or row.get("mdd_model_pct"))
    mh = (row.get("mdd_hodl")  or row.get("mdd_hodl_usd")  or
          row.get("mdd_hodl_btc")  or row.get("mdd_hodl_pct"))
    try:
        if mm and mh and float(mh)!=0:
            mdd = str(float(mm)/float(mh))
    except Exception:
        mdd = ""

# flips (aliases)
flips = (row.get("flips") or row.get("flips_total") or row.get("flips_per_year") or "").strip()

print((sats or "NA"), (mdd or "n/d"), (flips or "n/a"))
PY
)
EOF

if [[ "$SATS" == "NA" || -z "$SATS" ]]; then
  echo "[FAIL] sats_mult no encontrado (KPI: $KPI_CSV)"
  exit 3
fi

echo "[RESULT] 2025H1 CORE_2025 → sats_mult=$SATS | mdd_vs_hodl=$MDDRATIO | flips=$FLIPS"
