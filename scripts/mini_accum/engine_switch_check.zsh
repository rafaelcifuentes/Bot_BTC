#!/usr/bin/env zsh
set -euo pipefail

ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
VENV="$ROOT/.venv"
PRESET_LINK="$ROOT/configs/mini_accum/presets/PROD_E1_Y2.yaml"
PRESET_SRC="$ROOT/configs/mini_accum/presets/E1_Y2.yaml"
DATA_1D="$ROOT/data/ohlc/1d/BTC-USD.csv"
SUFFIX="E1Y2_2022_check_from_source"

# 0) Entorno
if [[ -f "$VENV/bin/activate" ]]; then
  . "$VENV/bin/activate"
fi

# 1) Asegura alias de preset estacional
if [[ ! -L "$PRESET_LINK" ]]; then
  ln -sfn "$(basename "$PRESET_SRC")" "$PRESET_LINK"
fi

# 2) Hash de datos 1D (para auditoría rápida)
if [[ -s "$DATA_1D" ]]; then
  echo "[DATA HASH 1D]"
  shasum -a 256 "$DATA_1D" | awk '{print $1}'
else
  echo "[WARN] No encuentro DATA 1D en $DATA_1D" >&2
fi

# 3) Ejecuta el motor con lógica E1_Y2 sobre 2022 (compute from source)
python -m mini_accum.cli \
  --config "$PRESET_LINK" \
  --start 2022-01-01 --end 2022-12-31 \
  --suffix "$SUFFIX"

# 4) Localiza el KPI recién generado (robusto)
KPI_CSV="$(ls -1t "$ROOT"/reports/mini_accum/*_kpis__${SUFFIX}.csv 2>/dev/null | head -n1 || true)"
if [[ -z "${KPI_CSV:-}" || ! -s "$KPI_CSV" ]]; then
  # fallback con find
  KPI_CSV="$(find "$ROOT/reports/mini_accum" -type f -name "*_kpis__${SUFFIX}.csv" -print -quit 2>/dev/null || true)"
fi
[[ -n "${KPI_CSV:-}" && -s "$KPI_CSV" ]] || { echo "[ERR] No salió KPI CSV para ${SUFFIX}" >&2; exit 2; }

EQ_CSV="${KPI_CSV/_kpis__/_equity__}"
echo "[KPI] $KPI_CSV"

# 5) Extrae métricas de forma robusta (aliases; fallback → equity)
read SATS MDDRATIO <<EOF
$(python - "$KPI_CSV" "$EQ_CSV" << 'PY'
import sys, csv, os

kpi_csv = sys.argv[1]
eq_csv  = sys.argv[2] if len(sys.argv)>2 else ""

row={}
with open(kpi_csv, newline='') as f:
    r = csv.DictReader(f)
    row = next(r, {})

# sats_mult con aliases
sats = ""
for key in ("sats_mult","net_btc_ratio","netBTC","btc_mult","mult","sats"):
    if key in row and row[key] and row[key].strip():
        sats = row[key].strip()
        break

def last_equity(path):
    if not path or not os.path.exists(path): return ""
    eq_col=None; last=""
    with open(path, newline='') as f:
        rdr = csv.reader(f)
        header = next(rdr, [])
        eq_col = header.index("equity") if "equity" in header else (len(header)-1 if header else 1)
        for rr in rdr:
            if len(rr)>eq_col and rr[eq_col].strip():
                last = rr[eq_col].strip()
    return last

if not sats:
    sats = last_equity(eq_csv)

# mdd_vs_hodl con alias+reconstrucción
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

print((sats or "NA"), (mdd or "n/d"))
PY
)
EOF

if [[ "$SATS" == "NA" || -z "$SATS" ]]; then
  echo "[FAIL] sats_mult no encontrado (KPI: $KPI_CSV)"
  exit 3
fi

echo "[RESULT] sats_mult=$SATS | mdd_vs_hodl=$MDDRATIO"
