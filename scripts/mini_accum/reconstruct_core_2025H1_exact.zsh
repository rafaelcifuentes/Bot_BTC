#!/usr/bin/env zsh
set -euo pipefail

ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
VENV="$ROOT/.venv"
PRESET_LINK="$ROOT/configs/mini_accum/presets/PROD_TOP.yaml"
PRESET_SRC="$ROOT/configs/mini_accum/presets/CORE_2025.yaml"

# Pins canónicos que dio el Comandante (2025H1)
EXPECT_HASH_4H="2bf8c646589db1cd52fdd7b4bfc822860d9b8283fe3f2e129961732a3ff0d947"
EXPECT_HASH_1D="4fc2fdcac21ac1f9acf0ceb624d1622c990b3e4e76302cf3d624c448ebc2441b"

# Targets contrato
EXPECT_SATS="1.138462"
EXPECT_MDD="0.741494"
EXPECT_FLIPS="2"

# 0) Entorno
if [[ -f "$VENV/bin/activate" ]]; then
  . "$VENV/bin/activate"
fi

echo "== RECONSTRUCT CORE_2025 / 2025H1 (motor) =="

# 1) Verifica alias de preset
if [[ ! -L "$PRESET_LINK" ]]; then
  echo "[WARN] PROD_TOP.yaml no es symlink. Lo creo -> CORE_2025.yaml"
  ln -sfn "$(basename "$PRESET_SRC")" "$PRESET_LINK"
fi
REAL_TGT="$(readlink "$PRESET_LINK" || true)"
if [[ "$REAL_TGT" != "$(basename "$PRESET_SRC")" ]]; then
  echo "[ERR] PROD_TOP.yaml apunta a '$REAL_TGT' (esperado: $(basename "$PRESET_SRC"))."
  exit 10
fi
echo "[OK] Preset alias: PROD_TOP.yaml -> CORE_2025.yaml"

# 2) Hashes de datos (auditoría dura)
need_fail=0
if [[ -s "$ROOT/data/ohlc/4h/BTC-USD.csv" ]]; then
  H4="$(shasum -a 256 "$ROOT/data/ohlc/4h/BTC-USD.csv" | awk '{print $1}')"
  echo "[DATA 4H] $H4"
  if [[ "$H4" != "$EXPECT_HASH_4H" ]]; then
    echo "[ERR] Hash 4H != esperado. Esto cambia flips/ROI. (esperado $EXPECT_HASH_4H)"
    need_fail=1
  fi
else
  echo "[ERR] Falta $ROOT/data/ohlc/4h/BTC-USD.csv"
  need_fail=1
fi

if [[ -s "$ROOT/data/ohlc/1d/BTC-USD.csv" ]]; then
  H1="$(shasum -a 256 "$ROOT/data/ohlc/1d/BTC-USD.csv" | awk '{print $1}')"
  echo "[DATA 1D] $H1"
  if [[ "$H1" != "$EXPECT_HASH_1D" ]]; then
    echo "[ERR] Hash 1D != esperado. Macro D1 cambia. (esperado $EXPECT_HASH_1D)"
    need_fail=1
  fi
else
  echo "[ERR] Falta $ROOT/data/ohlc/1d/BTC-USD.csv"
  need_fail=1
fi

if (( need_fail )); then
  echo "[FAIL] Datos no coinciden con el freeze. Restaura CSVs canónicos antes de continuar."
  exit 11
fi

# 3) Muestra claves sensibles del YAML (solo informativo)
echo "--- CORE_2025.yaml (claves sensibles) ---"
grep -E '(^|[^#])(bar:|fee:|slip:|EMA21|EMA55|macro|G200|RB|H30|DD15|bull|BULL|cost|adx|ADX)' \
  "$PRESET_SRC" || true
echo "-----------------------------------------"

# 4) Ejecuta el motor con ventana EXACTA OOS 2025H1
#    Fin exclusivo para evitar fugas: usamos 2025-07-01
START="2025-01-01"
END_EXCL="2025-07-01"
SUFFIX="OOS_2025H1_core_rebuild_exact"

python -m mini_accum.cli \
  --config "$PRESET_LINK" \
  --start "$START" --end "$END_EXCL" \
  --suffix "$SUFFIX"

# 5) Localiza KPI y extrae métricas (robusto)
KPI_CSV="$(ls -1t "$ROOT"/reports/mini_accum/*_kpis__${SUFFIX}.csv 2>/dev/null | head -n1 || true)"
[[ -n "${KPI_CSV:-}" && -s "$KPI_CSV" ]] || { echo "[ERR] No salió KPI para ${SUFFIX}"; exit 12; }
echo "[KPI] $KPI_CSV"

read SATS MDDRATIO FLIPS <<EOF
$(python - "$KPI_CSV" << 'PY'
import sys, csv
row={}
with open(sys.argv[1], newline='') as f:
    r=csv.DictReader(f); row=next(r, {})
def pick(d, keys): 
    for k in keys:
        v=d.get(k)
        if v and str(v).strip(): return str(v).strip()
    return ""
sats = pick(row, ("sats_mult","net_btc_ratio","netBTC","btc_mult","mult","sats"))
mdd  = pick(row, ("mdd_vs_hodl","mdd_vs_hodl_ratio"))
if not mdd:
    mm = pick(row, ("mdd_model","mdd_model_usd","mdd_model_btc","mdd_model_pct"))
    mh = pick(row, ("mdd_hodl","mdd_hodl_usd","mdd_hodl_btc","mdd_hodl_pct"))
    try:
        if mm and mh and float(mh)!=0: mdd = str(float(mm)/float(mh))
    except: mdd=""
flips = pick(row, ("flips","flips_total","flips_per_year"))
print((sats or "NA"), (mdd or "n/d"), (flips or "n/a"))
PY
)
EOF

if [[ "$SATS" == "NA" || -z "$SATS" ]]; then
  echo "[FAIL] sats_mult no encontrado (KPI: $KPI_CSV)"
  exit 13
fi

echo "[RESULT] sats_mult=$SATS | mdd_vs_hodl=$MDDRATIO | flips=$FLIPS"

# 6) Comparación contrato (tolerancias estrictas)
eps_sats="0.000002"
pass=1

python - <<PY || pass=0
exp=float("$EXPECT_SATS"); got=float("$SATS")
assert abs(got-exp) <= float("$eps_sats"), f"sats_mult mismatch: got={got} exp={exp}"
PY
if (( ! pass )); then echo "[FAIL] sats_mult != contrato ($SATS vs $EXPECT_SATS)"; exit 20; fi

if [[ "$MDDRATIO" != "n/d" ]]; then
  python - <<PY || pass=0
exp=float("$EXPECT_MDD"); got=float("$MDDRATIO")
# tolerancia suavecita por redondeo
assert abs(got-exp) <= 1e-6, f"mdd_vs_hodl mismatch: got={got} exp={exp}"
PY
  if (( ! pass )); then echo "[FAIL] mdd_vs_hodl != contrato ($MDDRATIO vs $EXPECT_MDD)"; exit 21; fi
fi

if [[ "$FLIPS" != "$EXPECT_FLIPS" ]]; then
  echo "[FAIL] flips != contrato ($FLIPS vs $EXPECT_FLIPS)"
  exit 22
fi

echo "[PASS] Reproducción EXACTA del contrato CORE_2025 (2025H1). Mantra ON 🚀🟠"
