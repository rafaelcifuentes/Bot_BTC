#!/usr/bin/env bash
set -euo pipefail

# Uso:
#   START=YYYY-MM-DD END=YYYY-MM-DD PRESET=configs/mini_accum/presets/CORE_2025.yaml \
#   bash scripts/mini_accum/run_sltp_sweep.sh
#
# Requisitos:
#   - baseline reciente (suffix=CORE_2025) ya corrido con run_oos.sh
#   - scripts: sltp_post.py, compare_ab.py, spa_reality_check.py
#   - overlay semilla: configs/mini_accum/overlays/sl_tp_defensivo_sweep.yaml

: "${START:?START requerido}"
: "${END:?END requerido}"
: "${PRESET:?PRESET requerido}"

BASE_SUFFIX="CORE_2025"             # ajusta si usas otro
SEED_OVERLAY="configs/mini_accum/overlays/sl_tp_defensivo_sweep.yaml"
OUT_DIR="reports/mini_accum"
SPA_DIR="${OUT_DIR}/spa"
mkdir -p "$SPA_DIR"

echo "[SWEEP] Ventana: $START..$END | PRESET=$PRESET | BASE_SUFFIX=$BASE_SUFFIX"

# Localiza artefactos BASE (últimos)
BASE_EQ=$(ls -1t ${OUT_DIR}/*_equity__${BASE_SUFFIX}.csv | head -n1)
BASE_KP=$(ls -1t ${OUT_DIR}/*_kpis__${BASE_SUFFIX}.csv   | head -n1)
if [[ -z "${BASE_EQ:-}" || -z "${BASE_KP:-}" ]]; then
  echo "[SWEEP] No encuentro baseline; corriendo baseline primero..."
  START="$START" END="$END" PRESET="$PRESET" bash scripts/mini_accum/run_oos.sh
  BASE_EQ=$(ls -1t ${OUT_DIR}/*_equity__${BASE_SUFFIX}.csv | head -n1)
  BASE_KP=$(ls -1t ${OUT_DIR}/*_kpis__${BASE_SUFFIX}.csv   | head -n1)
fi

echo "[SWEEP] BASE_EQ=$BASE_EQ"
echo "[SWEEP] BASE_KP=$BASE_KP"

SUMMARY="${OUT_DIR}/sweep_sltp_summary.csv"
if [[ ! -f "$SUMMARY" ]]; then
  echo "suffix,k_sl,k_tp,breakeven,smooth,net_base,mdd_base,fpy_base,net_cand,mdd_cand,fpy_cand,dnet,dmdd,dfpy,gate_ab,spa_p,spa_decision" > "$SUMMARY"
fi

# -------- Combos a probar (ajusta aquí) --------
# Formato: "k_sl k_tp breakeven smooth"
COMBOS=(
  "2.0 3.0 false false"
  "2.5 4.0 false true"
  "3.0 4.0 true  false"
)
# -----------------------------------------------

for combo in "${COMBOS[@]}"; do
  read -r KSL KTP BRK SMO <<< "$combo"
  SUF="${BASE_SUFFIX}__v1_1_sltp_k${KSL}_t${KTP}_brk${BRK}_sm${SMO}"

  # 1) Construye overlay temporal (mutando el seed con Python+yaml)
  TMP=$(mktemp -t sltp_XXXX.yaml)
  python3 - <<'PY' "$SEED_OVERLAY" "$TMP" "$KSL" "$KTP" "$BRK" "$SMO"
import sys, yaml
seed, dst, ksl, ktp, brk, smo = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]=='true', sys.argv[6]=='true'
d = yaml.safe_load(open(seed))
m = d["modules"]["sl_tp_defensivo"]
m["k_sl"] = ksl
m["k_tp"] = ktp
m["use_breakeven_after_tp1"] = brk
m["smooth_atr_with_ema"]     = smo
open(dst,"w").write(yaml.safe_dump(d, sort_keys=False))
print(dst)
PY

  echo "[SWEEP] Overlay tmp → $TMP"

  # 2) Post-proceso SL/TP (no toca core)
  BASE_FL=$(ls -1t ${OUT_DIR}/*_flips__${BASE_SUFFIX}.csv | head -n1)
  python3 scripts/mini_accum/sltp_post.py \
    --config  "$PRESET" \
    --overlay "$TMP" \
    --flips   "$BASE_FL" \
    --start   "$START" --end "$END" \
    --suffix  "$SUF"

  CAND_KP=$(ls -1t ${OUT_DIR}/post_*_kpis__${SUF}.csv | head -n1)
  CAND_EQ=$(ls -1t ${OUT_DIR}/post_*_equity__${SUF}.csv | head -n1)

  # 3) A/B (capturo PASS/FAIL)
  AB_OUT=$(python3 scripts/mini_accum/compare_ab.py \
    --dir "$OUT_DIR" \
    --base-suffix "$BASE_SUFFIX" \
    --cand-suffix "$SUF" \
    --start "$START" --end "$END")
  echo "$AB_OUT"

  GATE_AB=$(echo "$AB_OUT" | awk -F': ' '/^Veredicto:/ {print $2}')
  [[ -z "${GATE_AB:-}" ]] && GATE_AB="UNKNOWN"

  # 4) SPA/Reality Check (stdout→JSON)
  SPA_JSON="${SPA_DIR}/${SUF}_spa.json"
  python3 scripts/mini_accum/spa_reality_check.py \
    --equity-base "$BASE_EQ" \
    --equity-cand "$CAND_EQ" \
    > "$SPA_JSON"
  echo "[SPA] → $SPA_JSON"

  # 5) Extrae KPIs y SPA para resumen CSV
  python3 - <<'PY' "$BASE_KP" "$CAND_KP" "$SPA_JSON" "$SUMMARY" "$SUF" "$KSL" "$KTP" "$BRK" "$SMO"
import sys, json, pandas as pd
base_kp, cand_kp, spa_json, summary, suf, ksl, ktp, brk, smo = sys.argv[1:]
b = pd.read_csv(base_kp).iloc[0]
c = pd.read_csv(cand_kp).iloc[0]
with open(spa_json) as f:
  J = json.load(f)
spa_p = J.get("spa",{}).get("p_consistent", None)
spa_dec = J.get("decision_consistent", "NA")
row = dict(
  suffix=suf, k_sl=ksl, k_tp=ktp, breakeven=brk, smooth=smo,
  net_base=b.get("netBTC"), mdd_base=b.get("MDD") or b.get("mdd") or b.get("MDD_max"),
  fpy_base=b.get("FPY") or b.get("fpy"),
  net_cand=c.get("netBTC"), mdd_cand=c.get("MDD") or c.get("mdd") or c.get("MDD_max"),
  fpy_cand=c.get("FPY") or c.get("fpy"),
)
row["dnet"] = (row["net_cand"] - row["net_base"]) if (row["net_cand"] is not None and row["net_base"] is not None) else None
row["dmdd"] = (row["mdd_cand"] - row["mdd_base"]) if (row["mdd_cand"] is not None and row["mdd_base"] is not None) else None
row["dfpy"] = (row["fpy_cand"] - row["fpy_base"]) if (row["fpy_cand"] is not None and row["fpy_base"] is not None) else None
# Append
import csv
hdr = ["suffix","k_sl","k_tp","breakeven","smooth","net_base","mdd_base","fpy_base","net_cand","mdd_cand","fpy_cand","dnet","dmdd","dfpy","gate_ab","spa_p","spa_decision"]
# read last line of summary to see last gate_ab written? we just append — gate_ab filled from env
gate_ab = None
for i,a in enumerate(sys.argv):
  pass
gate_ab = ""  # placeholder; we’ll fill with sed later if needed
# Simpler: print a line and let caller replace 'gate_ab' via sed:
line = [row.get(k,"") for k in hdr[:-3]] + ["__GATE_AB__", spa_p, spa_dec]
with open(summary, "a", newline="") as f:
  csv.writer(f).writerow(line)
PY

  # Sustituye __GATE_AB__ por el veredicto (PASS/FAIL/…)
  sed -i '' "s/__GATE_AB__/${GATE_AB}/" "$SUMMARY" || true

done

echo
echo "[SWEEP] Resumen CSV → $SUMMARY"
tail -n +1 "$SUMMARY" | column -t -s,