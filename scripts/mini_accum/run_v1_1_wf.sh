#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# KISS — run_v1_1_wf.sh
# Orquesta V1_1 Walk-Forward end-to-end:
#   1) Merge configs por ventana (WF_2022, WF_2023, WF_2024) con override_v1_1.yaml
#      y fuerza reports_dir aislado: reports/mini_accum/<WIN>/v1_1
#   2) Ejecuta mini-accum-backtest para cada ventana
#   3) Copia artefactos KPI/equity a reports/mini_accum/kiss_v1 con nombre canónico
#   4) Normaliza KPIs (asegura columnas sats_mult, mdd_vs_hodl, fpy, fail_rate)
#   5) Genera tabla comparativa base_v0_1 vs v1_1 en walkforward/tabla_wf_base_vs_v1_1.md
# ----------------------------------------------------------------------------

set -euo pipefail

ROOT=${ROOT:-"$HOME/PycharmProjects/Bot_BTC"}
OVERRIDE=${OVERRIDE:-"$ROOT/data/tmp_wf/override_v1_1.yaml"}
WINDOWS_DEF=(WF_2022 WF_2023 WF_2024)
WINDOWS=(${WINDOWS:-${WINDOWS_DEF[@]}})

err() { echo "[ERR] $*" >&2; }
log() { echo "$*"; }
req() { command -v "$1" >/dev/null 2>&1 || { err "Requiero '$1' en PATH"; exit 1; }; }

# Requisitos
req mini-accum-backtest
req yq

mkdir -p "$ROOT/reports/mini_accum/kiss_v1" \
         "$ROOT/reports/mini_accum/walkforward"

merge_one() {
  local WIN="$1"
  local BASE="$ROOT/data/tmp_wf/config_${WIN}.yaml"
  local OUT="$ROOT/data/tmp_wf/config_${WIN}_v1_1.yaml"
  local RD="reports/mini_accum/${WIN}/v1_1"

  if [[ ! -f "$BASE" ]]; then
    err "No existe base $BASE (corre run_backtest_wf.sh antes)"; return 1
  fi
  if [[ ! -f "$OVERRIDE" ]]; then
    err "No existe override $OVERRIDE"; return 1
  fi
  mkdir -p "$ROOT/$RD"

  # Merge profundo con yq v4 y setear reports_dir por ventana/sufijo
  yq ea '. as $item ireduce ({}; . * $item)' "$BASE" "$OVERRIDE" \
  | yq e ".backtest.reports_dir = \"$RD\"" - \
  > "$OUT"

  log "[OK] merged $OUT → reports_dir=$RD"
}

run_one() {
  local WIN="$1"
  local CFG="$ROOT/data/tmp_wf/config_${WIN}_v1_1.yaml"
  local SRC_DIR="$ROOT/reports/mini_accum/${WIN}/v1_1"
  local DST_DIR="$ROOT/reports/mini_accum/kiss_v1"

  log "[RUN] v1_1 → $WIN"
  mini-accum-backtest --config "$CFG" || true

  # Copia último KPI/equity del directorio aislado de la ventana
  local KPI EQ
  KPI=$(ls -1t "$SRC_DIR"/*_kpis.csv 2>/dev/null | head -1 || true)
  EQ=$(ls -1t "$SRC_DIR"/*_equity.csv 2>/dev/null | head -1 || true)
  if [[ -n "${KPI:-}" && -f "$KPI" ]]; then
    cp -f "$KPI" "$DST_DIR/WF_WF_${WIN#WF_}_kpis__v1_1.csv"
  else
    err "No encontré KPI en $SRC_DIR"
  fi
  if [[ -n "${EQ:-}" && -f "$EQ" ]]; then
    cp -f "$EQ"  "$DST_DIR/WF_WF_${WIN#WF_}_equity__v1_1.csv"
  else
    err "No encontré equity en $SRC_DIR"
  fi
}

# 1) Merge por ventana
for w in "${WINDOWS[@]}"; do
  merge_one "$w"
done

# 2) Run por ventana y copiar artefactos
for w in "${WINDOWS[@]}"; do
  run_one "$w"
done

# 3) Normaliza KPIs (rellena mdd_vs_hodl, fpy, etc si falta)
log "[NORMALIZE] Ejecutando normalizador"
python "$ROOT/scripts/mini_accum/normalizador.py" || log "[WARN] normalizador con warnings"

# 4) Asegura alias base/v1_1 para compare_ab.py (equivalente a bbab_prepare_suffix)
log "[ALIAS] Preparando *_kpis__base_v0_1.csv y *_kpis__v1_1.csv"
cd "$ROOT/reports/mini_accum/kiss_v1"
# Base → ..._kpis__base_v0_1.csv
for f in base_v0_1_*_kpis__WF_WF_*.csv; do
  [[ -f "$f" ]] || continue
  win="${f##*__}"; win="${win%.csv}"
  cp -f "$f" "${win}_kpis__base_v0_1.csv"
done
# Candidato → ..._kpis__v1_1.csv (por si acaso)
for f in v1_1_*_kpis__WF_WF_*.csv; do
  [[ -f "$f" ]] || continue
  win="${f##*__}"; win="${win%.csv}"
  cp -f "$f" "${win}_kpis__v1_1.csv"
done

# 5) Genera tabla base vs v1_1 para WF_2022→WF_2024
log "[TABLE] Generando tabla base vs v1_1 (WF_2022→WF_2024)"
python "$ROOT/scripts/mini_accum/compare_ab.py" \
  --dir "$ROOT/reports/mini_accum/kiss_v1" \
  --base-suffix base_v0_1 \
  --cand-suffix v1_1 \
  --table --start WF_2022 --end WF_2024 \
  --out "$ROOT/reports/mini_accum/walkforward/tabla_wf_base_vs_v1_1.md"

log "[OK] Tabla → $ROOT/reports/mini_accum/walkforward/tabla_wf_base_vs_v1_1.md"