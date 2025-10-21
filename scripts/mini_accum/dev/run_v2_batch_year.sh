#!/usr/bin/env bash
set -euo pipefail
YEAR="${1:?Uso: run_v2_batch_year.sh <YEAR> <BASE_KPI.csv>}"
BASE_KPI="${2:?Falta BASE_KPI.csv}"
REPORTS_DIR="reports/mini_accum/WF_${YEAR}/v2_0"
mkdir -p "${REPORTS_DIR}" tmp/_v2cfg

# Linkea datos del año (ajusta si tienes paths por-año separados)
ln -sf ../../tmp_wf/BTC-USD_4h_WF_${YEAR}.csv data/ohlc/4h/BTC-USD.csv || true
ln -sf ../../tmp_wf/BTC-USD_1d_WF_${YEAR}.csv data/ohlc/1d/BTC-USD.csv || true

presets=(E1 E2 E3 E3a E3b_offhib E3b_onhib E3c E3c_wide E3d_adx22)
for P in "${presets[@]}"; do
  SRC="configs/mini_accum/v2_0/${P}.yaml"
  TMP="tmp/_v2cfg/${P}_${YEAR}.yaml"
  sed "s/WF_YYYY/WF_${YEAR}/g" "${SRC}" > "${TMP}"
  echo "[RUN] ${SRC} → OOS_${YEAR}_${P}"
  python -m mini_accum.cli \
    --config "${TMP}" \
    --start "${YEAR}-01-01" --end "${YEAR}-12-31" \
    --suffix "OOS_${YEAR}_${P}" || true
done

echo "[RUN] Resumen gate → ${REPORTS_DIR}"
python scripts/mini_accum/collect_v2_gate.py "${BASE_KPI}" "${REPORTS_DIR}"

echo "[OK] Ver:"
echo "  ${REPORTS_DIR}/v2_${YEAR}_gate_summary.csv"
echo "  ${REPORTS_DIR}/v2_${YEAR}_gate_summary.md"
