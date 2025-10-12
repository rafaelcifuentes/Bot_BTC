#!/usr/bin/env zsh
emulate -L zsh
set -euo pipefail

# (opcional) activar venv si existe, para pandas/numpy usados en helpers
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

# helpers KISS
if [[ -f scripts/mini_accum/helpers.zsh ]]; then
  source scripts/mini_accum/helpers.zsh
else
  echo "[GATE] SKIP: no helpers.zsh"; echo "[gate_latest.sh] done"
  return 0 2>/dev/null || exit 0
fi

# localizar base y candidato
BASE=$(pick_latest 'reports/mini_accum/*_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv')
CAND=$(pick_latest 'reports/mini_accum/*_kpis__WF_2025_v1_2*.csv')

if [[ -z "${BASE}" || -z "${CAND}" ]]; then
  echo "[GATE] SKIP: faltan BASE o CAND"
  echo "[gate_latest.sh] done"
  return 0 2>/dev/null || exit 0
fi

# anti-NaN
assert_kpi_has_sats "$BASE" || { echo "[GATE] SKIP: BASE sin sats"; echo "[gate_latest.sh] done"; return 0 2>/dev/null || exit 0; }
assert_kpi_has_sats "$CAND" || { echo "[GATE] SKIP: CAND sin sats"; echo "[gate_latest.sh] done"; return 0 2>/dev/null || exit 0; }

# gate (estricto si GATE_STRICT=1)
if [[ "${GATE_STRICT:-0}" == "1" ]]; then
  kiss_gate_lift "$BASE" "$CAND" 5 1
else
  kiss_gate_lift "$BASE" "$CAND" 5 0
fi

echo "[gate_latest.sh] done"
# Devuelve control tanto si te “sourcean” como si te ejecutan
return 0 2>/dev/null || exit 0
