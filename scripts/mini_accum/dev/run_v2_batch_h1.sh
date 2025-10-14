#!/usr/bin/env zsh
set -euo pipefail

ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
cd "$ROOT"

START="2025-01-01"
END="2025-06-30"

typeset -a CFGS
CFGS=(
  "configs/mini_accum/config_WF_2025_v2_0_E1.yaml:WF_2025_v2_0_E1_H1"
  "configs/mini_accum/config_WF_2025_v2_0_E2.yaml:WF_2025_v2_0_E2_H1"
  "configs/mini_accum/config_WF_2025_v2_0_E3.yaml:WF_2025_v2_0_E3_H1"
  "configs/mini_accum/config_WF_2025_v2_0_E3a.yaml:WF_2025_v2_0_E3a_H1"
  "configs/mini_accum/config_WF_2025_v2_0_E3b_offhib.yaml:WF_2025_v2_0_E3b_offhib_H1"
  "configs/mini_accum/config_WF_2025_v2_0_E3b_onhib.yaml:WF_2025_v2_0_E3b_onhib_H1"
  "configs/mini_accum/config_WF_2025_v2_0_E3c.yaml:WF_2025_v2_0_E3c_H1"
  "configs/mini_accum/config_WF_2025_v2_0_E3c_wide.yaml:WF_2025_v2_0_E3c_wide_H1"
  "configs/mini_accum/config_WF_2025_v2_0_E3d_adx22.yaml:WF_2025_v2_0_E3d_adx22_H1"
)

if [[ -z "${BASE_KPI:-}" ]]; then
  echo "[ERR] BASE_KPI no definido."
  exit 2
fi

echo "[RUN] Batch V2.0 H1 — $(date)"
for pair in "${CFGS[@]}"; do
  cfg="${pair%%:*}"
  suf="${pair##*:}"
  [[ -f "$cfg" ]] || { echo "[SKIP] $cfg no existe"; continue; }
  echo "[RUN] $cfg  →  $suf"
  python scripts/mini_accum/dev/dev.py \
    --config "$cfg" \
    --start "$START" --end "$END" \
    --suffix "$suf" \
    --strict --write-docs \
    --base-kpi "$BASE_KPI" || true
done

echo "[OK] Batch terminado."
echo "[RUN] Resumen gate → scripts/mini_accum/collect_v2_gate.py"
python scripts/mini_accum/collect_v2_gate.py "$BASE_KPI" reports/mini_accum/WF_2025/v2_0 || true
