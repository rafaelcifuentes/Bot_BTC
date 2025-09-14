#!/usr/bin/env bash
set -Eeuo pipefail
S="2025-05-10"; E="2025-09-06"
BASE="configs/mini_accum/presets/CORE_2025.yaml"
TMP=$(mktemp)
cp "$BASE" "$TMP"

# ---- ejemplo de tweaks seguros (edita aquí) ----
# yq -i '.signals.exit_active.confirm_bars=3' "$TMP"
# yq -i '.modules.atr_regime.percentile_p=70' "$TMP"
# -----------------------------------------------

export REPORT_SUFFIX="KISSv4_CORE_EXP"
mini-accum-backtest --config "$TMP" --start "$S" --end "$E"
# KPI en una línea
( setopt NULL_GLOB
  for f in reports/mini_accum/*_kpis__KISSv4_CORE_EXP.csv; do
    awk -F, -v F="$f" 'NR==2{printf "%-60s  netBTC=%.4f  mdd=%.3f  fpy=%.2f\n",F,$1,$4,$7}'; done )
echo "[TMP] $TMP (queda para auditoría)"
