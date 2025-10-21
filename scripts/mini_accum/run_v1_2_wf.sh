#!/usr/bin/env zsh
set -euo pipefail

CFG="${1:?uso: scripts/run_v1_2_wf.sh configs/mini_accum/v1_2_YYYY.yaml}"
YEAR=$(grep -oE 'WF_[0-9]{4}' "$CFG" | grep -oE '[0-9]{4}' | tail -n1 || true)
[[ -z "${YEAR:-}" ]] && YEAR=$(echo "$CFG" | grep -oE '[0-9]{4}' | tail -n1)

START="${YEAR}-01-01"
END="${YEAR}-12-31"
SUF="WF_${YEAR}_v1_2"

# Corre SIEMPRE con zsh (el runner usa variables de zsh)
zsh scripts/mini_accum/run_oos.sh "$START" "$END" "$CFG" "$SUF"

# Renombra artefactos
if whence -w rename_last_reports >/dev/null; then
  rename_last_reports "__${SUF}"
else
  echo "[WARN] rename_last_reports no cargado (source helpers.zsh)"
fi
