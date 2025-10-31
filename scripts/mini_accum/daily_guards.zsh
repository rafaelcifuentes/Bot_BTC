#!/usr/bin/env zsh
set -euo pipefail
ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
MANIFEST="${MANIFEST:-$ROOT/reports/mini_accum/kiss_v1/_snapshots/PROD_TOP/manifest.json}"
LOG="$ROOT/logs/contract.log"

mkdir -p "$ROOT/logs"

{
  print -r -- "==== $(date -u +%FT%TZ) ===="
  # 1) Contrato (Santo Grial)
  source "$ROOT/env/kiss_contract.env" || true
  "$ROOT/scripts/mini_accum/contract_check.zsh"

  # 2) KPI Guard
  . "$ROOT/.venv/bin/activate"
  if [[ -n "${OOS_2025H1_KPIS:-}" && -s "$OOS_2025H1_KPIS" ]]; then
    python "$ROOT/scripts/mini_accum/kpi_kiss_guard.py" \
      --min-sats 1.00 --max-fpy 26 \
      --manifest "$MANIFEST" \
      --oos-kpi "$OOS_2025H1_KPIS"
  else
    python "$ROOT/scripts/mini_accum/kpi_kiss_guard.py" \
      --min-sats 1.00 --max-fpy 26 \
      --manifest "$MANIFEST"
  fi
} >> "$LOG" 2>&1
