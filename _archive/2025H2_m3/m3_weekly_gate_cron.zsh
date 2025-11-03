#!/usr/bin/env zsh
set -euo pipefail

ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
PRESET="$ROOT/configs/mini_accum/presets/CORE_2025.yaml"
OVERLAY="$ROOT/configs/mini_accum/overlays/bull_hold_puro.yaml"

# Ventana por defecto: desde inicio de H2 hasta hoy (UTC)
START="${1:-2025-07-01}"
END="${2:-$(date -u +%Y-%m-%d)}"
SUF="OOS_2025H2_m3_puro"

# 1) Ejecutar overlay M3 "puro" (replica freeze H1: 0 flips del CORE)
zsh "$ROOT/scripts/mini_accum/run_with_overlay.zsh" "$PRESET" "$OVERLAY" "$START" "$END" "$SUF"

# 2) Tomar equity recién creado
EQ="$(ls -1t "$ROOT"/reports/mini_accum/*equity__${SUF}.csv | head -n1)"

# 3) Gate neto y bull_pct con costes (borrow 10%, funding 0)
zsh "$ROOT/scripts/mini_accum/ab_m3_check.zsh" "$EQ" 0.10 0.00 --min-net 1.05 --min-bull-pct 0.90
RC=$?

STAMP=$(date -u +%Y%m%d_%H%M%S)
DEC_DIR="$ROOT/reports/mini_accum/_decisions"
mkdir -p "$DEC_DIR"

if [[ "$RC" -eq 0 ]]; then
  echo "[PASS] M3 gate semanal (net≥1.05 y bull_pct≥0.90)."
  cp -a "$EQ" "$DEC_DIR/M3_PASS_${STAMP}.equity.csv"
  {
    echo "PASS net/bull gate"
    echo "EQUITY=$EQ"
    echo "START=$START END=$END"
  } > "$DEC_DIR/M3_PASS_${STAMP}.md"
else
  echo "[FAIL] M3 gate semanal. Mantener M2/M1."
  cp -a "$EQ" "$DEC_DIR/M3_FAIL_${STAMP}.equity.csv"
  {
    echo "FAIL net/bull gate"
    echo "EQUITY=$EQ"
    echo "START=$START END=$END"
  } > "$DEC_DIR/M3_FAIL_${STAMP}.md"
fi

exit "$RC"
