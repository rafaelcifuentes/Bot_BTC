#!/usr/bin/env zsh
emulate -L zsh
set -euo pipefail

# Carga helpers si existen (safe en bash/zsh)
[[ -f scripts/mini_accum/helpers.zsh ]] && source scripts/mini_accum/helpers.zsh

# --- Base y candidato ---
BASE=${BASE:-$(pick_latest 'reports/mini_accum/*_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv')}
# Prioriza v2.0; si no hay, cae a v1.2
CAND=${CAND:-$(pick_latest 'reports/mini_accum/*_kpis__WF_2025_v2_0*.csv')}
: ${CAND:=$(pick_latest 'reports/mini_accum/*_kpis__WF_2025_v1_2*.csv')}

LABEL=${LABEL:-WF_2025_v2_0_vs_BASE_2025H1}
print -rl -- "[LABEL] $LABEL"

# --- Pre-checks (anti-NaN) ---
assert_kpi_has_sats "$BASE"
assert_kpi_has_sats "$CAND"

# --- Gate ---
if safe_gate "$BASE" "$CAND" "$LABEL"; then
  DOC="## $(date +%F) — Gate PASS ${LABEL}\n- BASE: ${BASE}\n- CAND: ${CAND}\n- Decisión: **promocionar** candidato (cumple lift≥+5% y riesgo≤BASE)."
else
  DOC="## $(date +%F) — Gate FAIL ${LABEL}\n- BASE: ${BASE}\n- CAND: ${CAND}\n- Decisión: **mantener BASE**; candidato OFF (opt-in)."
fi

# --- Log documental mínimo (no falla si no existen) ---
PROG=docs/mini_accum/Progreso.md
DEC=docs/mini_accum/decisiones.md
mkdir -p docs/mini_accum
{ echo ""; echo "$DOC"; } | tee -a "$PROG" "$DEC" >/dev/null || true

echo "[gate_latest.sh] done"
return 0 2>/dev/null || exit 0