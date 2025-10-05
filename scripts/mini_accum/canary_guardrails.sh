#!/usr/bin/env bash
set -euo pipefail
echo "[CANARY] canary_guardrails.sh v1.3 ($(date -u +%F))"

BASE_EQ="${BASE_EQ:-}";      CAND_EQ="${CAND_EQ:-}"
BASE_FLIPS="${BASE_FLIPS:-}"; CAND_FLIPS="${CAND_FLIPS:-}"
WINDOW_DAYS="${WINDOW_DAYS:-30}"
MDD_MAX_DELTA="${MDD_MAX_DELTA:-0}"
FPY_MAX_DELTA="${FPY_MAX_DELTA:-2}"
ROI_MIN_DELTA_ANNUAL="${ROI_MIN_DELTA_ANNUAL:--0.04}"
PREFERRED_SUFFIXES="${PREFERRED_SUFFIXES:-}"

latest() { local pat="$1"; ls -1t ${pat} 2>/dev/null | head -n1 || true; }

# --- discovery por sufijos (respeta PREFERRED_SUFFIXES) ---
IFS=' ' read -r -a SUFS <<< "${PREFERRED_SUFFIXES:-}"

_pick_equity() { # $1=role base|cand -> echo filepath or empty
  local role="$1" suf f=""
  if ((${#SUFS[@]})); then
    for suf in "${SUFS[@]}"; do
      # suf debe incluir su "__..." completo; el patrón NO añade "__"
      # base: evitar sufijos con ATR; cand: requerir ATR
      if [[ "$role" == "base" && "$suf" == *ATR* ]]; then continue; fi
      if [[ "$role" == "cand" && "$suf" != *ATR* ]]; then continue; fi
      f=$(latest "reports/mini_accum/*_equity${suf}.csv")
      [[ -n "$f" ]] && { echo "$f"; return; }
    done
  fi
  # Fallbacks CORE
  if [[ "$role" == "base" ]]; then
    latest "reports/mini_accum/*_equity__CORE_2025.csv"
  else
    latest "reports/mini_accum/*_equity__CORE_2025_ATR14x2_0.csv"
  fi
}

_pick_flips() { # $1=role base|cand
  local role="$1" suf f=""
  if ((${#SUFS[@]})); then
    for suf in "${SUFS[@]}"; do
      if [[ "$role" == "base" && "$suf" == *ATR* ]]; then continue; fi
      if [[ "$role" == "cand" && "$suf" != *ATR* ]]; then continue; fi
      f=$(latest "reports/mini_accum/*_flips${suf}.csv")
      [[ -n "$f" ]] && { echo "$f"; return; }
    done
  fi
  if [[ "$role" == "base" ]]; then
    latest "reports/mini_accum/*_flips__CORE_2025.csv"
  else
    latest "reports/mini_accum/*_flips__CORE_2025_ATR14x2_0.csv"
  fi
}
# --- /discovery por sufijos ---

# Descubrimiento (si no vienen por ENV)
[[ -z "$BASE_EQ"    ]] && BASE_EQ="$(_pick_equity base)"
[[ -z "$CAND_EQ"    ]] && CAND_EQ="$(_pick_equity cand)"
[[ -z "$BASE_FLIPS" ]] && BASE_FLIPS="$(_pick_flips base)"
[[ -z "$CAND_FLIPS" ]] && CAND_FLIPS="$(_pick_flips cand)"

echo "[DEBUG] BASE_EQ=$BASE_EQ"
echo "[DEBUG] CAND_EQ=$CAND_EQ"
echo "[DEBUG] BASE_FLIPS=$BASE_FLIPS"
echo "[DEBUG] CAND_FLIPS=$CAND_FLIPS"

# Validación
for p in "$BASE_EQ" "$CAND_EQ" "$BASE_FLIPS" "$CAND_FLIPS"; do
  [[ -f "$p" ]] || { echo "[ERR] no existe $p"; exit 2; }
done

CMD=( python3 scripts/mini_accum/guardrails_calc.py
  --base-eq "$BASE_EQ"
  --cand-eq "$CAND_EQ"
  --base-flips "$BASE_FLIPS"
  --cand-flips "$CAND_FLIPS"
  --window-days "$WINDOW_DAYS"
  --mdd-max-delta "$MDD_MAX_DELTA"
  --fpy-max-delta "$FPY_MAX_DELTA"
  --roi-min-delta-annual "$ROI_MIN_DELTA_ANNUAL"
)
if [[ -n "${END_OVERRIDE:-}" ]]; then
  CMD+=( --end "$END_OVERRIDE" )
fi

"${CMD[@]}"