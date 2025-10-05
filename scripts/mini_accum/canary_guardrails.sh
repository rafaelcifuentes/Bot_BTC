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

discover() {
  local want_atr="$1" # "yes" para cand, "no" para base
  local kind="$2"     # "equity" | "flips"
  local suf
  IFS=' ' read -r -a sufs <<< "${PREFERRED_SUFFIXES:-}"
  if [[ ${#sufs[@]} -eq 0 ]]; then
    sufs=( "__Q2_2025_ATR2x3" "__Q2_2025" "__Q3_2024_ATR2x3" "__Q3_2024" "__CORE_2025_ATR14x2_0" "__CORE_2025" )
  fi
  for suf in "${sufs[@]}"; do
    if [[ "$want_atr" == "yes" && "$suf" != *ATR* ]]; then continue; fi
    if [[ "$want_atr" == "no"  && "$suf" == *ATR* ]]; then continue; fi
    local pick
    if [[ "$kind" == "equity" ]]; then
      pick=$(latest "reports/mini_accum/*_equity__${suf}.csv")
    else
      pick=$(latest "reports/mini_accum/*_flips__${suf}.csv")
    fi
    [[ -n "$pick" ]] && { echo "$pick"; return 0; }
  done
  # Fallbacks
  if [[ "$kind" == "equity" ]]; then
    [[ "$want_atr" == "yes" ]] && latest "reports/mini_accum/*_equity__CORE_2025_ATR14x2_0.csv" || latest "reports/mini_accum/*_equity__CORE_2025.csv"
  else
    [[ "$want_atr" == "yes" ]] && latest "reports/mini_accum/*_flips__CORE_2025_ATR14x2_0.csv" || latest "reports/mini_accum/*_flips__CORE_2025.csv"
  fi
}

# Descubrimiento (si no vienen por ENV)
[[ -z "$BASE_EQ"    ]] && BASE_EQ="$(discover no  equity)"
[[ -z "$CAND_EQ"    ]] && CAND_EQ="$(discover yes equity)"
[[ -z "$BASE_FLIPS" ]] && BASE_FLIPS="$(discover no  flips)"
[[ -z "$CAND_FLIPS" ]] && CAND_FLIPS="$(discover yes flips)"

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