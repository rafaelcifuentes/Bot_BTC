#!/usr/bin/env bash
set -euo pipefail
echo "== STRESS START $(date -u +'%Y-%m-%dT%H:%M:%SZ') =="
command -v yq >/dev/null 2>&1 || { echo "[ERR] yq no encontrado en PATH"; exit 1; }

FEE_GRID="${FEE_GRID:-2 3 4 5}"
SLIP_GRID="${SLIP_GRID:-1 2 3}"
E1_S_MIN="${E1_S_MIN:-2.9}"; E1_M_MAX="${E1_M_MAX:-0.12}"
V1_S_MIN="${V1_S_MIN:-1.0}"; V1_M_MAX="${V1_M_MAX:-1.0}"

MAP=(
  "PROD_E1_Y2_2022 reports/mini_accum/_freezes/E1_Y2_2022.freeze.txt"
  "PROD_KISSv1_2023  reports/mini_accum/_freezes/V1TOP_2023.freeze.txt"
  "PROD_KISSv1_2024  reports/mini_accum/_freezes/V1TOP_2024.freeze.txt"
  "PROD_KISSv1_2025H1 reports/mini_accum/_freezes/V1TOP_2025H1.freeze.txt"
)

pass=0; fail=0
printf "%-18s | %-2s | %5s | %4s/%-4s | %10s | %10s | %9s | %s\n"   "TAG" "PL" "FLIPS" "FEE" "SLIP" "S0" "S_ADJ" "MDD" "RES"

for pair in "${MAP[@]}"; do
  read -r TAG FREEZE <<<"$pair"
  [[ -n "${TAG:-}" ]] || continue
  if [[ ! -f "$FREEZE" ]]; then echo "[MISS] $FREEZE"; continue; fi
  s0="$(yq -r '.kpis.sats_mult   // .sats_mult   // "NA"' "$FREEZE")"
  m0="$(yq -r '.kpis.mdd_vs_hodl // .mdd_vs_hodl // "NA"' "$FREEZE")"
  flips="$(yq -r '.kpis.flips    // .flips       // 0' "$FREEZE")"
  case "$TAG" in
    PROD_E1_Y2_2022*) POL="E1"; smin="$E1_S_MIN"; mmax="$E1_M_MAX" ;;
    *)                 POL="V1"; smin="$V1_S_MIN"; mmax="$V1_M_MAX" ;;
  esac
  for FEE in $FEE_GRID; do
    for SLIP in $SLIP_GRID; do
      ratio="$(awk -v f="$FEE" -v s="$SLIP" 'BEGIN{printf "%.6f",(f+s)/3.0}')"
      s_adj="$(awk -v s="$s0" -v r="$ratio" 'BEGIN{ if(r>0) printf "%.6f",(s+0)/r; else print "NA"}')"
      if awk -v sa="$s_adj" -v sm="$smin" -v m="$m0" -v mm="$mmax" 'BEGIN{exit !((sa+0) >= (sm+0) && (m+0) <= (mm+0))}'; then
        RES="PASS"; ((pass++))
      else
        RES="FAIL"; ((fail++))
      fi
      printf "%-18s | %-2s | %5s | %4s/%-4s | %10s | %10s | %9s | %s\n"         "$TAG" "$POL" "$flips" "$FEE" "$SLIP" "$s0" "$s_adj" "$m0" "$RES"
    done
  done
done

echo "----"
echo "STRESS PASS=$pass  FAIL=$fail"
exit 0
