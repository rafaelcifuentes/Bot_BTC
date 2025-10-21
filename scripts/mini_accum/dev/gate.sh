# scripts/mini_accum/dev/gate.sh
#!/usr/bin/env bash
set -euo pipefail

# Umbrales por defecto (puedes override con env al invocar make)
E1_S_MIN="${E1_S_MIN:-2.9}"
E1_M_MAX="${E1_M_MAX:-0.12}"
V1_S_MIN="${V1_S_MIN:-1.0}"
V1_M_MAX="${V1_M_MAX:-1.0}"

pass=0
fail=0

cmp_ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit !( (a+0) >= (b+0) )}'; }  # a >= b ? exit 0 : 1
cmp_le() { awk -v a="$1" -v b="$2" 'BEGIN{exit !( (a+0) <= (b+0) )}'; }  # a <= b ? exit 0 : 1

check_one() {
  local tag="$1" freeze="$2" policy s m ok_s ok_m
  [ -f "$freeze" ] || { echo "[MISS] $freeze"; ((++fail)); return; }

  # KPIs sellados o legacy (fallbacks)
  s="$(yq -r '.kpis.sats_mult    // .sats_mult'      "$freeze")"
  m="$(yq -r '.kpis.mdd_vs_hodl  // .mdd_vs_hodl'    "$freeze")"

  if [[ "$tag" == PROD_E1_Y2_2022* ]]; then
    policy="E1";  cmp_ge "$s" "$E1_S_MIN" && ok_s=OK || ok_s=NO
                   cmp_le "$m" "$E1_M_MAX" && ok_m=OK || ok_m=NO
  else
    policy="V1";  cmp_ge "$s" "$V1_S_MIN" && ok_s=OK || ok_s=NO
                   cmp_le "$m" "$V1_M_MAX" && ok_m=OK || ok_m=NO
  fi

  if [[ "$ok_s" == OK && "$ok_m" == OK ]]; then
    printf "%-18s | policy=%-2s | S=%-9s (≥%s) | M=%-9s (≤%s) | PASS\n" \
      "$tag" "$policy" "$s" "$( [[ $policy == E1 ]] && echo "$E1_S_MIN" || echo "$V1_S_MIN")" \
      "$m" "$( [[ $policy == E1 ]] && echo "$E1_M_MAX" || echo "$V1_M_MAX")"
    ((++pass))
  else
    printf "%-18s | policy=%-2s | S=%-9s (≥%s:%s) | M=%-9s (≤%s:%s) | FAIL\n" \
      "$tag" "$policy" "$s" "$( [[ $policy == E1 ]] && echo "$E1_S_MIN" || echo "$V1_S_MIN")" "$ok_s" \
      "$m" "$( [[ $policy == E1 ]] && echo "$E1_M_MAX" || echo "$V1_M_MAX")" "$ok_m"
    ((++fail))
  fi
}

# Mapeo tag ↔ freeze (KISS)
while read -r TAG FREEZE; do
  [ -n "$TAG" ] || continue
  check_one "$TAG" "$FREEZE"
done <<'MAP'
PROD_E1_Y2_2022   reports/mini_accum/_freezes/E1_Y2_2022.freeze.txt
PROD_KISSv1_2023  reports/mini_accum/_freezes/V1TOP_2023.freeze.txt
PROD_KISSv1_2024  reports/mini_accum/_freezes/V1TOP_2024.freeze.txt
PROD_KISSv1_2025H1 reports/mini_accum/_freezes/V1TOP_2025H1.freeze.txt
MAP

echo "----"
echo "PASS=$pass  FAIL=$fail"
[ "$fail" -eq 0 ]