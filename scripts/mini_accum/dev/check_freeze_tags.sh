#!/usr/bin/env bash
set -euo pipefail

check_tag () {
  local tag="$1" freeze="$2"
  local ann s m d1 h4
  ann=$(git for-each-ref "refs/tags/$tag" --format='%(contents)' || true)
  s=$(yq -r '.kpis.sats_mult // .sats_mult' "$freeze")
  m=$(yq -r '.kpis.mdd_vs_hodl // .mdd_vs_hodl' "$freeze")
  d1=$(yq -r '.data_hashes.data_1d_sha256 // .data_1d_sha256' "$freeze")
  h4=$(yq -r '.data_hashes.data_4h_sha256 // .data_4h_sha256' "$freeze")
  printf "%-18s : " "$tag"
  echo "$ann" | grep -q "NetBTC=$s MDD_vs_HODL=$m | 1D=$d1 4H=$h4" && echo "OK" || { echo "MISMATCH"; return 1; }
}

err=0
check_tag PROD_E1_Y2_2022   reports/mini_accum/_freezes/E1_Y2_2022.freeze.txt   || err=1
check_tag PROD_KISSv1_2023  reports/mini_accum/_freezes/V1TOP_2023.freeze.txt   || err=1
check_tag PROD_KISSv1_2024  reports/mini_accum/_freezes/V1TOP_2024.freeze.txt   || err=1
check_tag PROD_KISSv1_2025H1 reports/mini_accum/_freezes/V1TOP_2025H1.freeze.txt || err=1
exit $err
