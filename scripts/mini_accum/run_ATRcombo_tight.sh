#!/usr/bin/env bash
set -euo pipefail
set -x

CFG_SRC="configs/final/F2_ATRcombo_TOP.yaml"
test -f "$CFG_SRC" || { echo "[ERR] Falta $CFG_SRC"; exit 1; }

apply_portable_sed () {
  local expr="$1" file="$2"
  if command -v gsed >/dev/null 2>&1; then gsed -i "$expr" "$file"; else sed -i '' "$expr" "$file"; fi
}

P_LIST=(45 50 55 60)
Y_LIST=(0.02 0.03 0.04)

for P in "${P_LIST[@]}"; do
  for Y in "${Y_LIST[@]}"; do
    YTAG=$(printf '%03d' "$(awk -v y="$Y" 'BEGIN{print int(y*100)}')")
    CFG="configs/exp/F2_ATRxb36_dw76_p${P}_y${YTAG}.yaml"

    if [ ! -f "$CFG" ]; then
      cp "$CFG_SRC" "$CFG"
      apply_portable_sed "s/^\\([[:space:]]*percentile_p:\\).*/\\1 ${P}/" "$CFG"
      apply_portable_sed "s/^\\([[:space:]]*yellow_band_pct:\\).*/\\1 ${Y}/" "$CFG"
      apply_portable_sed "s/^\\([[:space:]]*pause_affects_sell:\\).*/\\1 true/" "$CFG"
      # asegurar bandera 'enabled: true' por si estuviera a false en la plantilla
      apply_portable_sed "s/^\\([[:space:]]*enabled:\\).*/\\1 true/" "$CFG"
    fi

    echo "[CFG] $CFG ⇒ $(grep -E 'percentile_p|yellow_band_pct|pause_affects_sell|enabled' "$CFG" | tr '\n' ' ' )"

    REPORT_SUFFIX="v3p3N2g-F2-xb36-dw76-ATRp${P}-y${YTAG}-oos24H1" \
    mini-accum-backtest --config "$CFG" --start 2024-01-01 --end 2024-06-30

    REPORT_SUFFIX="v3p3N2g-F2-xb36-dw76-ATRp${P}-y${YTAG}-oos23Q4" \
    mini-accum-backtest --config "$CFG" --start 2023-10-01 --end 2023-12-31
  done
done
