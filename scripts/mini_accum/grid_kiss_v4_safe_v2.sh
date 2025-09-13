#!/usr/bin/env bash
set -Eeuo pipefail

is_sourced() { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }
fail() { if is_sourced; then return 1; else exit 1; fi; }
trap 'echo "[ERROR] línea $LINENO"; fail' ERR

command -v yq >/dev/null || { echo "Falta yq (brew install yq)"; fail; }

BASE="configs/mini_accum/config.yaml"
[[ -f "$BASE" ]] || { echo "No existe $BASE"; fail; }
TMP="$(mktemp)"; cp "$BASE" "$TMP"

START="2022-07-01"
END="2024-06-30"

# Devuelve el archivo *_kpis*.csv más nuevo (con o sin sufijo)
last_kpis() { ls -t reports/mini_accum/*_kpis*.csv 2>/dev/null | head -1 || true; }

for EF in 12 13 14; do
  for ES in 29 31 33; do
    for DW in 96 72; do
      EF="$EF" ES="$ES" DW="$DW" yq -i '
        .signals.ema_fast  = (env(EF)|tonumber) |
        .signals.ema_slow  = (env(ES)|tonumber) |
        .signals.exit_active.confirm_bars = 2 |
        .anti_whipsaw.dwell_bars_min_between_flips = (env(DW)|tonumber) |
        .filters.adx.enabled = true |
        .filters.adx.period  = 14 |
        .filters.adx.min     = 25 |
        .filters.exit_margin.enabled    = true |
        .filters.exit_margin.margin_pct = 0.0025
      ' "$TMP"

      SUF="KISSv4-ef${EF}-es${ES}-dw${DW}-adx25"
      REPORT_SUFFIX="$SUF" mini-accum-backtest --config "$TMP" --start "$START" --end "$END"

      KP=$(last_kpis)
      if [[ -n "$KP" && -f "$KP" ]]; then
        awk -F, -v suf="$SUF" 'NR==2{
          nb=$1; mdd=$4; fpy=$7;
          pass=(nb>=1.05 && mdd<=0.85 && fpy<=26);
          printf "%s => netBTC=%.4f  mdd_vs_HODL=%.3f  fpy=%.2f  %s\n",
                 suf, nb, mdd, fpy, (pass?"PASS":"FAIL")
        }' "$KP"
      else
        echo "$SUF => (KPIs no encontrados todavía)"
      fi
    done
  done
done

rm -f "$TMP"
echo "[OK] Grid terminado."