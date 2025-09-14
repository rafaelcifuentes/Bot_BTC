#!/usr/bin/env bash
set -Eeuo pipefail

is_sourced() { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }
fail() { if is_sourced; then return 1; else exit 1; fi; }

trap 'echo "[ERROR] línea $LINENO"; fail' ERR

require() { command -v "$1" >/dev/null 2>&1 || { echo "Falta $1. Instálalo (p.ej. brew install $1)"; fail; }; }
require yq

BASE="configs/mini_accum.yaml"
[[ -f "$BASE" ]] || { echo "No existe $BASE"; fail; }

TMP="$(mktemp)"; cp "$BASE" "$TMP"

START="2022-07-01"
END="2024-06-30"

last_file() { ls -t "$1" 2>/dev/null | head -1 || true; }
rename_if_exists() { local f="$1" suf="$2"; [[ -n "$f" && -f "$f" ]] && mv "$f" "${f%.*}__${suf}.${f##*.}"; }

for EF in 12 13 14; do
  for ES in 29 31 33; do
    for DW in 96 72; do
      # Parcheo seguro del YAML temporal
      yq -i '
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

      EQ=$(last_file "reports/mini_accum/*_equity.csv")
      KP=$(last_file "reports/mini_accum/*_kpis.csv")
      MD=$(last_file "reports/mini_accum/*_summary.md")
      rename_if_exists "$EQ" "$SUF"
      rename_if_exists "$KP" "$SUF"
      rename_if_exists "$MD" "$SUF"

      [[ -n "$KP" && -f "$KP" ]] && awk -F, -v suf="$SUF" 'NR==2{
        nb=$1; mdd=$4; fpy=$7;
        pass=(nb>=1.05 && mdd<=0.85 && fpy<=26);
        printf "%s => netBTC=%.4f  mdd_vs_HODL=%.3f  fpy=%.2f  %s\n",
               suf, nb, mdd, fpy, (pass?"PASS":"FAIL")
      }' "$KP" || echo "$SUF => (sin KPIs encontrados aún)"
    done
  done
done

rm -f "$TMP"
echo "[OK] Grid terminado."
