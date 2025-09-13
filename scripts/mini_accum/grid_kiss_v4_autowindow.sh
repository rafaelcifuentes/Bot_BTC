#!/usr/bin/env bash
set -Eeuo pipefail

# ---- helpers de seguridad ----
is_sourced() { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }
fail() { if is_sourced; then return 1; else exit 1; fi; }
trap 'echo "[ERROR] línea $LINENO"; fail' ERR
command -v yq >/dev/null || { echo "Falta yq (brew install yq)"; fail; }

# ---- datos y YAML ----
BASE="configs/mini_accum/config.yaml"
H4="data/ohlc/4h/BTC-USD.csv"
D1="data/ohlc/1d/BTC-USD.csv"
[[ -f "$BASE" && -f "$H4" && -f "$D1" ]] || { echo "Faltan archivos (YAML/4h/D1)"; fail; }

# Detecta ventana [START, END] desde el 4h (primer y último timestamp)
START=$(awk -F, 'NR==2{print substr($1,1,10); exit}' "$H4")
END=$(tail -n 1 "$H4" | awk -F, '{print substr($1,1,10)}')
echo "[INFO] Ventana detectada en 4h: $START → $END"

# Opción: desactivar macro_filter si exportás MACRO=off
MACRO_FLAG=${MACRO:-on}
TMP="$(mktemp)"; cp "$BASE" "$TMP"
if [[ "$MACRO_FLAG" == "off" ]]; then
  yq -i '.macro_filter.use = false' "$TMP"
  echo "[INFO] macro_filter.use = false (test sin macro)"
fi

# Devuelve el kpis.csv más reciente (con o sin sufijo)
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
        echo "$SUF => (KPIs no encontrados aún)"
      fi
    done
  done
done

rm -f "$TMP"
echo "[OK] Grid terminado en ventana real $START → $END"