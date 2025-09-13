#!/usr/bin/env bash
set -euo pipefail

# Uso: render.sh OUT_YAML XB DW
# Ej:  bash scripts/mini_accum/render.sh configs/exp/F1_xb36_dw72.yaml 36 72
OUT="${1:-}"
XB="${2:-}"
DW="${3:-}"
MB="${MB:-20}"  # macro_buffer_bps (puedes override con: MB=18 bash ...)

if [[ -z "${OUT}" || -z "${XB}" || -z "${DW}" ]]; then
  echo "Uso: $0 OUT_YAML XB DW   (opcional: MB=XX por env)"
  exit 2
fi

mkdir -p "$(dirname "$OUT")"

cat > "$OUT" <<YAML
version: 0.1
numeraire: BTC
asset: BTC-USD
timezone: UTC

data:
  ohlc_4h_csv: data/ohlc/_rescued/4h/BTC-USD.csv
  ohlc_d1_csv: data/ohlc/_rescued/1d/BTC-USD.csv
  ts_col: timestamp
  price_col: close
  tz_input: UTC

signals:
  ema_fast: 13
  ema_slow: 55
  cross_buffer_bps: ${XB}
  macro_buffer_bps: ${MB}
  entry: ema13 > ema55 and macro_green
  exit_active:
    rule: close < ema13
    confirm_bars: 0
  exit_passive:
    rule: ema13 < ema55

anti_whipsaw:
  dwell_bars_4h: ${DW}
  dwell_bars_min_between_flips: ${DW}
  grace_ttl_bars_4h: 1
  ttl_confirm_bars: 0

costs:
  fee_bps_per_side: 6.0
  slip_bps_per_side: 6.0

backtest:
  reports_dir: reports/mini_accum
  seed_btc: 1.0

kpis:
  accept:
    net_btc_ratio_min: 1.05
    mdd_vs_hodl_ratio_max: 0.85
    flips_per_year_max: 26
    flips_per_month_soft: 2

flip_budget:
  enforce_hard_yearly: true
  hard_per_year: 26
  allow_riskoff_over_budget: true

modules:
  hibernation_chop:
    enabled: false
    lookback_bars_4h: 60
    min_crosses: 4
  grace_ttl:
    enabled: true
  weekly_turnover_budget:
    enabled: true
    flips_per_week_max: 1
    dynamic_by_atr: true
    enforce_hard: false
  atr_regime:
    enabled: true
    lookback_bars: 14
    percentile_p: 40
    yellow_band_pct: 5
    pause_affects_sell: true
  macro_persist:
    enabled: false

force_sell_on_macro_red: false
YAML

echo "[render] $OUT  (xb=${XB}, dw=${DW}, mb=${MB})"
