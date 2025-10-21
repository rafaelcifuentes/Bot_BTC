#!/usr/bin/env bash
set -euo pipefail

write_yaml () {
  local path="$1"
  local variant="$2"   # base|E1|E2|E3|E3a|E3b|E3b_offhib

  mkdir -p "$(dirname "$path")"

  # Defaults v2 comunes
  local V2_BULL_HOLD_ENABLED=true
  local V2_BULL_HOLD_MIN_BARS=2
  local V2_BULL_HOLD_ADX_MIN=22

  local V2_COOLDOWN_ENABLED=true
  local V2_COOLDOWN_BARS=12

  local V2_HIB_ENABLED=true   # por defecto ON; se apaga en E3b_offhib

  local SOFT_PER_WEEK=2

  case "$variant" in
    "E1")
      V2_BULL_HOLD_MIN_BARS=1
      V2_COOLDOWN_BARS=8
      SOFT_PER_WEEK=3
      ;;
    "E2")
      V2_COOLDOWN_BARS=24
      ;;
    "E3"|"E3a"|"E3b")
      # como E3 base
      ;;
    "E3b_offhib")
      V2_HIB_ENABLED=false
      ;;
    *)
      # base
      ;;
  esac

  cat > "$path" <<YAML
# Mini-Accum v2.0 — ${variant} (opt-in)
# Superset v1 (TOP): DD15 / RB1 / H30 / G200 / BULL0

data:
  ohlc_4h_csv: data/ohlc/4h/BTC-USD.csv
  ohlc_d1_csv: data/ohlc/1d/BTC-USD.csv
  ts_col: ts
  tz_input: UTC

backtest:
  reports_dir: reports/mini_accum/WF_2025/v2_0
  seed_btc: 1.0

costs:
  fee_bps_per_side: 5
  slip_bps_per_side: 5

signals:
  ema_fast: 21
  ema_slow: 55
  cross_buffer_bps: 0
  exit_active:
    enabled: true
    confirm_bars: 1
    max_wait_bars_after_confirm: 2
    age_valve_enabled: false

filters:
  adx:
    enabled: true
    period: 14
    min: 22
  exit_atr:
    enabled: false
    period: 14
    mult: 1.5

anti_whipsaw:
  dwell_bars_min_between_flips: 2
  pause_after_flip_bars: 0
  pause_affects_sell: false

flip_budget:
  enforce_hard_yearly: true
  hard_per_year: 999999999
  soft_per_week: ${SOFT_PER_WEEK}

modules:
  atr_regime:
    enabled: false
    percentile_p: 36
    yellow_band_pct: 0.10
    pause_affects_sell: false
  xb_adaptive:
    enabled: false
  v2:
    bull_hold:
      enabled: ${V2_BULL_HOLD_ENABLED}
      min_bars_after_entry: ${V2_BULL_HOLD_MIN_BARS}
      adx_min: ${V2_BULL_HOLD_ADX_MIN}
    cooldown_after_loss:
      enabled: ${V2_COOLDOWN_ENABLED}
      cooldown_bars: ${V2_COOLDOWN_BARS}
    hibernation_on_chop:
      enabled: ${V2_HIB_ENABLED}

risk:
  hard_dd_pct: 0.15
rebalancing:
  frequency: "1W"
horizon:
  h_bars: 30
execution:
  gamma_bps: 200
  bull_bias_bps: 0

tag: DD15_RB1_H30_G200_BULL0
YAML
  echo "[SYNC] $path"
}

# Archivos destino
write_yaml configs/mini_accum/config_WF_2025_v2_0.yaml          base
write_yaml configs/mini_accum/config_WF_2025_v2_0_E1.yaml        E1
write_yaml configs/mini_accum/config_WF_2025_v2_0_E2.yaml        E2
write_yaml configs/mini_accum/config_WF_2025_v2_0_E3.yaml        E3
write_yaml configs/mini_accum/config_WF_2025_v2_0_E3a.yaml       E3a
write_yaml configs/mini_accum/config_WF_2025_v2_0_E3b.yaml       E3b
write_yaml configs/mini_accum/config_WF_2025_v2_0_E3b_offhib.yaml E3b_offhib

echo "[DONE] YAML v2.0 sincronizados con TOP"