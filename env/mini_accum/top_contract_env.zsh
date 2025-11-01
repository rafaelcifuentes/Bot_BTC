# === TOP / Santo Grial — Entorno reproducible (mini_accum) ===
export ROOT="$HOME/PycharmProjects/Bot_BTC"

# Snapshot (WF 22–24)
export MANIFEST="$ROOT/reports/mini_accum/kiss_v1/_snapshots/20251010_202006__NETBTC_4p340727__DD15_RB1_H30_G200_BULL0/manifest.json"
# Tolerancia
export EPS_ABS=0.000002

# OOS 2025H1 (KPI canónico)
export OOS_2025H1_KPIS="$ROOT/reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_G200_BULL0.csv"
export OOS_2025H1_SATS=1.138462

# Globs tolerantes (opcional)
export OOS_KPI_GLOB="$ROOT/reports/mini_accum/*_kpis__OOS_2025H1_*DD15_RB1_H30_G200_BULL0.csv"
export OOS_EQ_GLOB="$ROOT/reports/mini_accum/*_equity__OOS_2025H1_*DD15_RB1_H30_G200_BULL0.csv"

# Presets canónicos
export PRESET_CORE="$ROOT/configs/mini_accum/presets/CORE_2025.yaml"
export PRESET_E1Y2="$ROOT/configs/mini_accum/presets/E1_Y2.yaml"

# Datos pinneados (UTC)
export D1_CSV="$ROOT/data/ohlc/1d/BTC-USD.csv"
export OHLC4H_CSV="$ROOT/data/ohlc/4h/BTC-USD.norm.csv"
