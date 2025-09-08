cat > validate_winners.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Uso: $0 <BLOQUE> <HORIZON> <THRESHOLD>"
  echo "BLOQUE: T1_soft_refine | T2_soft_refine_plus | T3_soft_slow | S1_soft_fast | S2_soft_slow | P1_price | SL1_slope"
  exit 1
fi

BLOCK="$1"; HOR="$2"; TH="$3"

# Mapea bloque -> ENV de tendencia
case "$BLOCK" in
  T1_soft_refine)
    export TREND_FILTER=1 TREND_MODE=soft TREND_FREQ=2h TREND_SPAN=24 TREND_BAND_PCT=0.02
    export REARM_MIN=6 HYSTERESIS_PP=0.05
    ;;
  T2_soft_refine_plus)
    export TREND_FILTER=1 TREND_MODE=soft TREND_FREQ=2h TREND_SPAN=24 TREND_BAND_PCT=0.025
    export REARM_MIN=8 HYSTERESIS_PP=0.06
    ;;
  T3_soft_slow)
    export TREND_FILTER=1 TREND_MODE=soft TREND_FREQ=3h TREND_SPAN=36 TREND_BAND_PCT=0.03
    export REARM_MIN=8 HYSTERESIS_PP=0.06
    ;;
  S1_soft_fast)
    export TREND_FILTER=1 TREND_MODE=soft TREND_FREQ=2h TREND_SPAN=24 TREND_BAND_PCT=0.02
    export REARM_MIN=4 HYSTERESIS_PP=0.04
    ;;
  S2_soft_slow)
    export TREND_FILTER=1 TREND_MODE=soft TREND_FREQ=4h TREND_SPAN=48 TREND_BAND_PCT=0.04
    export REARM_MIN=8 HYSTERESIS_PP=0.05
    ;;
  P1_price)
    export TREND_FILTER=1 TREND_MODE=price_only TREND_FREQ=2h TREND_SPAN=24 TREND_BAND_PCT=0.04
    export REARM_MIN=8 HYSTERESIS_PP=0.05
    ;;
  SL1_slope)
    export TREND_FILTER=1 TREND_MODE=slope_only TREND_FREQ=2h TREND_SPAN=24 TREND_BAND_PCT=0.00
    export REARM_MIN=12 HYSTERESIS_PP=0.04
    ;;
  *)
    echo "Bloque no reconocido: $BLOCK" ; exit 2 ;;
esac

COMMON="--assets BTC-USD --signals_root reports/windows_fixed --ohlc_root data/ohlc/1m \
  --fee_bps 6 --slip_bps 6 --partial 50_50 --breakeven_after_tp1 \
  --risk_total_pct 0.75 --weights BTC-USD=1.0 \
  --gate_pf 1.6 --gate_wr 0.60 --gate_trades 30 \
  --out_csv reports/kpis_grid.csv --out_top reports/top_configs.csv"

mkdir -p logs

for W in "2022H2:2022-07-01:2022-12-31" "2023Q4:2023-10-01:2023-12-31" "2024H1:2024-01-01:2024-06-30"; do
  TAG="validate_${BLOCK}_H${HOR}_T${TH}_$(echo "$W" | cut -d: -f1)"
  echo
  echo "========== Validando $BLOCK (H=$HOR, T=$TH) en $W =========="
  set -x
  python scripts/run_grid_tzsafe.py --windows "$W" $COMMON --horizons "$HOR" --thresholds "$TH" 2>&1 | tee "logs/${TAG}_$(date +%F_%H%M).log"
  set +x
  python scripts/quick_kpi_diagnosis.py || true
  echo "=== Cruces [tzsafe] ($TAG) ==="
  grep "\[tzsafe\]" "logs/${TAG}_$(date +%F_%H%M).log" || true
done
BASH

chmod +x validate_winners.sh