#!/usr/bin/env bash
set -euo pipefail

# Cargar ENV compartido
if [[ ! -f env_day3.sh ]]; then
  echo "Falta env_day3.sh (ejecuta: source env_day3.sh)"; exit 1
fi
source env_day3.sh

run() { echo "+ $*"; eval "$@"; }

# ========== Paso 1: Validación Fold B pre-checks (Q4 y 2024H1) ==========
# Q4 2023
run "python scripts/run_grid_tzsafe.py \
  --windows '2023Q4:2023-10-01:2023-12-31' \
  --assets BTC-USD --horizons 90 120 --thresholds 0.64 0.66 0.68 \
  --signals_root \"$SIGNALS_ROOT\" --ohlc_root \"$OHLC_ROOT\" \
  --fee_bps $FEE_BPS --slip_bps $SLIP_BPS \
  --partial $PARTIAL $( [[ \"$BREAKEVEN_AFTER_TP1\" == \"1\" ]] && echo --breakeven_after_tp1 ) \
  --risk_total_pct $RISK_TOTAL_PCT --weights $WEIGHTS \
  --gate_pf $GATE_PF --gate_wr $GATE_WR --gate_trades $GATE_TRADES \
  --out_csv reports/val_Q4_p5050.csv --out_top reports/val_Q4_top_p5050.csv"

# 2024H1
run "python scripts/run_grid_tzsafe.py \
  --windows '2024H1:2024-01-01:2024-06-30' \
  --assets BTC-USD --horizons 90 120 --thresholds 0.64 0.66 0.68 \
  --signals_root \"$SIGNALS_ROOT\" --ohlc_root \"$OHLC_ROOT\" \
  --fee_bps $FEE_BPS --slip_bps $SLIP_BPS \
  --partial $PARTIAL $( [[ \"$BREAKEVEN_AFTER_TP1\" == \"1\" ]] && echo --breakeven_after_tp1 ) \
  --risk_total_pct $RISK_TOTAL_PCT --weights $WEIGHTS \
  --gate_pf $GATE_PF --gate_wr $GATE_WR --gate_trades $GATE_TRADES \
  --out_csv reports/val_2024H1_p5050.csv --out_top reports/val_2024H1_top_p5050.csv"

# ========== Paso 2: Merge de ganadores y near-miss (poda suave) ==========
python - <<'PY'
import os, pandas as pd
q4   = pd.read_csv("reports/val_Q4_p5050.csv")
h1   = pd.read_csv("reports/val_2024H1_p5050.csv")

# Ganadores por gates en ambos folds
keys = ["asset","horizon","threshold"]
gq4  = q4[(q4["pf"]>=1.6)&(q4["wr"]>=0.60)&(q4["trades"]>=30)].copy()
gh1  = h1[(h1["pf"]>=1.6)&(h1["wr"]>=0.60)&(h1["trades"]>=30)].copy()
win  = gq4.merge(gh1, on=keys, suffixes=("_Q4","_2024H1"))
if not win.empty:
    out = win[["asset","horizon","threshold","pf_Q4","wr_Q4","trades_Q4","pf_2024H1","wr_2024H1","trades_2024H1"]]
    print("\n=== Ganadores (ambos folds) ===")
    print(out)
    out.to_csv("reports/winners_BothFolds.csv", index=False)
else:
    print("\n=== Ganadores (ambos folds) ===\n(ninguno)")

# Near-miss sobre Q4 para poda suave
cand = q4.sort_values(by=["pf","wr","trades"], ascending=[False,False,False]).head(1).iloc[0]
H  = int(cand.horizon)
TH = round(float(cand.threshold) + 0.02, 2)
print(f"\nNear-miss Q4: H={H} TH_old={cand.threshold} -> TH_new={TH}")

# Guardar variables para el shell
with open(".day3_vars","w") as f:
    f.write(f'export H="{H}"\n')
    f.write(f'export TH_NEW="{TH}"\n')
print('Escrito .day3_vars (H y TH_NEW)')
PY

source .day3_vars
echo "[wf] Poda suave seleccionada: H=$H  TH_NEW=$TH_NEW"

# ========== Paso 3: Revalidación con poda suave ==========
# Q4 con TH_NEW
run "python scripts/run_grid_tzsafe.py \
  --windows '2023Q4:2023-10-01:2023-12-31' \
  --assets BTC-USD --horizons \"$H\" --thresholds \"$TH_NEW\" \
  --signals_root \"$SIGNALS_ROOT\" --ohlc_root \"$OHLC_ROOT\" \
  --fee_bps $FEE_BPS --slip_bps $SLIP_BPS \
  --partial $PARTIAL $( [[ \"$BREAKEVEN_AFTER_TP1\" == \"1\" ]] && echo --breakeven_after_tp1 ) \
  --risk_total_pct $RISK_TOTAL_PCT --weights $WEIGHTS \
  --gate_pf $GATE_PF --gate_wr $GATE_WR --gate_trades $GATE_TRADES \
  --out_csv reports/val_Q4_poda.csv --out_top reports/val_Q4_poda_top.csv"

# 2024H1 con el mismo TH_NEW (walk-forward)
run "python scripts/run_grid_tzsafe.py \
  --windows '2024H1:2024-01-01:2024-06-30' \
  --assets BTC-USD --horizons \"$H\" --thresholds \"$TH_NEW\" \
  --signals_root \"$SIGNALS_ROOT\" --ohlc_root \"$OHLC_ROOT\" \
  --fee_bps $FEE_BPS --slip_bps $SLIP_BPS \
  --partial $PARTIAL $( [[ \"$BREAKEVEN_AFTER_TP1\" == \"1\" ]] && echo --breakeven_after_tp1 ) \
  --risk_total_pct $RISK_TOTAL_PCT --weights $WEIGHTS \
  --gate_pf $GATE_PF --gate_wr $GATE_WR --gate_trades $GATE_TRADES \
  --out_csv reports/val_2024H1_poda.csv --out_top reports/val_2024H1_poda_top.csv"

# ========== Paso 4 (opcional): sanity check hacia atrás (2022H2) ==========
if [[ "${DO_SANITY_2022H2:-1}" == "1" ]]; then
  run "python scripts/run_grid_tzsafe.py \
    --windows '2022H2:2022-07-01:2022-12-31' \
    --assets BTC-USD --horizons \"$H\" --thresholds \"$TH_NEW\" \
    --signals_root \"$SIGNALS_ROOT\" --ohlc_root \"$OHLC_ROOT\" \
    --fee_bps $FEE_BPS --slip_bps $SLIP_BPS \
    --partial $PARTIAL $( [[ \"$BREAKEVEN_AFTER_TP1\" == \"1\" ]] && echo --breakeven_after_tp1 ) \
    --risk_total_pct $RISK_TOTAL_PCT --weights $WEIGHTS \
    --gate_pf $GATE_PF --gate_wr $GATE_WR --gate_trades $GATE_TRADES \
    --out_csv reports/val_2022H2_selected.csv --out_top reports/val_2022H2_selected_top.csv"
fi

echo
echo "[wf] Listo. Revisa CSVs en ./reports:"
echo " - val_Q4_p5050.csv / val_2024H1_p5050.csv"
echo " - winners_BothFolds.csv (si hubo ganadores)"
echo " - val_Q4_poda.csv / val_2024H1_poda.csv"
echo " - val_2022H2_selected.csv (si DO_SANITY_2022H2=1)"
