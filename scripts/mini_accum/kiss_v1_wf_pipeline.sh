#!/usr/bin/env bash
set -euo pipefail

# Raíz del repo (ajusta si ejecutas desde otro path)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WF_DIR="$REPO_DIR/reports/mini_accum/walkforward"
KPI_GLOB="$REPO_DIR/reports/mini_accum/kiss_v1/"'*kpis__WF_*.csv'
OUT_SUMMARY="$WF_DIR/wf_summary_kpis.csv"
OUT_BEST="$WF_DIR/wf_best_by_window.csv"
OUT_ROADMAP="$WF_DIR/Roadmap_PDCA.md"

# (A) Consolidar WF y chequear criterios de aceptación
python "$REPO_DIR/tools/mini_accum/wf_consolidate.py" \
  --kpis_glob "$KPI_GLOB" \
  --out_summary "$OUT_SUMMARY" \
  --out_best "$OUT_BEST" \
  --out_md "$OUT_ROADMAP" \
  --candidate "DD15_RB1_H30_G200_BULL0" \
  --accept_median_sats 1.05 \
  --accept_fail_rate_max 0.25 \
  --delta_sats_vs_nbhd_min 0.02

# (B) Stress de costes (±5/10/20 bps por lado) → actualiza Roadmap_PDCA.md (sección stress)
python "$REPO_DIR/tools/mini_accum/stress_costs.py" \
  --summary_csv "$OUT_SUMMARY" \
  --out_md_append "$OUT_ROADMAP" \
  --bps 5 10 20

# (C) Tests anti-overfitting (PBO/CSCV + Reality/SPA; DSR si hay datos de Sharpe o retornos)
python "$REPO_DIR/tools/mini_accum/stats_overfit.py" \
  --summary_csv "$OUT_SUMMARY" \
  --out_md_append "$OUT_ROADMAP" \
  --windows WF_2021 WF_2022 WF_2023 WF_2024 WF_2025H1 \
  --candidate "DD15_RB1_H30_G200_BULL0"

echo "[OK] Pipeline KISS v1 completo → $OUT_ROADMAP"