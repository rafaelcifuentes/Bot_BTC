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

# ---------------------------------------------
# [Optional] A/B semanal en sombra (condicional)
# Corre SOLO si existe un CSV candidato B.
#   - A = reports/mini_accum/walkforward/wf_summary_kpis.csv
#   - B se toma de:
#       1) $AB_CANDIDATE_CSV si está definido y existe
#       2) auto-descubrimiento si hay EXACTAMENTE uno en:
#          reports/mini_accum/experiments/*/wf_summary_kpis.csv
#   - Etiqueta = $AB_LABEL o el nombre del directorio del candidato.
#   - No falla el pipeline si el A/B falla o no hay candidato.
# ---------------------------------------------
(
  set -e
  ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
  A="$ROOT/reports/mini_accum/walkforward/wf_summary_kpis.csv"
  CAND=""
  LABEL="${AB_LABEL:-candidate}"

  # 1) Candidato explícito por variable de entorno
  if [ -n "${AB_CANDIDATE_CSV:-}" ] && [ -f "${AB_CANDIDATE_CSV}" ]; then
    CAND="${AB_CANDIDATE_CSV}"
    LABEL="${AB_LABEL:-$(basename "$(dirname "${CAND}")")}"
  fi

  # 2) Descubrimiento automático: exactamente un candidato en experiments
  if [ -z "$CAND" ]; then
    CANDS=$(find "$ROOT/reports/mini_accum/experiments" -maxdepth 2 -type f -name "wf_summary_kpis.csv" 2>/dev/null | sed -e '/^$/d')
    CNT=$(printf "%s\n" "$CANDS" | wc -l | tr -d ' ')
    if [ "$CNT" -eq 1 ]; then
      CAND=$(printf "%s\n" "$CANDS" | head -n 1)
      LABEL="${AB_LABEL:-$(basename "$(dirname "${CAND}")")}"
    fi
  fi

  if [ -n "$CAND" ] && [ -f "$CAND" ]; then
    echo "[A/B] Ejecutando en sombra → A=$(basename "$A") vs B=$(basename "$CAND") | label=$LABEL"
    if [ -x "$ROOT/scripts/mini_accum/ab_weekly.sh" ]; then
      "$ROOT/scripts/mini_accum/ab_weekly.sh" "$A" "$CAND" "$LABEL" || echo "[WARN] A/B falló (continuo pipeline)."
    else
      echo "[WARN] No existe o no es ejecutable: $ROOT/scripts/mini_accum/ab_weekly.sh (saltando A/B)."
    fi
  else
    echo "[A/B] No se encontró candidato; se omite. (Define AB_CANDIDATE_CSV o coloca exactamente un CSV en reports/mini_accum/experiments/*/wf_summary_kpis.csv)"
  fi
) || echo "[WARN] Bloque A/B condicional encontró un error (continuo pipeline)."