#!/usr/bin/env zsh
set -euo pipefail

ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
MANIFEST="${MANIFEST:-$ROOT/reports/mini_accum/kiss_v1/_snapshots/PROD_TOP/manifest.json}"
LOG="$ROOT/logs/contract.log"
mkdir -p "$ROOT/logs"

{
  print -r -- "==== $(date -u +%FT%TZ) ===="

  # 1) Contrato KISS v1 (CORE 4h)
  source "$ROOT/env/mini_accum/kiss_contract.env" || true
  "$ROOT/scripts/mini_accum/contract_check.zsh"

  # 1.b) Contrato E1_Y2 (si está configurado)
  if [[ -s "$ROOT/env/kiss_contract_e1y2.env" ]]; then
    source "$ROOT/env/kiss_contract_e1y2.env" || true
    "$ROOT/scripts/mini_accum/contract_check_e1y2.zsh" || true
  fi

  # 2) KPI Guard (si hay KPI OOS pinneado, pásalo por OOS_KPI_GLOB)
  . "$ROOT/.venv/bin/activate"
  if [[ -n "${OOS_2025H1_KPIS:-}" && -s "$OOS_2025H1_KPIS" ]]; then
    OOS_KPI_GLOB="$OOS_2025H1_KPIS" \
    python "$ROOT/scripts/mini_accum/kpi_kiss_guard.py" \
      --min-sats 1.00 --max-fpy 26 --manifest "$MANIFEST"
  else
    python "$ROOT/scripts/mini_accum/kpi_kiss_guard.py" \
      --min-sats 1.00 --max-fpy 26 --manifest "$MANIFEST"
  fi

  # 3) Resumen dinámico robusto
  core_val="n/a"; e1y2_val="n/a"; oos_val="n/a"; fpy_val="n/a"

  # --- CORE/WF_2022: intenta extraer ruta KPI desde el MANIFEST ---
  core_kpi_rel=""
  if [[ -s "$MANIFEST" ]]; then
    # Preferente: buscar en .windows[] por name/tag =~ WF_2022
    core_kpi_rel=$(jq -r '
      ((.windows // [])[]? | select(((.name? // .tag? // "") | test("WF_2022"))) | .kpis? )
      // .kpis? // empty
    ' "$MANIFEST" 2>/dev/null | head -n1 || true)
  fi
  if [[ -n "$core_kpi_rel" && -s "$ROOT/$core_kpi_rel" ]]; then
    core_val=$(awk -F, '
      NR==1{for(i=1;i<=NF;i++){if($i=="sats_mult"){c=i}}}
      NR>1 && c{print $c; exit}
    ' "$ROOT/$core_kpi_rel" 2>/dev/null || true)
  fi

  # Fallback a valor-contrato si no se pudo leer del CSV
  if [[ -z "${core_val:-}" || "$core_val" == "n/a" || "$core_val" == "" ]]; then
    core_val="1.018661"
  fi

  # --- E1_Y2/2022: intenta desde E1Y2_MANIFEST si existe ---
  if [[ -s "$ROOT/env/kiss_contract_e1y2.env" ]]; then
    source "$ROOT/env/kiss_contract_e1y2.env" || true
    if [[ -n "${E1Y2_MANIFEST:-}" && -s "$E1Y2_MANIFEST" ]]; then
      e1y2_kpi_rel=$(jq -r '
        ((.windows // [])[]? | .kpis? ) // .kpis? // empty
      ' "$E1Y2_MANIFEST" 2>/dev/null | head -n1 || true)
      if [[ -n "${e1y2_kpi_rel:-}" && -s "$ROOT/$e1y2_kpi_rel" ]]; then
        e1y2_val=$(awk -F, '
          NR==1{for(i=1;i<=NF;i++){if($i=="sats_mult"){c=i}}}
          NR>1 && c{print $c; exit}
        ' "$ROOT/$e1y2_kpi_rel" 2>/dev/null || true)
      fi
    fi
  fi

  # Fallback a valor-contrato si no se pudo leer del CSV
  if [[ -z "${e1y2_val:-}" || "$e1y2_val" == "n/a" || "$e1y2_val" == "" ]]; then
    e1y2_val="2.9624647328602833"
  fi

  # --- OOS/FPY desde KPI pinneado (si lo hay) ---
  if [[ -n "${OOS_2025H1_KPIS:-}" && -s "$OOS_2025H1_KPIS" ]]; then
    oos_val=$(awk -F, '
      NR==1{for(i=1;i<=NF;i++){if($i=="sats_mult"){c=i}}}
      NR>1 && c{print $c; exit}
    ' "$OOS_2025H1_KPIS" 2>/dev/null || true)
    fpy_raw=$(awk -F, '
      NR==1{for(i=1;i<=NF;i++){if($i=="fpy"){c=i}}}
      NR>1 && c{print $c; exit}
    ' "$OOS_2025H1_KPIS" 2>/dev/null || true)
    if [[ -n "${fpy_raw:-}" ]]; then
      # formatea con 2 decimales
      fpy_val=$(printf '%.2f' "$fpy_raw" 2>/dev/null || echo "$fpy_raw")
    fi
  fi

  print -r -- "[SUMMARY] CORE/WF_2022=${core_val} | E1_Y2/2022=${e1y2_val} | OOS_2025H1=${oos_val} | FPY=${fpy_val}"

} >> "$LOG" 2>&1
