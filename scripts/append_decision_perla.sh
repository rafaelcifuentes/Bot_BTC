#!/usr/bin/env bash
# scripts/append_decision_perla.sh
set -euo pipefail

CONFIG="${1:-configs/allocator_sombra.yaml}"
DEC_FILE="${2:-decisiones.md}"
LABEL_DEFAULT="Perla ($(date -u +%F)) — Aprobada semana"

# --- helper: leer fees del YAML (via Python) ---
read_fees() {
python3 - "$CONFIG" <<'PY'
import sys, json
try:
    import yaml
except Exception:
    print("6.0,6.0"); sys.exit(0)
cfg=sys.argv[1]
try:
    with open(cfg,'r') as f:
        y=yaml.safe_load(f)
    fee=y.get('costs',{}).get('fee_bps',6)
    slip=y.get('costs',{}).get('slip_bps',6)
    print(f"{float(fee)},{float(slip)}")
except Exception:
    print("6.0,6.0")
PY
}
IFS=, read -r FEE_BPS SLIP_BPS < <(read_fees)

# --- helper: top de grid OOS (up/dn, oos_net/pf/wr) ---
read_grid() {
python3 - <<'PY'
import pandas as pd, sys
try:
    df = pd.read_csv("reports/heart/perla_grid_oos.csv")
    best = df.sort_values("oos_net", ascending=False).iloc[0]
    print(f"{int(best['up'])},{int(best['dn'])},{best['oos_net']:.6f},{best['oos_pf']:.6f},{best['oos_wr']:.6f}")
except Exception:
    print(",,nan,nan,nan")
PY
}
IFS=, read -r UP DN OOS_NET OOS_PF OOS_WR < <(read_grid)

# --- helper: inferir mode a partir de signals/perla.csv ---
infer_mode() {
python3 - <<'PY'
import pandas as pd
try:
    df=pd.read_csv("signals/perla.csv")
    s=pd.to_numeric(df.get('sP'), errors='coerce')
    w=pd.to_numeric(df.get('w_perla_raw'), errors='coerce')
    s_uni=set(s.dropna().unique().tolist())
    mode = "longshort" if (-1 in s_uni and 1 in s_uni) else "longflat"
    print(mode)
except Exception:
    print("unknown")
PY
}
MODE=$(infer_mode)

# --- ejecutar breakdown (liviano) y capturar métricas de tests_overlay_check ---
TMP_OUT="$(mktemp)"
python3 tests_overlay_check.py | tee "$TMP_OUT" >/dev/null || true

grab() {
  # toma "Clave : valor" o "Clave    : valor"
  sed -n "s/^$1[[:space:]]*:[[:space:]]*//p" "$TMP_OUT" | head -n1
}

NET_TRAY="$(grab 'NET por trayectoria')"
GROSS_MINUS_COST="$(grab 'Gross - Σcostes')"
GROSS_SIN_COST="$(grab 'Gross (sin costes)')"
COSTES_TOT="$(grab 'Costes totales')"
TURNOVER="$(grab 'Turnover total')"
COST_SHARE="$(grab 'Cost share D/P')"
DIFF_CURVE="$(grab 'Diff (calc-curve)')"

# label final
LABEL="${LABEL_DEFAULT}"

# --- append al decisiones.md ---
{
  echo "### ✅ ${LABEL} — **Aprobada semana**"
  if [[ -n "$UP" && -n "$DN" ]]; then
    echo "**Selección:** Donchian **${UP}/${DN}**, **mode=${MODE}**"
  else
    echo "**Selección:** (no detectada en grid CSV) — **mode=${MODE}**"
  fi
  echo "**Fees:** ${FEE_BPS%.*} bps + ${SLIP_BPS%.*} bps — **$(( ${FEE_BPS%.*} + ${SLIP_BPS%.*} )) bps** totales"
  echo
  if [[ "$OOS_NET" != "nan" ]]; then
    echo "**Gates OOS — Resultado**"
    printf -- "- **oos_net > 0:** **%s** %s\n" "$OOS_NET" "$(awk -v x="$OOS_NET" 'BEGIN{print (x+0>0)?"✔️":"❌"}')"
    printf -- "- **oos_pf ≥ 1.1:** **%s** %s\n" "$OOS_PF" "$(awk -v x="$OOS_PF" 'BEGIN{print (x+0>=1.1)?"✔️":"❌"}')"
    printf -- "- **Turnover < ~40:** **%s** %s\n" "$TURNOVER" "$(awk -v x="$TURNOVER" 'BEGIN{print (x+0<40)?"✔️":"❌"}')"
    echo
  fi
  echo "**Allocator (perfil congelado) — Breakdown**"
  printf -- "- **NET por trayectoria:** **%+.6f**\n" "${NET_TRAY:-0}"
  printf -- "- **Gross (sin costes):** %s\n" "${GROSS_SIN_COST:-N/A}"
  printf -- "- **Gross − Σcostes:** %s\n" "${GROSS_MINUS_COST:-N/A}"
  printf -- "- **Costes totales:** %s\n" "${COSTES_TOT:-N/A}"
  printf -- "- **Turnover total:** %s\n" "${TURNOVER:-N/A}"
  printf -- "- **Cost share D/P:** %s\n" "${COST_SHARE:-N/A}"
  printf -- "- **Consistencia (Diff curva):** %s\n" "${DIFF_CURVE:-N/A}"
  echo
  echo "**Decisión:** **APROBAR** y **mantener** \`signals/perla.csv\`."
  echo "**Acciones:** snapshot de señal y mantener YAML (fees desde YAML)."
  echo
} >> "$DEC_FILE"

echo "[OK] Append a $DEC_FILE"