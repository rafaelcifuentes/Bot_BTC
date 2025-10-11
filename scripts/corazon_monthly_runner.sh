#!/bin/bash
# 🧭 scripts/corazon_monthly_runner.sh
# Uso: ./scripts/corazon_monthly_runner.sh

set -e

ROOT="$(dirname "$0")/.."
FREEZE_DATE="2025-09-08"  # última fecha congelada válida
STATUS_DATE="$(date -u +%F)"
STATUS_MONTH="$(date -u +%Y-%m)"
MD_PATH="$ROOT/reports/heart/monthly_status_${STATUS_MONTH}.md"
CSV_PATH="$ROOT/reports/heart/monthly_status_${STATUS_MONTH}.csv"
BASE="$ROOT/reports/heart/kpis_diamante_btc_costes_freeze_${FREEZE_DATE}_bars.csv"

# Verifica existencia
if [ ! -f "$BASE" ]; then
  echo "❌ ERROR: No se encontró archivo base: $BASE"
  exit 1
fi

# Crea CSV si no existe y escribe encabezado
if [ ! -f "$CSV_PATH" ]; then
  echo "📄 Creando nuevo archivo: $CSV_PATH"
  echo "module,date,freeze,xi_star,pass_fail,pf_base,pf_overlay,mdd_base,mdd_overlay,vol_base,vol_overlay,net_base,net_overlay" > "$CSV_PATH"
fi

# Agrega línea CSV
echo "➕ Agregando fila a $CSV_PATH"
python - <<PY >> "$CSV_PATH"
import pandas as pd
FREEZE = "$FREEZE_DATE"
STATUS = "$STATUS_DATE"
df = pd.read_csv("$BASE").iloc[0]

mdd_ratio = abs(df["mdd_base"]) / max(1e-12, abs(df["mdd_overlay"]))
vol_ratio = df["vol_base"] / max(1e-12, df["vol_overlay"])
xi_star = min(mdd_ratio, vol_ratio) * 0.85

status = "PASS" if (
    df["pf_overlay"] >= 0.9 * df["pf_base"]
    and abs(df["mdd_overlay"]) <= abs(df["mdd_base"])
    and df["vol_overlay"] <= df["vol_base"]
) else "FAIL"

row = {
    "module": "corazon",
    "date": STATUS,
    "freeze": FREEZE,
    "xi_star": f"{xi_star:.4f}",
    "pass_fail": status,
    "pf_base": f"{df['pf_base']:.4f}",
    "pf_overlay": f"{df['pf_overlay']:.4f}",
    "mdd_base": f"{df['mdd_base']:.6f}",
    "mdd_overlay": f"{df['mdd_overlay']:.6f}",
    "vol_base": f"{df['vol_base']:.6f}",
    "vol_overlay": f"{df['vol_overlay']:.6f}",
    "net_base": f"{df['net_base']:.5f}",
    "net_overlay": f"{df['net_overlay']:.5f}",
}

print(",".join(str(v) for v in row.values()))
PY

# Crea MD si no existe
if [ ! -f "$MD_PATH" ]; then
  echo "📄 Generando nuevo archivo MD: $MD_PATH"
  cat > "$MD_PATH" <<EOF
# 🧭 Corazón – Monthly Status Freeze (${STATUS_MONTH})

**Fecha:** ${STATUS_DATE}
**Último FREEZE disponible:** ${FREEZE_DATE}
**Última señal procesada:** signals/diamante.csv (freeze)

---

## 🔒 Estado congelado

| Módulo    | Estado     | Último freeze    | ξ*     | PASS/FAIL | Notas                     |
|-----------|------------|------------------|--------|-----------|----------------------------|
| Corazón   | Congelado  | ${FREEZE_DATE}   | …      | …         | KPIs aún consistentes      |
| Diamante  | Congelado  | ${FREEZE_DATE}   | —      | —         | No se ha reactivado        |
| Perla     | Congelado  | ${FREEZE_DATE}   | —      | —         | runtime.perla_enabled: false |

---

## 📊 KPIs vs baseline (último FREEZE)

**Archivo base:** kpis_diamante_btc_costes_freeze_${FREEZE_DATE}_bars.csv
**KPIs overlay:** diamante_overlay_diamante_btc_costes_freeze_${FREEZE_DATE}_bars.csv

| Métrica         | Base       | Overlay    | Δ %      | Mejor |
|------------------|------------|------------|----------|--------|
| Profit Factor    | …          | …          | …        | …     |
| Max Drawdown     | …          | …          | …        | …     |
| Volatilidad σ    | …          | …          | …        | …     |
| Net Profit       | …          | …          | …        | …     |

---

## ✅ Veredicto

Corazón se mantiene congelado. Este snapshot sirve como control de consistencia mensual.

---

## 📁 Artefactos clave

- reports/heart/kpis_diamante_btc_costes_freeze_${FREEZE_DATE}_bars.csv
- reports/heart/diamante_overlay_diamante_btc_costes_freeze_${FREEZE_DATE}_bars.csv
- corazon/daily_xi.csv
EOF
fi

# Abre ambos en editor
if command -v code &>/dev/null; then
  code "$MD_PATH" "$CSV_PATH"
else
  echo "✏️ Edita manualmente con nano o tu editor preferido:"
  echo "nano $MD_PATH"
  echo "nano $CSV_PATH"
fi
