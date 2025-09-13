#!/usr/bin/env bash
set -euo pipefail

SUF="v3p3N2g-F2-H1_FZ-oos24H1"
RPT="reports/mini_accum"
CFG="configs/baselines/F2_H1_FZ.yaml"
HASH="$(git rev-parse --short HEAD 2>/dev/null || echo NO_GIT)"
STAMP="$(date +%Y%m%d_%H%M)"
OUT="artifacts/H1_FZ_${STAMP}_${HASH}"

eq="$(ls -t ${RPT}/*equity__${SUF}.csv | head -1)"
kp="$(ls -t ${RPT}/*kpis__${SUF}.csv   | head -1)"
sm="$(ls -t ${RPT}/*summary__${SUF}.md | head -1)"
fl="$(ls -t ${RPT}/*flips__${SUF}.csv  | head -1)"

mkdir -p "${OUT}"
cp "${eq}" "${OUT}/H1_FZ_equity.csv"
cp "${kp}" "${OUT}/H1_FZ_kpis.csv"
cp "${sm}" "${OUT}/H1_FZ_summary.md"
cp "${fl}" "${OUT}/H1_FZ_flips.csv"
cp "${CFG}" "${OUT}/F2_H1_FZ.yaml"

# Manifest con métricas clave
NET=$(awk -F, 'NR==2{print $1}' "${OUT}/H1_FZ_kpis.csv")
MDDM=$(awk -F, 'NR==2{print $2}' "${OUT}/H1_FZ_kpis.csv")
MDDH=$(awk -F, 'NR==2{print $3}' "${OUT}/H1_FZ_kpis.csv")
MDDR=$(awk -F, 'NR==2{print $4}' "${OUT}/H1_FZ_kpis.csv")
FLIP=$(awk -F, 'NR==2{print $5}' "${OUT}/H1_FZ_kpis.csv")

cat > "${OUT}/MANIFEST.json" <<JSON
{
  "artifact": "H1_FZ",
  "commit": "${HASH}",
  "config": "${CFG}",
  "report_suffix": "${SUF}",
  "window": "2024H1",
  "metrics": {
    "net_btc_ratio": ${NET:-null},
    "mdd_model_usd": ${MDDM:-null},
    "mdd_hodl_usd": ${MDDH:-null},
    "mdd_vs_hodl_ratio": ${MDDR:-null},
    "flips_total": ${FLIP:-null}
  },
  "files": [
    "H1_FZ_equity.csv",
    "H1_FZ_kpis.csv",
    "H1_FZ_summary.md",
    "H1_FZ_flips.csv",
    "F2_H1_FZ.yaml"
  ]
}
JSON

# README mínimo
cat > "${OUT}/README.md" <<'MD'
# Baseline H1_FZ (FREEZE)

- Ventana: 2024-01-01 → 2024-06-30 (H1'24 holdout)
- Señal: EMA(13/55), **exit_active desactivada**
- Anti-whipsaw: dwell_min=76, sin pausa, sin ATR regime
- Presupuesto: hard_per_year=26

## Motivación
Replicar/estabilizar el desempeño del TOP en H1 con menor whipsaw. Resultados: PASS y mejora vs TOP.

## Archivos
- `H1_FZ_equity.csv`, `H1_FZ_kpis.csv`, `H1_FZ_summary.md`, `H1_FZ_flips.csv`, `F2_H1_FZ.yaml`
- Ver `MANIFEST.json` para hash y métricas clave.
MD

echo "[OK] Artifact listo en ${OUT}"
