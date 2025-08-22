#!/usr/bin/env bash
set -euo pipefail

REPORTS_DIR="${1:-reports}"
HORIZON="${2:-60}"

echo "[INFO] Tracker: reports_dir=${REPORTS_DIR} | horizon=${HORIZON}"

# Crear scripts/ si no existe
mkdir -p scripts

# Ejecutar tracker
python scripts/diamante_tracker.py \
  --reports_dir "${REPORTS_DIR}" \
  --horizon "${HORIZON}" \
  --out_csv "${REPORTS_DIR}/diamante_tracker.csv" \
  --out_csv_best "${REPORTS_DIR}/diamante_tracker_best.csv" \
  --out_md "${REPORTS_DIR}/diamante_tracker.md"