#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
VENV_BIN="${VENV_BIN:-$REPO_ROOT/.venv/bin}"
DICT="$VENV_BIN/mini-accum-dictamen"
PY="$VENV_BIN/python"

OUT_TSV="${OUT_TSV:-/tmp/d.tsv}"
REPORTS_DIR="${REPORTS_DIR:-$REPO_ROOT/reports/mini_accum}"

echo "[dictamen] -> $OUT_TSV"
"$DICT" --reports-dir "$REPORTS_DIR" --out "$OUT_TSV" --format tsv --quiet

echo -e "suffix\twindow\tnetBTC\tmdd_vs_HODL\tflips\tfpy\tPASS\tFAIL"
awk -F'\t' 'NR>1 {printf "%s\t%s\t%.3f\t%.3f\t%d\t%.1f\t%s\t%s\n",$1,$2,$3,$6,$7,$8,$9,$10}' "$OUT_TSV" | \
  column -t -s $'\t' | sed -n '1,80p'

echo ""
echo "[flip stats] (min4h / weeks>1) para flips recientes en H1"
# Escoge algunos flips CSV recientes (puedes ajustar el pattern a tu sufijo preferido)
ls -1 "$REPORTS_DIR"/*flips__*oos24H1.csv 2>/dev/null | tail -n 10 | while read -r f; do
  "$PY" "$REPO_ROOT/scripts/mini_accum/flip_stats.py" "$f" || true
done
