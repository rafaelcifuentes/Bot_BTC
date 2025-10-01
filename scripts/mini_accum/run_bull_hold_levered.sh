#!/bin/bash

set -euo pipefail

# -- Pick Python from venv if present, else system python3/python
if [ -x "./.venv/bin/python" ]; then
  PY="./.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  echo "[ERR] No se encontró intérprete de Python. Activa .venv o instala python3." >&2
  exit 1
fi

: "${PATH:=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin}"

BASE="configs/mini_accum/kiss_v1.yaml"
OVER="configs/mini_accum/overlays/bull_hold_levered.yaml"
MERGED="configs/mini_accum/kiss_v1_bullhold_lev.yaml"
WINF="configs/mini_accum/windows_walkforward.csv"
WINF_LEV="configs/mini_accum/windows_walkforward_levered.csv"

# 0) sello temporal
STAMP=$(date +"%Y%m%d_%H%M%S")
TMP_KPI_DIR="reports/mini_accum/tmp_kpis_lev_${STAMP}"
mkdir -p "$TMP_KPI_DIR"

#
# 1) merge (Python, sin tocar baseline)
"$PY" - <<'PY' "$BASE" "$OVER" "$MERGED"
import yaml, sys
from pathlib import Path
from collections.abc import Mapping
base, over, out = map(Path, sys.argv[1:4])
def deep_merge(a,b):
    for k,v in (b or {}).items():
        if isinstance(v,dict) and isinstance(a.get(k),dict):
            deep_merge(a[k],v)
        else:
            a[k]=v
    return a
A=yaml.safe_load(base.read_text(encoding="utf-8")) or {}
B=yaml.safe_load(over.read_text(encoding="utf-8")) or {}
M=deep_merge(A,B)
out.write_text(yaml.safe_dump(M, sort_keys=False, allow_unicode=True), encoding="utf-8")
print("[OK] merged ->", out)
PY

# 2) backups y swap
cp -p "$BASE" "${BASE}.BAK"
[ -f "$WINF" ] && cp -p "$WINF" "${WINF}.BAK" || true
cp -f "$MERGED" "$BASE"
cp -f "$WINF_LEV" "$WINF"

# 3) run pipeline
export FORCE_REBUILD=1
# Asegura que el python de la venv esté visible para scripts hijos que llamen 'python'
export PATH="$(pwd)/.venv/bin:$PATH"
export PY="$PY"
scripts/mini_accum/kiss_v1_wf_pipeline.sh

# 4) copiar KPI recientes (~90 min) de forma portable en macOS
NOW=$(date +%s)
CUTOFF=$((NOW - 90*60))
FOUND=0
while IFS= read -r path; do
  cp "$path" "$TMP_KPI_DIR"/
  FOUND=1
done < <(find reports/mini_accum/kiss_v1 -name "*kpis__*.csv" -exec stat -f "%m %N" {} \; | awk -v C="$CUTOFF" '$1>=C {print substr($0, index($0,$2))}')

if [ "$FOUND" -eq 0 ]; then
  echo "[WARN] No KPI nuevos en ~90 min; copiando por fecha en nombre (hoy)."
  DATE=$(date +%Y%m%d)
  cp reports/mini_accum/kiss_v1/*${DATE}_*kpis__*.csv "$TMP_KPI_DIR"/ 2>/dev/null || true
fi


# 4.1) copia garantizada de ventanas históricas claves (2021/2022)
# (evita que queden fuera por no ser "recientes" en mtime)
# Nota: soporta nombres con 'WF_WF_YYYY' y 'WF_YYYY'
for pat in "*kpis__WF_WF_2021*.csv" "*kpis__WF_WF_2022*.csv" "*kpis__WF_2021*.csv" "*kpis__WF_2022*.csv"; do
  cp reports/mini_accum/kiss_v1/$pat "$TMP_KPI_DIR"/ 2>/dev/null || true
done
# Diagnóstico: cuenta cuántos KPI 2021/2022 quedaron en TMP
WF_CNT=$(ls -1 "$TMP_KPI_DIR"/*kpis__WF_202[12]*.csv 2>/dev/null | wc -l | tr -d ' ')
echo "[INFO] KPIs 2021/2022 en tmp: ${WF_CNT:-0}"

# 4.2) normaliza nombres 'WF_WF_YYYY' -> 'WF_YYYY' (robusto en zsh/bash)
"$PY" - <<'PY' "$TMP_KPI_DIR"
import sys, pathlib
d = pathlib.Path(sys.argv[1])
n=0
for p in d.glob("*kpis__WF_WF_*"):
    new = p.with_name(p.name.replace("WF_WF_", "WF_"))
    if new != p:
        try:
            p.rename(new)
            n+=1
        except Exception as e:
            print(f"[WARN] No pude renombrar {p.name} -> {new.name}: {e}")
print(f"[OK] Normalizados {n} archivos 'WF_WF_' en {d}")
PY

# 5) consolidación aislada (no pisa tu summary final)
"$PY" tools/mini_accum/wf_consolidate.py \
  --kpis_glob "${TMP_KPI_DIR}/*kpis__*.csv" \
  --out_summary "reports/mini_accum/walkforward/wf_summary_kpis__LEV.csv" \
  --out_best    "reports/mini_accum/walkforward/wf_best_by_window__LEV.csv" \
  --out_md      "reports/mini_accum/walkforward/Roadmap_PDCA.md" \
  --candidate   "DD15_RB1_H30_G200_BULL0"

echo "[OK] Consolidado LEV -> reports/mini_accum/walkforward/wf_summary_kpis__LEV.csv"

# 6) restaurar baseline/ventanas
mv -f "${BASE}.BAK" "$BASE"
if [ -f "${WINF}.BAK" ]; then mv -f "${WINF}.BAK" "$WINF"; else rm -f "$WINF"; fi

echo "[DONE] BULL_HOLD LEVERED listo. KPI aislados en: $TMP_KPI_DIR"
