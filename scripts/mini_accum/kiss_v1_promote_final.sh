#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="$REPO_DIR/configs/mini_accum/kiss_v1.yaml"
ROADMAP="$REPO_DIR/reports/mini_accum/walkforward/Roadmap_PDCA.md"
PROGRESO="$REPO_DIR/docs/Mini_accum/Progreso.md"  # si no existe, se ignora
CAND="DD15_RB1_H30_G200_BULL0"

TS="$(date +%Y%m%d_%H%M)"
VERS_FINAL="KISSv1_BASE_${TS}_final"

# 1) Actualiza YAML (status final + versión)
python - <<'PY' "$CFG" "$VERS_FINAL"
import sys, yaml, datetime, io, os
cfg_path, vers = sys.argv[1], sys.argv[2]
with open(cfg_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f) or {}
data['version'] = vers
data['frozen']  = True
data['status']  = 'final'
notes = data.get('notes') or []
notes.append({'baseline': 'final', 'locked_at_utc': datetime.datetime.utcnow().isoformat()})
data['notes'] = notes
with open(cfg_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
print("[OK] YAML actualizado:", cfg_path)
PY

# 2) Marca "Definitivo" en el Roadmap y añade badge de versión
mkdir -p "$(dirname "$ROADMAP")"
# Si la línea de Estado existe como Provisional, cámbiala; si no, solo añade bloque
if grep -q "Baseline & Lock-in" "$ROADMAP" 2>/dev/null; then
  # BSD sed (macOS)
  sed -i '' 's/Estado: \*\*Provisional\*\*/Estado: **Definitivo**/g' "$ROADMAP" || true
fi
{
  echo ""
  echo "### Baseline & Lock-in (Final)"
  echo "- Versión: \`$VERS_FINAL\`"
  echo "- Candidato: \`$CAND\`"
  echo "- Estado: **Definitivo**"
} >> "$ROADMAP"

# 3) (Opcional) deja constancia también en Progreso.md si existe
if [ -f "$PROGRESO" ]; then
  {
    echo ""
    echo "### Baseline KISS v1 (Final) — $VERS_FINAL"
    echo "- Candidato: \`$CAND\`"
    echo "- Estado: **Definitivo** (PBO≈0.107, DSR positivo)"
  } >> "$PROGRESO" || true
fi

echo "[OK] Baseline promovido a definitivo."
echo "  - $CFG"
echo "  - $ROADMAP"
[ -f "$PROGRESO" ] && echo "  - $PROGRESO" || true