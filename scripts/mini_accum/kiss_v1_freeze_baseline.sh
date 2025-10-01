# Guarda como: scripts/mini_accum/kiss_v1_freeze_baseline.sh
#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="$REPO_DIR/configs/mini_accum/kiss_v1.yaml"
ROADMAP="$REPO_DIR/reports/mini_accum/walkforward/Roadmap_PDCA.md"
CAND="DD15_RB1_H30_G200_BULL0"

TS="$(date +%Y%m%d_%H%M)"
VERS="KISSv1_BASE_${TS}_provisional"
BACKUP="$REPO_DIR/configs/mini_accum/kiss_v1_BASE_${TS}.yaml"

# 1) Backup del YAML actual
cp "$CFG" "$BACKUP"

# 2) Intenta anotar version/frozen con PyYAML; si falla, agrega claves al final
python - <<'PY' "$CFG" "$VERS"
import sys, yaml, datetime, io
cfg_path, vers = sys.argv[1], sys.argv[2]
try:
    with open(cfg_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    data['version'] = vers
    data['frozen'] = True
    notes = data.get('notes') or []
    notes.append({'baseline': 'provisional', 'frozen_at_utc': datetime.datetime.utcnow().isoformat()})
    data['notes'] = notes
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print("[OK] YAML actualizado con PyYAML.")
except Exception as e:
    print("[WARN] PyYAML no disponible o error:", e)
    # Fallback: anexar claves al final (puede duplicar si ya existen)
    with open(cfg_path, 'a', encoding='utf-8') as f:
        f.write(f'\nversion: "{vers}"\nfrozen: true\n')
    print("[OK] Fallback: claves agregadas al final del YAML.")
PY

# 3) Añade/actualiza sección en Roadmap_PDCA.md
mkdir -p "$(dirname "$ROADMAP")"
{
  echo ""
  echo "## Baseline & Lock-in"
  echo "- Versión: \`$VERS\`"
  echo "- Candidato: \`$CAND\`"
  echo "- Estado: **Provisional** (quitar cuando DSR>0 y PBO ≤ 0.25, ideal ≤ 0.20, y 2 semanas OOS sin regresión)."
  echo "- Copia YAML: \`configs/mini_accum/kiss_v1_BASE_${TS}.yaml\`"
} >> "$ROADMAP"

echo "[OK] Baseline congelado y documentado:"
echo "  - $CFG"
echo "  - $BACKUP"
echo "  - $ROADMAP"