#!/usr/bin/env bash
set -euo pipefail

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BK="backups/$STAMP"
mkdir -p "$BK"

# --- helpers ---
csum(){ if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" > "$1.sha256"; else sha256sum "$1" > "$1.sha256"; fi; }
have(){ [ -e "$1" ]; }

echo "[INFO] Snapshot -> $BK"

# 1) Identidad git
{ git rev-parse HEAD; git describe --tags --always || true; } > "$BK/GIT_HEAD.txt"

# 2) Código (solo tracked)
git archive --format=tar --prefix=Bot_BTC_code/ HEAD | gzip > "$BK/code_HEAD.tgz"
csum "$BK/code_HEAD.tgz"

# 3) Presets + deploy + docs canónicas
LIST=()
for p in presets configs mk/deploy.mk mk/stress.mk scripts/ops/check_health.sh \
         deploy/ACTIVE.tag deploy/live_fee_slip README.md docs/mini_accum/ONE_PAGER.md; do
  have "$p" && LIST+=("$p")
done
if ((${#LIST[@]})); then
  tar czf "$BK/presets_deploy.tgz" "${LIST[@]}"
  csum "$BK/presets_deploy.tgz"
fi

# 4) Freezes (fuente de verdad) + artefactos operativos si existen
OPSL=()
have reports/mini_accum/_freezes && OPSL+=(reports/mini_accum/_freezes)
for p in reports/mini_accum/STRESS_HEAD.txt reports/mini_accum/STRESS_PROB.txt \
         reports/mini_accum/GATE_STATUS.txt reports/mini_accum/COMPARISON/cost_sensivity.md \
         reports/mini_accum/ONE_PAGER.md; do
  have "$p" && OPSL+=("$p")
done
if ((${#OPSL[@]})); then
  tar czf "$BK/freezes_ops.tgz" "${OPSL[@]}"
  csum "$BK/freezes_ops.tgz"
fi

# 5) Datos mínimos reproducibles (opcional)
if have data/BTC-USD_4h.csv; then
  tar czf "$BK/data_min.tgz" data/BTC-USD_4h.csv
  csum "$BK/data_min.tgz"
fi

# 6) Foto del gate
{
  printf 'FeeSlip: '; cat deploy/live_fee_slip 2>/dev/null || echo 'N/A'
  printf 'ACTIVE.tag: '; cat deploy/ACTIVE.tag 2>/dev/null || echo 'N/A'
  printf 'ACTIVE.mode: '; cat deploy/ACTIVE.mode 2>/dev/null || echo 'N/A'
} > "$BK/GATE_SNAPSHOT.txt"

# 7) Copia externa (si EXT_BACKUP_DIR está definida y es escribible)
if [[ -n "${EXT_BACKUP_DIR:-}" ]]; then
  DEST="${EXT_BACKUP_DIR%/}/$STAMP"
  mkdir -p "$DEST" 2>/dev/null || true
  if rsync -ah --info=progress2 "$BK"/ "$DEST"/; then
    echo "[OK] Copia externa en: $DEST" | tee "$BK/EXTERNAL_COPY.txt"
  else
    echo "[WARN] No pude copiar a EXT_BACKUP_DIR='$EXT_BACKUP_DIR' (continuo con backup local)" | tee "$BK/EXTERNAL_COPY.txt"
  fi
fi

echo "[DONE] Backup local en: $BK"
ls -la "$BK"
