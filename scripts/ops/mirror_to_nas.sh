#!/usr/bin/env bash
set -eo pipefail
set -u

# rsync 3.x si existe, si no el del sistema
if command -v /opt/homebrew/bin/rsync >/dev/null 2>&1; then
  RSYNC=/opt/homebrew/bin/rsync
elif command -v /usr/local/bin/rsync >/dev/null 2>&1; then
  RSYNC=/usr/local/bin/rsync
else
  RSYNC=$(command -v rsync)
fi

# Flags compatibles
RSYNC_FLAGS=(-a)
$RSYNC --help 2>&1 | grep -qi -- '--info=.*progress2' && RSYNC_FLAGS+=(--info=progress2) || RSYNC_FLAGS+=(--progress)
$RSYNC --help 2>&1 | grep -q  -- '--mkpath'           && RSYNC_FLAGS+=(--mkpath)
$RSYNC --help 2>&1 | grep -q  -- '--no-owner'         && RSYNC_FLAGS+=(--no-owner --no-group)

# Último snapshot si no viene STAMP del entorno
STAMP="${STAMP:-$(find backups -maxdepth 1 -mindepth 1 -type d -print | sort | tail -n1 | xargs -I{} basename {})}"

SRC_DIR="backups/${STAMP}"
DST_ROOT="${EXT_BACKUP_DIR:-/Volumes/sda1/rafaelcifuentes/Bot BTC}"
DST_DIR="${DST_ROOT}/${STAMP}"

# Guard-rails: nada de variables vacías
[ -n "${STAMP}" ]   || { echo "[ABORT] STAMP vacío";   exit 1; }
[ -d "${SRC_DIR}" ] || { echo "[ABORT] SRC_DIR no existe: '${SRC_DIR}'"; exit 1; }
[ -n "${DST_DIR}" ] || { echo "[ABORT] DST_DIR vacío"; exit 1; }

# Asegura destino (independiente de --mkpath)
mkdir -p "${DST_DIR}"

# Recibo ANTES del rsync para que viaje en el espejo
TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'External copy -> %s  @ %s\n' "${DST_DIR}" "${TS_UTC}" > "${SRC_DIR}/EXTERNAL_COPY.txt"

# Copia
set +e
"${RSYNC}" "${RSYNC_FLAGS[@]}" "${SRC_DIR}/" "${DST_DIR}/"
RC=$?
set -e

if [ $RC -ne 0 ]; then
  echo "[WARN] rsync falló (rc=$RC); no se copió al NAS"
  exit 0
fi

# Verificación del recibo en NAS + plan B
if [ -f "${DST_DIR}/EXTERNAL_COPY.txt" ]; then
  echo "[OK] Copia externa en: ${DST_DIR}"
else
  cp -f "${SRC_DIR}/EXTERNAL_COPY.txt" "${DST_DIR}/EXTERNAL_COPY.txt" && \
    echo "[OK] Recibo copiado por cp: ${DST_DIR}/EXTERNAL_COPY.txt" || \
    echo "[WARN] No pude grabar EXTERNAL_COPY.txt en el NAS"
fi
