#!/usr/bin/env bash
set -euo pipefail

HOST="192.168.0.1"
SHARE="sda1"
MNT="/Volumes/sda1"

# montar si no está montado
if ! mount | grep -q "on ${MNT} "; then
  # Asume que el directorio ya existe (créalo con sudo fuera de cron)
  /sbin/mount_smbfs "//guest:@${HOST}/${SHARE}" "${MNT}" || /usr/sbin/mount_smbfs "//guest:@${HOST}/${SHARE}" "${MNT}"
fi

# Verificación de escritura del destino externo (usa EXT_BACKUP_DIR si viene del entorno o default)
DEST="${EXT_BACKUP_DIR:-/Volumes/sda1/rafaelcifuentes/Bot BTC}"
mkdir -p "$DEST"
touch "$DEST/.rw_test" && rm -f "$DEST/.rw_test"
echo "[OK] SMB listo y escribible: $DEST"
