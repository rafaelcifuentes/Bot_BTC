#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
ROOT="$HOME/PycharmProjects/Bot_BTC"
LOG="$ROOT/logs/cron.log"

# Log de inicio
echo "$(date -u +%FT%TZ) [INFO] daily_attest: start" >> "$LOG"

# Ejecuta attestation y captura rc SIN usar PIPESTATUS
/usr/bin/env bash "$ROOT/scripts/mini_accum/attest.sh" >> "$LOG" 2>&1
rc=$?

# Notifica según rc
if [ "$rc" -eq 0 ]; then
  LEVEL=INFO CHAN=mini_accum /usr/bin/env python3 "$ROOT/scripts/mini_accum/notify.py" "Attest diario OK"
else
  LEVEL=WARN CHAN=mini_accum /usr/bin/env python3 "$ROOT/scripts/mini_accum/notify.py" "Attest diario FAIL"
  exit 1
fi
