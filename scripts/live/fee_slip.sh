#!/usr/bin/env bash
set -euo pipefail
# Prioridad: variable > archivo cache > por defecto 2/1
if [ -n "${LIVE_FEE_SLIP:-}" ]; then
  printf '%s\n' "$LIVE_FEE_SLIP"
  exit 0
fi
if [ -f deploy/live_fee_slip ]; then
  sed -n '1p' deploy/live_fee_slip
  exit 0
fi
echo "2/1"
