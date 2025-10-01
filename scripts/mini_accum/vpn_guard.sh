#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"

# ¿La ruta por defecto sale por utun? (típico de VPN)
iface="$(route -n get default 2>/dev/null | awk "/interface:/{print \$2}")"
if [[ "${iface:-}" == utun* ]]; then exit 0; fi

# ¿Hay algún utun* activo?
if ifconfig 2>/dev/null | grep -qE '^utun[0-9]:'; then exit 0; fi

# Sin VPN → notifica y sal con código 1 (cron omitirá el job si usas &&)
LEVEL=WARN CHAN=mini_accum /usr/bin/env python3 "$ROOT/scripts/mini_accum/notify.py" "VPN GUARD: túnel no activo — job omitido"
exit 1
