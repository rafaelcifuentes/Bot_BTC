#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
LOG="$ROOT/logs/cron.log"
echo "$(date -u +%FT%TZ) [INFO] cron: weekly_runner tick" >> "$LOG"

# VPN guard
if ! "$ROOT/scripts/mini_accum/vpn_guard.sh"; then
  echo "$(date -u +%FT%TZ) [WARN] cron: vpn_guard blocked weekly_runner" >> "$LOG"
  exit 0
fi

# lock
LOCK="$ROOT/logs/weekly_runner.lock"
if ! ( set -o noclobber; echo $$ > "$LOCK" ) 2>/dev/null; then
  echo "$(date -u +%FT%TZ) [WARN] cron: weekly_runner lock busy — skipping" >> "$LOG"
  exit 0
fi
trap 'rm -f "$LOCK"' EXIT

# pre-run: garantizar claves
"$ROOT/scripts/mini_accum/fix_latest_json.sh" || true
echo "$(date -u +%FT%TZ) [INFO] cron: wrote minimal/normalized latest.json (pre-run)" >> "$LOG"

# run
/usr/bin/env caffeinate -dims -t 2100 /bin/bash "$ROOT/weekly_runner.sh"
rc=$?

# post-run: garantizar claves (por si el writer final omitió)
"$ROOT/scripts/mini_accum/fix_latest_json.sh" || true

echo "$(date -u +%FT%TZ) [INFO] cron: weekly_runner done rc=$rc" >> "$LOG"
exit $rc
