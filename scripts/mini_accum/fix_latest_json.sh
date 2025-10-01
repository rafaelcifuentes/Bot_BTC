#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
SIG="$ROOT/signals/mini_accum/latest.json"
mkdir -p "$(dirname "$SIG")"
if [ ! -s "$SIG" ]; then
  printf '{ "ts_utc":"%s","version":"KISSv1_BASE_UNKNOWN","health":"ok","guards":{},"position_pct_btc":0.0 }\n' "$(date -u +%FT%TZ)" > "$SIG"
  echo "$(date -u +%FT%TZ) [INFO] fix_latest: created minimal latest.json"
else
  tmp="$(mktemp)"
  jq '.ts_utc //= (now|todate)
      | .version //= "KISSv1_BASE_UNKNOWN"
      | .health  //= "ok"
      | .guards  //= {}
      | .position_pct_btc //= 0.0' "$SIG" > "$tmp" && mv "$tmp" "$SIG"
  echo "$(date -u +%FT%TZ) [INFO] fix_latest: normalized latest.json"
fi
