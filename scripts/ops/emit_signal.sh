#!/usr/bin/env bash
set -euo pipefail
: "${ENABLE_SIGNALS:=0}"
: "${SIGNALS_FILE:=signals/stream.csv}"

[ "$ENABLE_SIGNALS" = "1" ] || exit 0
mkdir -p "$(dirname "$SIGNALS_FILE")"
[ -s "$SIGNALS_FILE" ] || echo "ts_utc,tag,action,price,reason" > "$SIGNALS_FILE"

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TAG="${1:-UNKNOWN}"
ACT="${2:-HOLD}"
PRICE="${3:-nan}"
REASON="${4:-n/a}"
printf "%s,%s,%s,%s,%s\n" "$TS" "$TAG" "$ACT" "$PRICE" "$REASON" >> "$SIGNALS_FILE"
