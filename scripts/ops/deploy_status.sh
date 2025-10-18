#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="$(cat "$ROOT/deploy/ACTIVE.tag" 2>/dev/null || echo UNKNOWN)"
MODE="$(cat "$ROOT/deploy/ACTIVE.mode" 2>/dev/null || echo UNKNOWN)"
RUNTIME="$ROOT/deploy/live_fee_slip"

fee="?"
slip="?"

if [[ -s "$RUNTIME" ]]; then
  fee=$(awk -F= '$1=="fee_bps_per_side"{print $2}' "$RUNTIME")
  [[ -z "$fee" ]] && fee=$(awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="fee_bps_per_side")c=i} NR==2{print $c}' "$RUNTIME")
  slip=$(awk -F= '$1=="slip_bps_per_side"{print $2}' "$RUNTIME")
  [[ -z "$slip" ]] && slip=$(awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="slip_bps_per_side")c=i} NR==2{print $c}' "$RUNTIME")
fi

echo "[STATUS] MODE=$MODE TAG=$TAG | fee_bps_per_side=${fee} slip_bps_per_side=${slip}"
