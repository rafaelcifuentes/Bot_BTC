#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if ! scripts/ops/health.sh --strict; then
  echo "ROLLBACK: health FAIL => DRIVE_2023" >&2
  echo "LIVE" > deploy/ACTIVE.mode
  echo "DRIVE_2023" > deploy/ACTIVE.tag
fi
