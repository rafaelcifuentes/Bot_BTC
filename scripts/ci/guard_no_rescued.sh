#!/usr/bin/env bash
set -euo pipefail
if grep -R "_rescued" -n configs scripts \
   --exclude="*.bak" \
   --exclude="guard_no_rescued.sh" ; then
  echo "ERROR: se detectó '_rescued' en configs/scripts" >&2
  exit 1
fi
