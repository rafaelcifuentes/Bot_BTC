#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
exec /bin/bash "$ROOT/weekly_runner.sh"
