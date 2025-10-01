#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
cd "$ROOT/reports/mini_accum" 2>/dev/null || exit 0
# Packs
ls -1t reserve_pack.*.tgz 2>/dev/null | tail -n +13 | xargs -I{} rm -f -- "{}" || true
# Checksums
ls -1t reserve_pack.*.tgz.sha256 2>/dev/null | tail -n +13 | xargs -I{} rm -f -- "{}" || true
