#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/PycharmProjects/Bot_BTC"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
PKG="$ROOT/reports/mini_accum/reserve_pack.$TS.tgz"
LOG="$ROOT/logs/cron.log"
crontab -l > "$ROOT/reports/mini_accum/cron_snapshot.$TS.txt" || true
{
  echo "$(date -u +%FT%TZ) [INFO] weekly_pack: start"
  tar -czf "$PKG" -C "$ROOT" \
    reports/mini_accum/perf_seal.json \
    reports/mini_accum/robustness_seal.json \
    reports/mini_accum/code_seal.sha256 \
    reports/mini_accum/cron_snapshot.$TS.txt \
    signals/mini_accum/latest.json \
    logs/cron.log 2>/dev/null || true
  shasum -a 256 "$PKG" | tee "$PKG.sha256"
  echo "$(date -u +%FT%TZ) [INFO] weekly_pack: done $PKG"
} >> "$LOG" 2>&1
LEVEL=INFO CHAN=mini_accum /usr/bin/env python3 "$ROOT/scripts/mini_accum/notify.py" "Reserve pack semanal escrito: $(basename "$PKG")"
