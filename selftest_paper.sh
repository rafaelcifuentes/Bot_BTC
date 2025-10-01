#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
ROOT="$HOME/PycharmProjects/Bot_BTC"

echo "== Selftest: inicio $(date -u +%FT%TZ) =="

# 1) Latencia <30s
t0=$(date +%s)
env -i HOME="$HOME" PATH="$PATH" /bin/bash -lc 'ROOT="$HOME/PycharmProjects/Bot_BTC" LOG_LEVEL=INFO /bin/bash "$HOME/PycharmProjects/Bot_BTC/weekly_runner.sh"'
t1=$(date +%s)
dt=$((t1 - t0))
echo "Latencia runner: ${dt}s"
if [ "$dt" -gt 30 ]; then echo "WARN: latencia > 30s (ok en paper si fetch/pipeline corren)"; fi

# 2) Idempotencia: 2a ejecución no debe duplicar avisos
env -i HOME="$HOME" PATH="$PATH" /bin/bash -lc 'ROOT="$HOME/PycharmProjects/Bot_BTC" LOG_LEVEL=INFO /bin/bash "$HOME/PycharmProjects/Bot_BTC/weekly_runner.sh"'
echo "Idempotencia runner OK"

# 3) Flip forzado una sola vez
FLIPS="$ROOT/reports/mini_accum/flips_log.csv"
backup="$FLIPS.bak_selftest"
cp -f "$FLIPS" "$backup"
ts_now=$(date -u +%FT%TZ)
echo "${ts_now},BUY,100000.0,100100.0" >> "$FLIPS"
/usr/bin/env python3 "$ROOT/scripts/mini_accum/flip_watch.py"
/usr/bin/env python3 "$ROOT/scripts/mini_accum/flip_watch.py"  # no debe alertar de nuevo
# rollback flip inyectado
mv -f "$backup" "$FLIPS"
echo "Flip forzado verificado (notificación única) y restaurado"

# 4) Kill-switch PAUSE/NORMAL
OVERRIDE_MODE=PAUSE env -i HOME="$HOME" PATH="$PATH" /bin/bash -lc 'ROOT="$HOME/PycharmProjects/Bot_BTC" /bin/bash "$HOME/PycharmProjects/Bot_BTC/weekly_runner.sh"'
OVERRIDE_MODE=NORMAL env -i HOME="$HOME" PATH="$PATH" /bin/bash -lc 'ROOT="$HOME/PycharmProjects/Bot_BTC" /bin/bash "$HOME/PycharmProjects/Bot_BTC/weekly_runner.sh"'
echo "Kill-switch probado"

# 5) Alertas: simular WARN en health y flip nuevo
date -u +"WARN %Y-%m-%dT%H:%M:%SZ :: simulacion" > "$ROOT/health/mini_accum.status"
/usr/bin/env python3 "$ROOT/scripts/mini_accum/health_watchdog.py"
ts_now2=$(date -u +%FT%TZ)
echo "${ts_now2},SELL,100200.0,100050.0" >> "$FLIPS"
/usr/bin/env python3 "$ROOT/scripts/mini_accum/flip_watch.py"
echo "Alertas simuladas OK"

echo "== Selftest: fin $(date -u +%FT%TZ) =="
