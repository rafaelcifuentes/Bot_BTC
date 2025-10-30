#!/bin/zsh
# --- begin canary lock (atomic, KISS) ---
LOCKDIR=/tmp/bb_canary.lock
if mkdir "$LOCKDIR" 2>/dev/null; then
  trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT
else
  echo "[SKIP] bb_day: otro proceso en curso" >> "$HOME/PycharmProjects/Bot_BTC/evidence/cron_canary.log"
  exit 0
fi
# --- end canary lock ---
set -euo pipefail

# --- defaults (safe) ---
: ${ROOT:="$HOME/PycharmProjects/Bot_BTC"}
: ${EXCHANGE:=binance}          # Binance.com (global). Compatible con testnet.binance.vision si BINANCE_TESTNET=1
: ${DRYRUN:=1}                  # canario en sombra por defecto
: ${FRESHNESS_MAX_HOURS:=8}     # criterio de frescura para no pausar
: ${STALE_HOURS:=$FRESHNESS_MAX_HOURS}   # compat: el wrapper usa STALE_HOURS
: ${MAX_TRADE_USD:=10}
: ${POS_CAP_PCT:=0.10}

export ROOT EXCHANGE DRYRUN FRESHNESS_MAX_HOURS STALE_HOURS MAX_TRADE_USD POS_CAP_PCT

cd "$ROOT"

# --- safety: solo binance global ---
if [[ "$EXCHANGE" != "binance" ]]; then
  echo "[ABORT] EXCHANGE must be 'binance' (global). Got: $EXCHANGE" >&2
  exit 2
fi

# --- self-heal de frescura (si la señal > FRESHNESS_MAX_HOURS, refresca antes) ---
SIG="$ROOT/signals/mini_accum/latest.json"
if [[ -f "$SIG" ]]; then
  age_h=$(.venv/bin/python - "$SIG" <<'PY'
import json, sys, datetime as dt, pathlib
p = pathlib.Path(sys.argv[1])
j = json.loads(p.read_text())
ts = j.get("ts_utc") or j.get("ts_iso") or j.get("ts") or j.get("updated_at")
if not ts:
    print(1e9); sys.exit(0)
t = dt.datetime.fromisoformat(ts.replace("Z","+00:00"))
print((dt.datetime.now(dt.timezone.utc)-t).total_seconds()/3600)
PY
)
  float AGE=$age_h
  float MAX=$FRESHNESS_MAX_HOURS
  if (( AGE >= MAX )); then
    echo "[INFO] bb_day: signal stale (${AGE}h >= ${MAX}h) — refreshing via runner_cron.sh…"
    ./scripts/mini_accum/runner_cron.sh >> logs/cron.log 2>&1 || echo "[WARN] runner_cron.sh failed; proceeding"
    # Recalcular edad tras el refresh
    age_h=$(.venv/bin/python - "$SIG" <<'PY'
import json, sys, datetime as dt, pathlib
p = pathlib.Path(sys.argv[1])
j = json.loads(p.read_text())
ts = j.get("ts_utc") or j.get("ts_iso") or j.get("ts") or j.get("updated_at")
if not ts:
    print(1e9); sys.exit(0)
t = dt.datetime.fromisoformat(ts.replace("Z","+00:00"))
print((dt.datetime.now(dt.timezone.utc)-t).total_seconds()/3600)
PY
)
    float AGE2=$age_h
    echo "[INFO] bb_day: post-refresh age=${AGE2}h"
    if (( AGE2 >= MAX )); then
      echo "[SKIP] bb_day: still stale after refresh (${AGE2}h > ${MAX}h) — skipping live_wrapper"
      exit 0
    fi
  fi
fi

# --- evidencia por día (UTC) ---
DAY="dayN_$(date -u +%F)"
mkdir -p "evidence/$DAY"

# --- run ---
.venv/bin/python scripts/mini_accum/live_wrapper.py \
  | tee -a "evidence/$DAY/canary_live.$(date -u +%Y%m%dT%H%M%SZ).log"