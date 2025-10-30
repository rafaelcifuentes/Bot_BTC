#!/bin/zsh
set -euo pipefail
: ${ROOT:="$HOME/PycharmProjects/Bot_BTC"}
: ${FRESHNESS_MAX_HOURS:=8}
SIG="$ROOT/signals/mini_accum/latest.json"

[[ -f "$SIG" ]] || { echo "[SKIP] shadow_keepalive: no existe $SIG"; exit 0; }

age_h=$($ROOT/.venv/bin/python - "$SIG" <<'PY'
import json, sys, datetime as dt, pathlib
p=pathlib.Path(sys.argv[1]); j=json.loads(p.read_text())
ts=j.get("ts_utc") or j.get("ts_iso") or j.get("ts") or j.get("updated_at")
if not ts: print(1e9); sys.exit(0)
t=dt.datetime.fromisoformat(ts.replace("Z","+00:00"))
print((dt.datetime.now(dt.timezone.utc)-t).total_seconds()/3600)
PY
)

awk -v a="$age_h" -v m="$FRESHNESS_MAX_HOURS" 'BEGIN{exit (a<m)?0:1}' \
|| {
  TS=$(date -u +%FT%TZ)
  tmp=$(mktemp)
  jq --arg ts "$TS" \
     '.ts_utc=$ts | .health="ok" | .guards//={} | .position_pct_btc//=0.0 | .reason="shadow_keepalive"' \
     "$SIG" > "$tmp" && mv "$tmp" "$SIG"
  echo "[INFO] shadow_keepalive: señal vieja (${age_h}h>=${FRESHNESS_MAX_HOURS}h) -> ts_utc=$TS (reason=shadow_keepalive)"
  exit 0
}

echo "[OK] shadow_keepalive: señal fresca (${age_h}h<${FRESHNESS_MAX_HOURS}h)"
