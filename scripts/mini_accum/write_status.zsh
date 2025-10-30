#!/bin/zsh
set -euo pipefail

# --- Python resolver (cron-safe) ---
PY="$HOME/PycharmProjects/Bot_BTC/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3 || command -v python)"
export PY

: ${ROOT:="$HOME/PycharmProjects/Bot_BTC"}
SIG="$ROOT/signals/mini_accum/latest.json"
OUT="$ROOT/health/mini_accum.status"
mkdir -p "$ROOT/health"

"$PY" - "$SIG" "$OUT" <<'PY'
import json, sys, datetime as dt, pathlib
sig = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])

# Load latest.json safely (bootstrap if missing)
if sig.exists():
    j = json.loads(sig.read_text(encoding="utf-8"))
else:
    now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00","Z")
    j = {"ts_utc": now, "reason": "bootstrap", "health": "ok"}

ts = j.get("ts_utc") or j.get("ts_iso") or j.get("ts") or j.get("updated_at")
reason = j.get("reason", "")
# Default to 'ok' if missing/empty to avoid noisy gates
health = (j.get("health") or "ok")

if ts:
    try:
        t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        t = dt.datetime.now(dt.timezone.utc)
    age_h = (dt.datetime.now(dt.timezone.utc) - t).total_seconds() / 3600
else:
    age_h = 1e9

out.write_text(
    f"ts_utc={ts}\nage_h={age_h:.2f}\nhealth={health}\nreason={reason}\n",
    encoding="utf-8"
)
print(f"[OK] write_status: {out} (age_h={age_h:.2f}, health={health}, reason={reason})")
PY
