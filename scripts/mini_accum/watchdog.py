#!/usr/bin/env python3
from pathlib import Path
import json, sys, datetime as dt

UTC = dt.timezone.utc
now = dt.datetime.now(UTC)

msgs = []
ok = True

sig = Path("signals/mini_accum/latest.json")
if sig.exists():
    try:
        ts = json.loads(sig.read_text())["ts_utc"]
        if ts.endswith("Z"): ts = ts.replace("Z", "+00:00")
        t = dt.datetime.fromisoformat(ts)
        if t.tzinfo is None: t = t.replace(tzinfo=UTC)
        age_h = (now - t).total_seconds()/3600
        if age_h > 9:
            ok = False
            msgs.append(f"[WARN] signal stale {age_h:.1f}h (>{9}h)")
    except Exception as e:
        ok = False
        msgs.append(f"[WARN] cannot parse latest.json: {e}")
else:
    ok = False
    msgs.append("[WARN] missing signals/mini_accum/latest.json")

hst = Path("health/mini_accum.status")
if not hst.exists():
    msgs.append("[WARN] missing health/mini_accum.status")

print("\n".join(msgs) if msgs else "[OK] watchdog: fresh")
sys.exit(0 if ok else 1)
