#!/usr/bin/env python3
import logging
from logging.handlers import TimedRotatingFileHandler
import os, json, subprocess
from pathlib import Path
from datetime import datetime, timezone

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", Path(__file__).resolve().parents[2]))
SIG_PATH    = PROJECT_DIR / "signals/mini_accum/latest.json"
STATUS_PATH = PROJECT_DIR / "health/mini_accum.status"
WATCHDOG_H  = float(os.getenv("WATCHDOG_HOURS", "8"))

# Logging (daily rotation at UTC midnight)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOGS_DIR = PROJECT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("mini_accum.watchdog")
if not logger.handlers:
    handler = TimedRotatingFileHandler(
        str(LOGS_DIR / "watchdog.log"),
        when="midnight", utc=True, backupCount=7, encoding="utf-8"
    )
    fmt = logging.Formatter("%(asctime)sZ [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.info("LOG_LEVEL=%s aplicado", LOG_LEVEL)

def notify(level: str, msg: str, chan: str="mini_accum"):
    env = os.environ.copy()
    env["LEVEL"] = level
    env["CHAN"]  = chan
    try:
        _map = {"WARN":"warning","WARNING":"warning","INFO":"info","ERROR":"error","CRITICAL":"critical","DEBUG":"debug"}
        _lvl = _map.get(level.upper(), "info")
        getattr(logger, _lvl, logger.info)("[notify:%s] %s", chan, msg)
    except Exception:
        logger.info("[notify:%s] %s", chan, msg)
    try:
        subprocess.run(
            ["/usr/bin/env", "python3", str(PROJECT_DIR / "scripts/mini_accum/notify.py"), msg],
            check=False, env=env
        )
    except Exception:
        pass

logger.info("Watchdog run started")

ok = True
reasons = []

# Señal: frescura + health
try:
    sig = json.loads(SIG_PATH.read_text(encoding="utf-8"))
    ts = datetime.fromisoformat(sig.get("ts_utc","").replace("Z","+00:00"))
    age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
    if age_h > WATCHDOG_H:
        ok = False
        reasons.append(f"stale signal {age_h:.1f}h>{WATCHDOG_H:.0f}h")
        logger.warning("Reason: stale signal %.1fh > %.0fh", age_h, WATCHDOG_H)
    if sig.get("health") != "OK":
        ok = False
        reasons.append(f"signal.health={sig.get('health')}")
        logger.warning("Reason: signal.health=%s", sig.get("health"))
except Exception as e:
    ok = False
    reasons.append(f"bad latest.json: {e}")
    logger.exception("Reason: bad latest.json: %s", e)

# Health file
try:
    text = STATUS_PATH.read_text(encoding="utf-8").strip()
    if not text.startswith("OK"):
        ok = False
        reasons.append(f"status={text or 'EMPTY'}")
        logger.warning("Reason: status=%s", text or "EMPTY")
except Exception as e:
    ok = False
    reasons.append(f"no status: {e}")
    logger.exception("Reason: no status: %s", e)

msg = "Watchdog OK" if ok else "Watchdog WARN: " + "; ".join(reasons)
if ok:
    logger.info(msg)
else:
    logger.warning(msg)
notify("INFO" if ok else "WARN", msg)
print(f"[INFO] mini_accum: {msg}")
