#!/usr/bin/env python3
import os, csv, logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", Path(__file__).resolve().parents[2]))
FLIPS = PROJECT_DIR / "reports/mini_accum/flips_log.csv"
STATE = PROJECT_DIR / "health/flip_watch.state"
LOGS  = PROJECT_DIR / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("mini_accum.flip_watch")
if not logger.handlers:
    h = TimedRotatingFileHandler(str(LOGS / "flip_watch.log"), when="midnight", utc=True, backupCount=7, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)sZ [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.info("LOG_LEVEL=%s aplicado", LOG_LEVEL)

def notify(level: str, msg: str):
    os.environ["LEVEL"] = level
    os.environ["CHAN"] = "mini_accum"
    os.system(f'/usr/bin/env python3 "{PROJECT_DIR}/scripts/mini_accum/notify.py" "{msg}"')

# Cargar última marca
last_ts = "1970-01-01T00:00:00+00:00"
if STATE.exists():
    last_ts = STATE.read_text(encoding="utf-8").strip() or last_ts

def parse_ts(x: str) -> datetime:
    x = x.strip().replace("Z","+00:00")
    return datetime.fromisoformat(x)

new_latest = last_ts
new_flips = []

if FLIPS.exists():
    with FLIPS.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("ts") or row.get("TS") or ""
            if not ts:
                continue
            if parse_ts(ts) > parse_ts(last_ts):
                new_flips.append(ts)
                if parse_ts(ts) > parse_ts(new_latest):
                    new_latest = ts

if new_flips:
    for ts in new_flips:
        msg = f"Nuevo FLIP detectado @ {ts}"
        logger.info(msg)
        notify("INFO", msg)
    STATE.write_text(new_latest, encoding="utf-8")
else:
    logger.info("Sin nuevos flips desde %s", last_ts)
print(f"[INFO] mini_accum: flip_watch {'+%d flips'%len(new_flips) if new_flips else 'no-change'}")
