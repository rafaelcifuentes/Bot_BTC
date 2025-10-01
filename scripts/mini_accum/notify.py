#!/usr/bin/env python3
import os, sys, logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", Path(__file__).resolve().parents[2]))
LOGS_DIR = PROJECT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.getenv("LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper()
CHAN = os.getenv("CHAN", "mini_accum")
msg = " ".join(sys.argv[1:]).strip() or "(sin mensaje)"

logger = logging.getLogger(f"notify.{CHAN}")
if not logger.handlers:
    h = TimedRotatingFileHandler(str(LOGS_DIR / "notify.log"), when="midnight", utc=True, backupCount=7, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)sZ [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Línea inicial obligatoria
logger.info("LOG_LEVEL=%s aplicado", LOG_LEVEL)

# Salida estándar + log
line = f"[{LOG_LEVEL}] {CHAN}: {msg}"
_map = {"WARN":"warning","WARNING":"warning","INFO":"info","ERROR":"error","CRITICAL":"critical","DEBUG":"debug"}
getattr(logger, _map.get(LOG_LEVEL.upper(), "info"), logger.info)(msg)
print(line)
