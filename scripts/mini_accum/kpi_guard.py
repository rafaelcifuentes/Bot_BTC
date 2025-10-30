#!/usr/bin/env python3
import os, csv, math, logging, sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", Path(__file__).resolve().parents[2]))
LOGS = PROJECT_DIR / "logs"; LOGS.mkdir(parents=True, exist_ok=True)
STATUS = PROJECT_DIR / "health/mini_accum.status"
FLIPS = PROJECT_DIR / "reports/mini_accum/flips_log.csv"
LIVE  = PROJECT_DIR / "reports/mini_accum/live_kpis.csv"

def _parse_limit(val: str, default: float) -> float:
    try:
        s = (val or "").strip()
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        return float(s)
    except Exception:
        return default

EXIT_ON_WARN = os.getenv("EXIT_ON_WARN", "0").lower() in {"1","true","yes","y"}

FPY_LIMIT   = _parse_limit(os.getenv("FPY_LIMIT", "26"), 26.0)
DRIFT_LIMIT = _parse_limit(os.getenv("DRIFT_LIMIT", "0.03"), 0.03)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("mini_accum.kpi_guard")
if not logger.handlers:
    h = TimedRotatingFileHandler(str(LOGS / "kpi_guard.log"), when="midnight", utc=True, backupCount=7, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)sZ [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.info("LOG_LEVEL=%s aplicado", LOG_LEVEL)

def notify(level: str, msg: str):
    os.environ["LEVEL"] = level
    os.environ["CHAN"] = "mini_accum"
    script = PROJECT_DIR / "scripts/mini_accum/notify.py"
    if script.exists():
        os.system(f'/usr/bin/env python3 "{script}" "{msg}"')
    else:
        logger.debug("notify.py no encontrado; omitido. msg=%s", msg)

now = datetime.now(timezone.utc)
warns = []

# 1) FPY anualizado límite 26
fpy_annual = 0.0
if FLIPS.exists():
    ts_list = []
    with FLIPS.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ts = (row.get("ts") or "").strip()
            if not ts: continue
            ts_list.append(datetime.fromisoformat(ts.replace("Z","+00:00")))
    ts_list = [t for t in ts_list if t <= now]
    if ts_list:
        t_min, t_max = min(ts_list), max(ts_list)
        span_days = max((t_max - t_min).total_seconds() / 86400.0, 1.0)
        fpy_annual = len(ts_list) * 365.0 / span_days
        if fpy_annual > FPY_LIMIT + 1e-9:
            warns.append(f"FPY anualizado {fpy_annual:.2f} > {FPY_LIMIT:g}")
else:
    logger.info("No hay flips_log.csv; FPY no evaluado")

# 2) Drift Net/HODL ±3% si columnas existen
drift_chk = "n/a"
if LIVE.exists():
    with LIVE.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows:
        last = rows[-1]
        # Buscar columnas conocidas
        def pick(d, keys):
            for k in keys:
                if k in d and str(d[k]).strip() != "":
                    try:
                        return float(str(d[k]).replace(",", ""))
                    except Exception:
                        continue
            return None
        net = pick(last, ["net_btc_ratio","netBTC","net_btc"])
        hodl = pick(last, ["hodl_btc_ratio","hodlBTC","hodl_btc"])
        if net is not None and hodl is not None and hodl != 0:
            drift = (net - hodl)/hodl
            drift_chk = f"{drift*100:.2f}%"
            if abs(drift) > DRIFT_LIMIT + 1e-12:
                warns.append(f"Drift Net/HODL {drift*100:.2f}% fuera de ±{DRIFT_LIMIT*100:.0f}%")
        else:
            logger.info("LIVE KPIs sin columnas Net/HODL compatibles; drift n/a")
    else:
        logger.info("LIVE KPIs vacío")
else:
    logger.info("No existe live_kpis.csv; drift n/a")

status_line = ""
if warns:
    status_line = f"WARN {now.strftime('%Y-%m-%dT%H:%M:%SZ')} :: " + " | ".join(warns)
    STATUS.write_text(status_line + "\n", encoding="utf-8")
    logger.warning(status_line)
    notify("WARN", f"KPI Guard: {'; '.join(warns)}")
else:
    status_line = f"OK {now.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    STATUS.write_text(status_line + "\n", encoding="utf-8")
    logger.info(status_line)
    notify("INFO", f"KPI Guard OK (FPY={fpy_annual:.2f}, drift={drift_chk})")

print(f"[INFO] mini_accum: KPI Guard -> {status_line}")
if warns and EXIT_ON_WARN:
    sys.exit(1)
sys.exit(0)
