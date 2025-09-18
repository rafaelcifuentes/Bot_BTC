#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import logging, time
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
handler = TimedRotatingFileHandler(
    LOG_DIR / "mini_accum.log", when="midnight", interval=1, backupCount=7,
    utc=True, encoding="utf-8"
)
console = logging.StreamHandler()

class _UTCFormatter(logging.Formatter):
    converter = time.gmtime
fmt = _UTCFormatter("%(asctime)sZ %(levelname)s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%S")

handler.setFormatter(fmt)
console.setFormatter(fmt)

logging.basicConfig(level=logging.DEBUG, handlers=[handler, console])
logger = logging.getLogger("mini_accum")

from pathlib import Path
import logging
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# -*- coding: utf-8 -*-
"""
KISS signal emitter (Python-only)
- Lee OHLCV D1/4h desde CSV (paths por ENV o YAML)
- Calcula EMA200 (D1) y EMA21/EMA55 (4h)
- Emite latest.json con {ts_utc, position_pct_btc, reason, version, health, guards}
- Health "WARN" si datos stale (respetando ENFORCE_FRESH)
"""
import os, json
from pathlib import Path
import pandas as pd
try:
    import yaml  # opcional (para leer kiss_v1.yaml)
except Exception:
    yaml = None

# ----------------------------
# Utilidades
# ----------------------------
def _read_yaml(path: Path) -> dict:
    if yaml is None or not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def _load_close_series(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV no existe: {csv_path}")
    df = pd.read_csv(csv_path)
    # heurística de columna de tiempo
    time_col = None
    for cand in ("ts", "timestamp", "date", "datetime", "time"):
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        # toma la primera columna como tiempo
        time_col = df.columns[0]
    # heurística de columna de cierre
    close_col = None
    for cand in ("close", "Close", "c", "price"):
        if cand in df.columns:
            close_col = cand
            break
    if close_col is None:
        # si hay ≥2 columnas, asume segunda es close
        if len(df.columns) >= 2:
            close_col = df.columns[1]
        else:
            raise ValueError(f"No encontré columna de cierre en {csv_path}")

    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col).sort_index()
    # normaliza a UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    out = df[[close_col]].rename(columns={close_col: "close"})
    return out

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

# ----------------------------
# Lógica principal
# ----------------------------
def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]  # .../Bot_BTC
    yaml_path = repo_root / "configs/mini_accum/kiss_v1.yaml"
    cfg = _read_yaml(yaml_path) if yaml_path.exists() else {}
    paths = (cfg.get("paths") or {}) if isinstance(cfg, dict) else {}

    # prioridad: ENV > YAML > defaults
    d1_csv = Path(os.getenv("D1_CSV") or paths.get("d1_csv") or "data/ohlc/1d/BTC-USD.csv").expanduser()
    h4_csv = Path(os.getenv("H4_CSV") or paths.get("h4_csv") or "data/ohlc/4h/BTC-USD.csv").expanduser()

    print(f"[INFO] D1_CSV -> {d1_csv}")
    print(f"[INFO] H4_CSV -> {h4_csv}")

    d1 = _load_close_series(d1_csv)
    h4 = _load_close_series(h4_csv)
    if d1.empty or h4.empty:
        raise ValueError("Series vacías tras cargar CSV (¿CSV corrupto o sin datos?)")

    # EMAs
    d1["ema200"] = _ema(d1["close"], 200)
    h4["ema21"]  = _ema(h4["close"], 21)
    h4["ema55"]  = _ema(h4["close"], 55)

    # Señal por reglas KISS
    d1_last = d1.index.max()
    h4_last = h4.index.max()
    now     = pd.Timestamp.now(tz="UTC")

    macro_green = bool(d1.loc[d1_last, "close"] > d1.loc[d1_last, "ema200"])
    trend_up    = bool(h4.loc[h4_last, "ema21"] > h4.loc[h4_last, "ema55"])
    position    = 1.0 if (macro_green and trend_up) else 0.0

    # Freshness check (stale)
    d1_age_h = (now - d1_last).total_seconds() / 3600.0
    h4_age_h = (now - h4_last).total_seconds() / 3600.0
    stale_d1 = d1_age_h > 48
    stale_h4 = h4_age_h > 8
    is_stale = stale_d1 or stale_h4

    if is_stale:
        print(f"[WARN] Datos stale: D1={d1_last}  H4={h4_last}  now={now}")

    # Health
    enforce_fresh = os.getenv("ENFORCE_FRESH", "0") not in ("0", "", "false", "False", "no", "NO")
    health = "OK"
    if is_stale and enforce_fresh:
        health = "WARN"

    # Version & guards
    version = os.getenv("KISS_VERSION") or (cfg.get("version") if isinstance(cfg, dict) else None) or "KISSv1_BASE_20250915_1642_final"
    bull_hold_env = os.getenv("BULL_HOLD", "0") not in ("0", "", "false", "False", "no", "NO")

    # Output JSON
    signal = {
        "version": version,
        "ts_utc": now.isoformat(),
        "position_pct_btc": position,
        "reason": f"macro_green={macro_green}, trend_up={trend_up}",
        "health": health,
        "guards": {
            "bull_hold": bool(bull_hold_env),
            "override": "NONE"
        }
    }

    out_path = repo_root / "signals/mini_accum/latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(signal, indent=2), encoding="utf-8")
    print(f"[OK] Señal escrita -> {out_path}")
    print(f"D1 last: {d1_last} close= {d1.loc[d1_last, 'close']} ema200= {d1.loc[d1_last, 'ema200']}")
    print(f"H4 last: {h4_last} ema21= {h4.loc[h4_last, 'ema21']} ema55= {h4.loc[h4_last, 'ema55']}")
    print(json.dumps(signal, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
