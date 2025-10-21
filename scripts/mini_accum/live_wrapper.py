#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
from datetime import datetime, timezone

def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# --- Config (Binance-only for now; KISS) ---
ROOT = os.environ.get("ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
EXCHANGE = os.environ.get("EXCHANGE", "binanceus")   # fijo por ahora para evitar confusiones
DRYRUN = os.environ.get("DRYRUN", "1")
MAX_TRADE_USD = os.environ.get("MAX_TRADE_USD", "10")
POS_CAP_PCT = os.environ.get("POS_CAP_PCT", "0.10")
SYMBOL = os.environ.get("SYMBOL", "BTC/USDC")
FRESHNESS_MAX_HOURS = float(os.environ.get("FRESHNESS_MAX_HOURS", "6"))
WATCHDOG_HOURS = float(os.environ.get("WATCHDOG_HOURS", "8"))  # reservado

# --- Huellas de arranque ---
print(f"{ts()} [INFO] canary_live: start EXCHANGE={EXCHANGE} DRYRUN={DRYRUN} USD<={MAX_TRADE_USD} cap={POS_CAP_PCT}")
print(f"{ts()} [INFO] canary_live: python={sys.executable}")
print(sys.version.split()[0])
print(f"sys.executable: {sys.executable}")
print(f"{ts()} INFO mini_accum: LOG_LEVEL=INFO aplicado")

# --- Chequeo de frescura de señal ---
sig_path = os.path.join(ROOT, "signals", "mini_accum", "latest.json")
STALE_H = float(os.getenv('STALE_HOURS', '24'))
age_h = None
try:
    with open(sig_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    ts_utc = j.get("ts_utc")
    if ts_utc:
        sigts = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
        age_h = (datetime.now(timezone.utc) - sigts).total_seconds() / 3600.0
except Exception as e:
    print(f"{ts()} [WARN] canary_live: cannot read signal ({e})")

if age_h is not None and age_h > FRESHNESS_MAX_HOURS:
    print(f"[PAUSE] stale signal: {age_h:.1f}h")
    print(f"{ts()} [WARN] mini_accum: stale signal: {age_h:.1f}h — PAUSE")
    sys.exit(0)

# --- DRYRUN / LIVE ---
print(f"{ts()} [INFO] canary_live: ready (signal {'fresh' if age_h is not None else 'unknown'})")
if DRYRUN == "1":
    print("[PAPER] flip: simulated (no order)")
else:
    # Aquí iría la ejecución real en Binance cuando salgamos de sombra
    print("[LIVE] would place order here (not implemented)")

print(f"{ts()} [INFO] canary_live: done")