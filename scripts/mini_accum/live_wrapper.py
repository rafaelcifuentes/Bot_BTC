#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
from datetime import datetime, timezone
import datetime as dt

def ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# --- Config (Binance-only for now; KISS) ---
ROOT = os.environ.get("ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
EXCHANGE = os.environ.get("EXCHANGE", "binanceus")   # fijo por ahora para evitar confusiones
DRYRUN = os.environ.get("DRYRUN", "1") == "1"
MAX_TRADE_USD = os.environ.get("MAX_TRADE_USD", "10")
POS_CAP_PCT = os.environ.get("POS_CAP_PCT", "0.10")
SYMBOL = os.environ.get("SYMBOL", "BTC/USDC")
FRESHNESS_MAX_HOURS = float(os.environ.get("STALE_HOURS", os.environ.get("FRESHNESS_MAX_HOURS", "6")))
WATCHDOG_HOURS = float(os.environ.get("WATCHDOG_HOURS", "8"))  # reservado

# --- Huellas de arranque ---
print(f"{ts()} [INFO] canary_live: start EXCHANGE={EXCHANGE} DRYRUN={int(DRYRUN)} USD<={MAX_TRADE_USD} cap={POS_CAP_PCT}")
print(f"{ts()} [INFO] canary_live: python={sys.executable}")
print(sys.version.split()[0])
print(f"sys.executable: {sys.executable}")
print(f"{ts()} INFO mini_accum: LOG_LEVEL=INFO aplicado")

# --- Chequeo de frescura de señal ---
sig_path = os.path.join(ROOT, "signals", "mini_accum", "latest.json")

def _load_latest(path:str):
    try:
        return json.load(open(path))
    except Exception:
        return {}


def _pick_latest_ts(j, sig_path):
    import datetime, os, json, math
    from datetime import timezone
    candidates = []
    def add_iso(k):
        v=j.get(k)
        if isinstance(v,str) and v:
            try: candidates.append(dt.datetime.fromisoformat(v.replace('Z','+00:00')))
            except: pass
        elif isinstance(v,(int,float)) and math.isfinite(v):
            try: candidates.append(dt.datetime.fromtimestamp(v,tz=dt.timezone.utc))
            except: pass
    for k in ("ts_utc","ts_iso","ts","timestamp","updated_at","decided_at"):
        add_iso(k)
    # fallback: mtime del archivo
    try:
        mt=os.path.getmtime(sig_path)
        candidates.append(dt.datetime.fromtimestamp(mt,tz=dt.timezone.utc))
    except: pass
    if not candidates:
        return None
    return max(candidates)

latest = _load_latest(sig_path)
latest_dt = _pick_latest_ts(latest, sig_path)
age_h = None if latest_dt is None else (dt.datetime.now(dt.timezone.utc) - latest_dt).total_seconds()/3600
try:
    with open(sig_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    ts_utc = j.get("ts_utc")
    if ts_utc:
        sigts = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
        age_h = (dt.datetime.now(dt.timezone.utc) - sigts).total_seconds() / 3600.0
except Exception as e:
    print(f"{ts()} [WARN] canary_live: cannot read signal ({e})")

if age_h is not None and age_h > FRESHNESS_MAX_HOURS:
    print(f"[PAUSE] stale signal: {age_h:.1f}h")
    print(f"{ts()} [WARN] mini_accum: stale signal: {age_h:.1f}h — PAUSE")
    sys.exit(0)

# --- DRYRUN / LIVE ---
print(f"{ts()} [INFO] canary_live: ready (signal {'fresh' if age_h is not None else 'unknown'})")
if DRYRUN:
    print("[PAPER] flip: simulated (no order)")
else:
    # Aquí iría la ejecución real en Binance cuando salgamos de sombra
    #print("[LIVE] would place order here (not implemented)")
    # --- EXECUCIÓN OPCIONAL (canario) ---
    # Requisitos para operar:
    #   DO_TRADE=1  SIDE=buy|sell  USD_MAX=10
    # Además respeta DRYRUN: si DRYRUN=1 solo simula.

    DO_TRADE = os.getenv("DO_TRADE", "0") == "1"
    SIDE = os.getenv("SIDE")  # 'buy' o 'sell'
    USD_MAX = float(os.getenv("USD_MAX", "0"))

    if not DO_TRADE:
        print("[LIVE] trading disabled (DO_TRADE!=1) — only logging")
    else:
        if SIDE not in ("buy", "sell") or USD_MAX <= 0:
            print("[LIVE] missing SIDE or USD_MAX — skip")
        else:
            try:
                import ccxt

                ex_name = os.getenv("EXCHANGE", "binanceus")
                ex = getattr(ccxt, ex_name)({"enableRateLimit": True})
                symbol = os.getenv("SYMBOL", SYMBOL)

                # Precio y qty aprox
                px = float(ex.fetch_ticker(symbol)["last"])
                qty = round(USD_MAX / px, 6)

                # TODO: aquí podrías chequear minNotional, cap de posición, etc.
                if DRYRUN:
                    print(f"[DRYRUN] would {SIDE} ~${USD_MAX} ({qty} BTC) @ ~{px}")
                else:
                    # Descomenta cuando quieras realmente enviar:
                    # order = ex.create_order(symbol, 'market', SIDE, qty)
                    # print("[LIVE] order placed:", order)
                    print(f"[LIVE] (armed) {SIDE} ~${USD_MAX} ({qty} BTC) @ ~{px} — crear orden real aquí")
            except Exception as e:
                print("[LIVE] error placing order:", e)
    # --- FIN bloque opcional ---

print(f"{ts()} [INFO] canary_live: done")