#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
from datetime import datetime, timezone

def env(name, default=None, cast=str):
    v = os.getenv(name, default)
    if v is None:
        return None
    try:
        return cast(v)
    except Exception:
        return v

def main():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    order_mode = str(env("ORDER_MODE", "ARMED")).upper()
    do_trade   = int(env("DO_TRADE", "1"))
    dryrun     = int(env("DRYRUN", "0"))
    preset     = str(env("KISS_PROD_PRESET", "configs/mini_accum/presets/PROD_TOP.yaml"))

    print(f"{ts} [INFO] live_exec: mode={order_mode} do_trade={do_trade} dryrun={dryrun} preset={preset}")

    if not os.path.exists(preset):
        print(f"{ts} [WARN] live_exec: preset not found: {preset}", file=sys.stderr)
        # ARMED no corta por preset; solo avisa.
        return 0

    if order_mode == "ARMED":
        # Modo ARMED: NO envía órdenes. Solo traza explícita.
        print(f"{ts} [INFO] live_exec: (ARMED) - create real order here (simulated, NOT sent)")
        return 0

    # TESTNET/REAL: conectar ejecutor real aquí cuando toque.
    print(f"{ts} [INFO] live_exec: mode {order_mode} not implemented in this stub")
    return 0

if __name__ == "__main__":
    sys.exit(main())
