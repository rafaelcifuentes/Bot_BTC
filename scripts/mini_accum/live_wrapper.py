#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mini_accum → Corazón (executor/guardian) — KISS pass-through
- No re-pondera ni mezcla. 0% dilución.
- Llamar cada 4h (ideal: tras cerrar la vela, al open de la siguiente).
- RUN_MODE=paper => simula y registra; RUN_MODE=live => ccxt (si falla, cae a paper).
"""

import os, json, csv
from datetime import datetime, timezone, timedelta
from pathlib import Path

RUN_MODE     = os.getenv("RUN_MODE", "paper").lower()  # "paper" o "live"
REPO         = Path(__file__).resolve().parents[2]
SIGNAL_PATH  = Path(os.getenv("SIGNAL_PATH", REPO/"signals/mini_accum/latest.json"))
STATE_PATH   = Path(os.getenv("STATE_PATH",  REPO/"reports/mini_accum/state.json"))
FLIPS_LOG    = Path(os.getenv("FLIPS_LOG",   REPO/"reports/mini_accum/flips_log.csv"))
LIVE_KPIS    = Path(os.getenv("LIVE_KPIS",   REPO/"reports/mini_accum/live_kpis.csv"))
STATUS_PATH  = Path(os.getenv("STATUS_PATH", REPO/"health/mini_accum.status"))
SYMBOL       = os.getenv("SYMBOL", "BTC/USDC")
EXCHANGE     = os.getenv("EXCHANGE", "binanceus")
SOFT_MONTH   = int(os.getenv("SOFT_FLIPS_PER_MONTH", "2"))
HARD_YEAR    = int(os.getenv("HARD_FLIPS_PER_YEAR", "26"))
FRESH_MAX_H  = int(os.getenv("FRESHNESS_MAX_HOURS", "6"))

def utcnow(): return datetime.now(timezone.utc)

def load_json(p, default):
    try:
        with open(p, "r", encoding="utf-8") as fh: return json.load(fh)
    except Exception: return default

def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh: json.dump(obj, fh, ensure_ascii=False, indent=2)

def append_csv(p, row):
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=row.keys())
        if new: w.writeheader()
        w.writerow(row)

def flips_in_period(p, days):
    if not p.exists(): return 0
    cutoff = utcnow() - timedelta(days=days)
    n = 0
    with open(p, newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            try:
                ts = datetime.fromisoformat(row["ts_utc"].replace("Z","+00:00"))
                if ts >= cutoff: n += 1
            except Exception: continue
    return n

def set_status(msg):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as fh: fh.write(f"{msg}\n")

def main():
    # 1) Cargar señal
    sig = load_json(SIGNAL_PATH, {})
    if not sig:
        set_status("PAUSE: no signal")
        print("[PAUSE] No signal JSON.")
        return 0

    # 2) Validaciones KISS
    try:
        ts = datetime.fromisoformat(sig["ts_utc"].replace("Z","+00:00"))
    except Exception:
        set_status("PAUSE: bad ts_utc")
        print("[PAUSE] bad ts_utc in signal.")
        return 0

    age_h = (utcnow() - ts).total_seconds() / 3600.0
    if age_h > FRESH_MAX_H:
        set_status(f"PAUSE: stale signal ({age_h:.1f}h)")
        print(f"[PAUSE] stale signal: {age_h:.1f}h")
        return 0

    if sig.get("health") != "OK":
        set_status("PAUSE: health!=OK")
        print("[PAUSE] health!=OK")
        return 0

    desired = float(sig.get("position_pct_btc", 0.0))
    if desired not in (0.0, 1.0):
        set_status("PAUSE: invalid desired pos")
        print("[PAUSE] invalid desired position")
        return 0

    # 3) Presupuesto de flips
    flips_30d = flips_in_period(FLIPS_LOG, 30)
    flips_365 = flips_in_period(FLIPS_LOG, 365)
    soft_ok   = flips_30d <= SOFT_MONTH
    hard_ok   = flips_365 < HARD_YEAR

    # 4) Cargar estado y decidir flip (por defecto: 0.0 = Stable)
    state = load_json(STATE_PATH, {"position_pct_btc": 0.0, "last_flip_ts": None, "version": ""})
    current = float(state.get("position_pct_btc", 0.0))
    delta = abs(desired - current)

    if delta < 0.01:
        set_status("OK: no-op")
        print("[NO-OP] same position.")
        append_csv(LIVE_KPIS, {
            "ts_utc": utcnow().isoformat(),
            "mode": RUN_MODE, "symbol": SYMBOL,
            "pos": current, "flips_30d": flips_30d, "flips_365": flips_365
        })
        return 0

    if not hard_ok:
        set_status("PAUSE: hard flip budget")
        print("[PAUSE] hard flip budget exceeded.")
        return 0
    if not soft_ok:
        print("[WARN] soft monthly budget exceeded; proceeding (soft cap).")

    # 5) Ejecutar flip
    ok = False
    run_mode_used = RUN_MODE
    if RUN_MODE == "live":
        try:
            import ccxt  # type: ignore
            ex = getattr(ccxt, EXCHANGE)()
            # Asume credenciales via env si aplica
            side = "buy" if desired > current else "sell"
            amount = float(os.getenv("TRADE_AMOUNT", "1"))
            ex.create_order(symbol=SYMBOL, type="market", side=side, amount=amount)
            ok = True
            print(f"[LIVE] {side} {amount} {SYMBOL}")
        except Exception as e:
            print(f"[LIVE->PAPER] ccxt failed: {e}")
            run_mode_used = "paper"

    if run_mode_used == "paper" and not ok:
        print(f"[PAPER] flip: {current} -> {desired} ({SYMBOL})")
        ok = True

    if ok:
        state.update({"position_pct_btc": desired, "last_flip_ts": utcnow().isoformat(), "version": sig.get("version","")})
        save_json(STATE_PATH, state)
        append_csv(FLIPS_LOG, {
            "ts_utc": utcnow().isoformat(),
            "from": current, "to": desired,
            "reason": sig.get("reason",""), "version": sig.get("version","")
        })
        append_csv(LIVE_KPIS, {
            "ts_utc": utcnow().isoformat(), "mode": run_mode_used, "symbol": SYMBOL,
            "pos": desired, "flips_30d": flips_30d+1, "flips_365": flips_365+1
        })
        set_status("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
