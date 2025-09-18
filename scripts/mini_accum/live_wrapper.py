#!/usr/bin/env python3
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

import os, json
OVERRIDE = os.getenv("OVERRIDE_MODE", "NONE").upper()  # NONE|PAUSE
if OVERRIDE == "PAUSE":
    print("[PAUSE] override_mode=PAUSE")
    # escribe health en PAUSE y sal
    Path("health").mkdir(exist_ok=True, parents=True)
    (Path("health")/"mini_accum.status").write_text("PAUSE\n", encoding="utf-8")
    raise SystemExit(0)
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# -*- coding: utf-8 -*-
"""
mini_accum → Corazón (executor/guardian) — KISS pass-through
- No re-pondera ni mezcla. 0% dilución.
- Llamar cada 4h (ideal: tras cerrar la vela, al open de la siguiente).
- RUN_MODE=paper => simula y registra; RUN_MODE=live => ccxt (si falla, cae a paper).
"""
# --- ADD: logging + watchdog + helpers ---

import os, json, csv
from datetime import datetime, timezone, timedelta
from pathlib import Path

import logging, logging.handlers
from pathlib import Path
import pandas as pd
LOG_DIR.mkdir(parents=True, exist_ok=True)

_logger = logging.getLogger("mini_accum.live")
_logger.setLevel(logging.DEBUG)

_fh = logging.handlers.TimedRotatingFileHandler(
    LOG_DIR / "mini_accum.log", when="midnight", backupCount=14, utc=True
)
_fh.setFormatter(logging.Formatter("%(asctime)sZ %(levelname)s %(message)s"))
_fh.setLevel(logging.DEBUG)
_logger.addHandler(_fh)

_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
_sh.setLevel(logging.INFO)
_logger.addHandler(_sh)

def _last_close(csv_path: str):
    try:
        df = pd.read_csv(csv_path, parse_dates=["ts"])
        if df.empty:
            return None, None
        ts = df["ts"].iloc[-1]
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return float(df["close"].iloc[-1]), ts
    except Exception as e:
        _logger.warning(f"last_close failed for {csv_path}: {e}")
        return None, None

def _watchdog_check(sig_ts_utc: str, max_hours: float = 8.0) -> bool:
    try:
        now = pd.Timestamp.utcnow().tz_localize("UTC")
        ts = pd.Timestamp(sig_ts_utc).tz_convert("UTC")
        age_h = (now - ts).total_seconds() / 3600.0
        return age_h <= max_hours
    except Exception as e:
        _logger.warning(f"watchdog parse error: {e}")
        return False


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

    # --- ADD inside main(), after reading latest.json into 'sig' and before flipping ---

    # 1) Watchdog (si la señal está vieja, no actuamos)
    WD_HOURS = float(os.getenv("WATCHDOG_HOURS", "8"))
    if WD_HOURS > 0:
        is_fresh = _watchdog_check(sig.get("ts_utc", ""), max_hours=WD_HOURS)
        if not is_fresh:
            # escribe estado de salud y aborta acción
            Path("health").mkdir(exist_ok=True, parents=True)
            with open("health/mini_accum.status", "a", encoding="utf-8") as fh:
                fh.write("watchdog,WARN,stale_signal\n")
            _logger.warning("Watchdog: señal stale (> %sh). No-Op y WARN.", WD_HOURS)
            print("Watchdog: stale signal — no-op")
            return 0

    # 2) Calcular/actualizar NetBTC vs HODL live (con último close 4h; fallback 1d)
    price, pts = _last_close(os.getenv("H4_CSV", "data/ohlc/4h/BTC-USD.csv"))
    if price is None:
        price, pts = _last_close(os.getenv("D1_CSV", "data/ohlc/1d/BTC-USD.csv"))

    state_path = Path("state/mini_accum_state.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        st = json.loads(state_path.read_text())
    else:
        st = {"prev_price": None, "model_equity": 1.0, "hodl_equity": 1.0}

    prev_price = st.get("prev_price")
    model_eq = float(st.get("model_equity", 1.0))
    hodl_eq = float(st.get("hodl_equity", 1.0))

    if price is not None:
        if prev_price is None:
            # primer sample: inicializa y no mueves equity
            st["prev_price"] = price
        else:
            ratio = price / float(prev_price)
            # HODL siempre sigue precio
            hodl_eq *= ratio
            # modelo sólo cuando está en BTC (pos=1)
            pos = float(sig.get("position_pct_btc", 0.0))
            if pos >= 0.999:
                model_eq *= ratio
            # guarda
            st["prev_price"] = price
            st["model_equity"] = model_eq
            st["hodl_equity"] = hodl_eq

        # persistir estado
        state_path.write_text(json.dumps(st, ensure_ascii=False, indent=2))

        # loggear a live_kpis.csv
        kpis_path = Path("reports/mini_accum/live_kpis.csv")
        kpis_path.parent.mkdir(parents=True, exist_ok=True)
        net_btc_vs_hodl = (model_eq / hodl_eq) if hodl_eq > 0 else 1.0
        with open(kpis_path, "a", encoding="utf-8") as fh:
            fh.write(
                f"{pd.Timestamp.utcnow().tz_localize('UTC')},{price},{sig.get('position_pct_btc')},"
                f"{model_eq:.6f},{hodl_eq:.6f},{net_btc_vs_hodl:.6f}\n"
            )
        _logger.debug(f"LiveKPIs: price={price} model={model_eq:.6f} hodl={hodl_eq:.6f} net={net_btc_vs_hodl:.6f}")


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
