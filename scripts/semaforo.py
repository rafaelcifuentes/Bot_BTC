#!/usr/bin/env python3
import csv, json, re, sys
from pathlib import Path
from datetime import datetime

REG = Path("registry_runs.csv")
OUT = Path("reports/semaforo.csv")

# Fallback regex por si toca leer texto en vez de claves
RE_PF90   = re.compile(r'(?:90d|base\s*90)[^\r\n]*?pf[^0-9\-]*([0-9]+(?:\.[0-9]+)?)', re.I)
RE_PF180  = re.compile(r'(?:180d|base\s*180)[^\r\n]*?pf[^0-9\-]*([0-9]+(?:\.[0-9]+)?)', re.I)
RE_MDD180 = re.compile(r'(?:180d|base\s*180)[^\r\n]*?mdd[^0-9\-]*([0-9]+(?:\.[0-9]+)?)', re.I)

def parse_dt(s: str) -> datetime:
    try:
        s = (s or "").replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.min

def extract(data, raw_text=None):
    pf90 = pf180 = mdd180 = None

    # 1) baseline.{90d,180d}
    try:
        base = data.get("baseline") if isinstance(data, dict) else None
        if isinstance(base, dict):
            d90 = base.get("90d") or base.get("d90") or base.get("h90")
            if isinstance(d90, dict):
                v = d90.get("pf") or d90.get("PF") or d90.get("profit_factor")
                if isinstance(v, (int, float)): pf90 = float(v)
            d180 = base.get("180d") or base.get("d180")
            if isinstance(d180, dict):
                v = d180.get("pf") or d180.get("PF") or d180.get("profit_factor")
                if isinstance(v, (int, float)): pf180 = float(v)
                v2 = d180.get("mdd") or d180.get("MDD") or d180.get("max_drawdown") or d180.get("mdd_pct")
                if isinstance(v2, (int, float)): mdd180 = float(v2)
    except Exception:
        pass

    # 1b) base90 / base180
    try:
        b90 = data.get("base90") if isinstance(data, dict) else None
        if b90 is None and isinstance(data, dict):
            for k in ("base_90", "b90"):
                if isinstance(data.get(k), dict):
                    b90 = data[k]; break
        if pf90 is None and isinstance(b90, dict):
            v = b90.get("pf")
            if isinstance(v, (int, float)): pf90 = float(v)

        b180 = data.get("base180") if isinstance(data, dict) else None
        if b180 is None and isinstance(data, dict):
            for k in ("base_180", "b180"):
                if isinstance(data.get(k), dict):
                    b180 = data[k]; break
        if isinstance(b180, dict):
            if pf180 is None:
                v = b180.get("pf")
                if isinstance(v, (int, float)): pf180 = float(v)
            if mdd180 is None:
                v = b180.get("mdd")
                if isinstance(v, (int, float)): mdd180 = float(v)
    except Exception:
        pass

    # 2) Claves planas
    try:
        if pf90 is None:
            for k in ("pf_90d", "pf90", "90d_pf"):
                v = data.get(k) if isinstance(data, dict) else None
                if isinstance(v, (int, float)): pf90 = float(v); break
        if pf180 is None:
            for k in ("pf_180d", "pf180", "180d_pf"):
                v = data.get(k) if isinstance(data, dict) else None
                if isinstance(v, (int, float)): pf180 = float(v); break
        if mdd180 is None:
            for k in ("mdd_180d", "mdd180", "180d_mdd", "max_drawdown_180d"):
                v = data.get(k) if isinstance(data, dict) else None
                if isinstance(v, (int, float)): mdd180 = float(v); break
    except Exception:
        pass

    # 3) Fallback regex sobre texto crudo
    if raw_text:
        if pf90 is None:
            m = RE_PF90.search(raw_text)
            if m: pf90 = float(m.group(1))
        if pf180 is None:
            m = RE_PF180.search(raw_text)
            if m: pf180 = float(m.group(1))
        if mdd180 is None:
            m = RE_MDD180.search(raw_text)
            if m: mdd180 = float(m.group(1))

    return pf90, pf180, mdd180

def status_from(pf180, mdd180):
    if pf180 is None: return "unknown"
    if pf180 >= 1.10 and (mdd180 is None or mdd180 <= 0.20): return "green"
    if pf180 >= 1.00: return "yellow"
    return "red"

def main():
    rows = []
    if not REG.exists():
        print("⚠️ No existe", REG)
        sys.exit(1)

    with REG.open() as f:
        r = csv.reader(f)
        for row in r:
            if not row: continue
            if row[0] and row[0].startswith("timestamp"): continue
            row = [c.strip() for c in row]
            if len(row) < 4: continue
            ts, strat, mode, summary_path = row[:4]
            sp = Path(summary_path) if summary_path else None

            pf90 = pf180 = mdd180 = None
            if sp and sp.exists():
                txt = sp.read_text()
                try:
                    data = json.loads(txt)
                except Exception:
                    data = {}
                pf90, pf180, mdd180 = extract(data, txt)

            rows.append((
                parse_dt(ts), ts, strat, mode, summary_path,
                pf90, pf180, mdd180, status_from(pf180, mdd180)
            ))

    # Elegir la última por (strategy, mode)
    picked = {}
    for rec in rows:
        key = (rec[2], rec[3])
        if key not in picked or rec[0] > picked[key][0]:
            picked[key] = rec

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","strategy","mode","summary_path","pf_90d","pf_180d","mdd_180d","status"])
        for key in sorted(picked):
            dt, ts, strat, mode, path, p90, p180, m180, st = picked[key]
            w.writerow([ts, strat, mode, path, p90, p180, m180, st])

    print(f"✅ Semáforo regenerado → {OUT}")

if __name__ == "__main__":
    main()
