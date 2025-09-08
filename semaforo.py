# scripts/semaforo.py
from __future__ import annotations
import csv, json, re, os
from pathlib import Path
from datetime import datetime, timezone

REG = Path("registry_runs.csv")
OUT = Path("reports/semaforo.csv")

# --- fechas ---
def iso_parse(s: str) -> str:
    s = (s or "").strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
    except Exception:
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z"):
            try:
                return datetime.strptime(s, fmt).isoformat()
            except Exception:
                pass
        return s

# --- regex fallback sobre texto semi-estructurado ---
RE_PF90   = re.compile(r"90d[^\n]*?pf[^0-9\-]*([0-9]+(?:\.[0-9]+)?)", re.I)
RE_PF180  = re.compile(r"180d[^\n]*?pf[^0-9\-]*([0-9]+(?:\.[0-9]+)?)", re.I)
RE_MDD180 = re.compile(r"180d[^\n]*?mdd[^0-9\-]*([0-9]+(?:\.[0-9]+)?)", re.I)

def extract_from_text(text: str):
    pf90 = pf180 = mdd180 = None
    if text:
        m = RE_PF90.search(text);   pf90   = float(m.group(1)) if m else None
        m = RE_PF180.search(text);  pf180  = float(m.group(1)) if m else None
        m = RE_MDD180.search(text); mdd180 = float(m.group(1)) if m else None
    return pf90, pf180, mdd180

# --- utilidades JSON ---
def load_json(p: Path):
    try:
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def walk_collect(d, want_keys):
    out = {}
    def rec(x):
        if isinstance(x, dict):
            for k, v in x.items():
                kl = str(k).lower()
                if isinstance(v, (int, float)) and any(w in kl for w in want_keys):
                    out[kl] = float(v)
                elif isinstance(v, str):
                    m = re.search(r"[-+]?\d+(\.\d+)?", v)
                    if m and any(w in kl for w in want_keys):
                        out[kl] = float(m.group(0))
                rec(v)
        elif isinstance(x, list):
            for v in x: rec(v)
    rec(d)
    return out

def extract_from_json_like(data):
    """Intenta baseline → escaneo recursivo → regex sobre el volcado JSON."""
    pf90 = pf180 = mdd180 = None

    # 1) Baseline con estructura típica
    try:
        base = data.get("baseline") if isinstance(data, dict) else None
        if isinstance(base, dict):
            n90 = base.get("90d")
            n180 = base.get("180d")
            if isinstance(n90, dict):
                v = n90.get("pf")
                if isinstance(v, (int, float)): pf90 = float(v)
            if isinstance(n180, dict):
                v = n180.get("pf")
                if isinstance(v, (int, float)): pf180 = float(v)
                v = n180.get("mdd")
                if isinstance(v, (int, float)): mdd180 = float(v)
    except Exception:
        pass

    # 2) Escaneo si falta algo
    if pf90 is None or pf180 is None or mdd180 is None:
        bag = walk_collect(data, ["pf", "mdd", "90", "180", "baseline"])
        if pf90 is None:
            for k, v in bag.items():
                if "pf" in k and "90" in k: pf90 = v; break
        if pf180 is None:
            for k, v in bag.items():
                if "pf" in k and "180" in k: pf180 = v; break
        if mdd180 is None:
            for k, v in bag.items():
                if "mdd" in k and "180" in k: mdd180 = v; break

    # 3) Regex sobre el volcado JSON, por si sólo hay texto
    if pf90 is None or pf180 is None or mdd180 is None:
        blob = json.dumps(data, ensure_ascii=False)
        t90, t180, tmdd = extract_from_text(blob)
        if pf90 is None:  pf90 = t90
        if pf180 is None: pf180 = t180
        if mdd180 is None: mdd180 = tmdd

    return pf90, pf180, mdd180

def derive(path: Path, prefix_from: str, prefix_to: str, suffix_from=".json", suffix_to=".json") -> Path:
    base = path.name
    if base.startswith(prefix_from):
        base = prefix_to + base[len(prefix_from):]
    base = base.replace(suffix_from, suffix_to)
    return path.with_name(base)

def try_all_sources(summary_path: Path):
    """Orden de intento:
       1) summary json
       2) competitiveness_summary json (derivado del nombre)
       3) regex sobre texto del summary
    """
    pf90 = pf180 = mdd180 = None

    # 1) summary (JSON)
    data = load_json(summary_path)
    if data is not None:
        pf90, pf180, mdd180 = extract_from_json_like(data)

    # 2) competitiveness_summary (JSON), si falta algo
    if pf90 is None or pf180 is None or mdd180 is None:
        comp = derive(summary_path, "summary_", "competitiveness_summary_")
        d2 = load_json(comp)
        if d2 is not None:
            a, b, c = extract_from_json_like(d2)
            if pf90   is None: pf90   = a
            if pf180  is None: pf180  = b
            if mdd180 is None: mdd180 = c

    # 3) Regex duro sobre el texto del summary
    if pf90 is None or pf180 is None or mdd180 is None:
        try:
            blob = summary_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            blob = ""
        a, b, c = extract_from_text(blob)
        if pf90   is None: pf90   = a
        if pf180  is None: pf180  = b
        if mdd180 is None: mdd180 = c

    return pf90, pf180, mdd180

def status_from(pf180, mdd180):
    if pf180 is None: return "unknown"
    if pf180 >= 1.10 and (mdd180 is None or mdd180 <= 0.20): return "green"
    if pf180 >= 1.00: return "yellow"
    return "red"

def main():
    rows = []
    if not REG.exists():
        print("⚠️ No existe registry_runs.csv"); return

    with REG.open(encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            c0 = (row[0] or "").strip().lower()
            if c0.startswith("timestamp"):
                continue

            row = [(c or "").strip() for c in row]
            # Soporta 4 o 5 columnas en el registry
            if len(row) < 4:
                continue
            ts, strat, mode, summary_path = row[:4]
            sp = Path(summary_path) if summary_path else None
            if sp is None or not sp.exists():
                # intenta normalizar espacios
                sp = Path(summary_path.strip()) if summary_path else None

            pf90 = pf180 = mdd180 = None
            if sp and sp.exists():
                pf90, pf180, mdd180 = try_all_sources(sp)

            st = status_from(pf180, mdd180)
            rows.append((iso_parse(ts), ts, strat, mode, summary_path, pf90, pf180, mdd180, st))

    # Elegir el más reciente por (strategy, mode)
    picked = {}
    for dt, ts, strat, mode, path, p90, p180, m180, st in rows:
        key = (strat, mode)
        if key not in picked or dt > picked[key][0]:
            picked[key] = (dt, ts, strat, mode, path, p90, p180, m180, st)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","strategy","mode","summary_path","pf_90d","pf_180d","mdd_180d","status"])
        for key in sorted(picked):
            _, ts, strat, mode, path, p90, p180, m180, st = picked[key]
            w.writerow([ts, strat, mode, path, p90, p180, m180, st])

    print(f"✅ Semáforo regenerado → {OUT}")

if __name__ == "__main__":
    main()