import csv
from pathlib import Path
from datetime import datetime

REG = Path("registry_runs.csv")
TMP = Path("registry_runs.tmp")

def parse_ts(s: str):
    s = (s or "").strip()
    if not s:
        return datetime.min
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.min

if not REG.exists():
    print("Nada que deduplicar (no existe registry_runs.csv).")
    raise SystemExit(0)

rows = []
header = ["timestamp","strategy","mode","summary_path","walk_path"]
with REG.open() as f:
    r = csv.reader(f)
    for row in r:
        if not row:
            continue
        if row[0].startswith("timestamp"):
            header = row  # preserva encabezado original si existe
            continue
        row = [c.strip() for c in row]
        if len(row) >= 5:
            ts, strat, mode, sp, wp = row[:5]
            rows.append((parse_ts(ts), ts, strat, mode, sp, wp, row))

best = {}
for dt, ts, strat, mode, sp, wp, raw in rows:
    key = (strat, mode, sp, wp)
    if key not in best or dt > best[key][0]:
        best[key] = (dt, raw)

kept = [v[1] for v in sorted(best.values(), key=lambda x: x[0])]

with TMP.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(kept)

TMP.replace(REG)
print(f"✅ Registry deduplicado → {REG}")
print(f"Quedaron {len(kept)} filas (sin duplicados).")
