import csv
from pathlib import Path

reg = Path("registry_runs.csv")
missing = []
if reg.exists():
    with reg.open() as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0].startswith("timestamp"):
                continue
            p = (row[3] if len(row) > 3 else "").strip()
            if p and not Path(p).exists():
                missing.append(p)

if missing:
    print("❌ Faltan:\n- " + "\n- ".join(missing))
else:
    print("✅ Todos los summary_path del registry existen.")
