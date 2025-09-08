import os, glob, sys, os.path as _p
from datetime import datetime, timezone

REG = "registry_runs.csv"

def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

paths = sorted(
    glob.glob("reports/summary_perla_negra_*.json"),
    key=os.path.getmtime, reverse=True
)

if len(paths) < 2:
    sys.exit("⚠️ No hay suficientes summaries de PerlaNegra en reports/")

live, freeze = paths[0], paths[1]

def wf(p):
    base = os.path.basename(p)
    return _p.join("reports", base.replace("summary_", "walkforward_").replace(".json",".csv"))

os.makedirs("reports", exist_ok=True)
with open(REG, "a") as f:
    f.write(f"{now()},PerlaNegra,freeze,{freeze},{wf(freeze)}\n")
    f.write(f"{now()},PerlaNegra,live,  {live},{wf(live)}\n")

print("✅ Registradas PerlaNegra (freeze & live) en registry_runs.csv")