#!/usr/bin/env python3
import argparse, csv, glob, os
from pathlib import Path
from datetime import datetime, timezone

def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def pick_last_two(pattern: str):
    paths = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return paths[:2]

def wf_path(summary_path: str) -> str:
    base = os.path.basename(summary_path)
    return os.path.join("reports", base.replace("summary_", "walkforward_").replace(".json", ".csv"))

def exists_in_registry(reg: Path, strategy: str, mode: str, summary_path: str) -> bool:
    if not reg.exists():
        return False
    with reg.open() as f:
        r = csv.reader(f)
        for row in r:
            if not row or (row[0] and row[0].startswith("timestamp")):
                continue
            if len(row) < 4:
                continue
            s = (row[1] or "").strip()
            m = (row[2] or "").strip()
            p = (row[3] or "").strip()
            if s == strategy and m == mode and p == summary_path:
                return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, help="Nombre de la estrategia (ej. V2, PerlaNegra)")
    ap.add_argument("--pattern", required=True, help="Glob de summaries (ej. 'reports/summary_v2_*.json')")
    ap.add_argument("--registry", default="registry_runs.csv")
    args = ap.parse_args()

    summaries = pick_last_two(args.pattern)
    if len(summaries) < 2:
        print(f"⚠️ No hay suficientes summaries para {args.strategy} con patrón {args.pattern}")
        return

    live, freeze = summaries[0], summaries[1]
    reg = Path(args.registry)
    reg.parent.mkdir(parents=True, exist_ok=True)

    wrote = False
    # append evitando duplicar si ya existe la misma fila
    with reg.open("a", newline="") as f:
        w = csv.writer(f)
        if not exists_in_registry(reg, args.strategy, "freeze", freeze):
            w.writerow([now(), args.strategy, "freeze", freeze, wf_path(freeze)])
            wrote = True
        if not exists_in_registry(reg, args.strategy, "live", live):
            w.writerow([now(), args.strategy, "live", live, wf_path(live)])
            wrote = True

    if wrote:
        print(f"✅ Registradas {args.strategy} (freeze & live) en {args.registry}")
    else:
        print("ℹ️ Nada que registrar (ya existen las entradas más recientes).")

if __name__ == "__main__":
    main()