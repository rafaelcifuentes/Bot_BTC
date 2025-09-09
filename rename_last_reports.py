#!/usr/bin/env python3
import os, sys, glob

SUFFIX = None
# 1) intenta leer desde línea de comandos: --suffix foo
args = sys.argv[1:]
if "--suffix" in args:
    i = args.index("--suffix")
    if i+1 < len(args):
        SUFFIX = args[i+1]

# 2) o desde env var
if SUFFIX is None:
    SUFFIX = os.environ.get("REPORT_SUFFIX")

if not SUFFIX:
    print("usage: python rename_last_reports.py --suffix <SUFFIX>")
    print("   or: REPORT_SUFFIX=<SUFFIX> python rename_last_reports.py")
    sys.exit(2)

patterns = [
    "reports/mini_accum/base_v0_1_*_equity.csv",
    "reports/mini_accum/base_v0_1_*_kpis.csv",
    "reports/mini_accum/base_v0_1_*_summary.md",
]

renamed = 0
for pat in patterns:
    files = sorted(glob.glob(pat))
    if not files:
        continue
    src = files[-1]
    head, tail = os.path.split(src)
    name, ext = os.path.splitext(tail)
    dst = os.path.join(head, f"{name}__{SUFFIX}{ext}")
    try:
        os.replace(src, dst)
        print(f"[RENAMED] {tail} -> {os.path.basename(dst)}")
        renamed += 1
    except Exception as e:
        print(f"[WARN] Falló rename {tail}: {e}")

if renamed == 0:
    print("[WARN] No se renombró ningún archivo. ¿Ejecutaste el backtest antes?")
    sys.exit(1)
