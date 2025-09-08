#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convierte logs de Perla Negra V2 (weekly v2) a CSV resumido.
Uso:
  python tools/log2csv_perla_v2.py logs/perla_negra_v2_*.log > reports/perla_negra_v2_summary.csv
"""

import sys, os, re, csv, glob

PAT_RUN = re.compile(r'perla_negra_v2_(\d{8}_\d{4})\.log')

# Ejemplos de líneas a parsear:
# Solo LONG  → Net 1701.02, PF 5.03, WR 9.87%, Trades 446, MDD 121.57, Score 13.99
# Solo SHORT → Net 1538.65, PF 2.47, WR 28.25%, Trades 446, MDD 113.00, Score 13.62
# Baseline   → wL 0.60, wS 0.40 | Net 1640.32, PF 3.41, WR 38.12%, Trades 446, MDD 74.32, Score 22.07
# Selected   → wL 0.35, wS 0.55 | Net 1434.80, PF 2.98, WR 38.12%, Trades 446, MDD 45.86, Score 31.28 | feasible=True | cap=118.65

PAT_SOLO = re.compile(
    r'^\s*Solo\s+(LONG|SHORT)\s+→\s+Net\s+([-\d\.]+),\s+PF\s+([-\d\.]+),\s+WR\s+([-\d\.]+)%,\s+Trades\s+(\d+),\s+MDD\s+([-\d\.]+),\s+Score\s+([-\d\.]+)',
    re.IGNORECASE
)

PAT_BASE = re.compile(
    r'^\s*Baseline\s+→\s+wL\s+([0-9\.]+),\s+wS\s+([0-9\.]+)\s+\|\s+Net\s+([-\d\.]+),\s+PF\s+([-\d\.]+),\s+WR\s+([-\d\.]+)%,\s+Trades\s+(\d+),\s+MDD\s+([-\d\.]+),\s+Score\s+([-\d\.]+)',
    re.IGNORECASE
)

PAT_SEL = re.compile(
    r'^\s*Selected\s+→\s+wL\s+([0-9\.]+),\s+wS\s+([0-9\.]+)\s+\|\s+Net\s+([-\d\.]+),\s+PF\s+([-\d\.]+),\s+WR\s+([-\d\.]+)%,\s+Trades\s+(\d+),\s+MDD\s+([-\d\.]+),\s+Score\s+([-\d\.]+)',
    re.IGNORECASE
)

def parse_file(path):
    run_id = "-"
    m = PAT_RUN.search(os.path.basename(path))
    if m:
        run_id = m.group(1)

    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()

                m1 = PAT_SOLO.search(s)
                if m1:
                    leg = f"Solo {m1.group(1).title()}"
                    net = float(m1.group(2))
                    pf = float(m1.group(3))
                    wr = float(m1.group(4))
                    trades = int(m1.group(5))
                    mdd = float(m1.group(6))
                    score = float(m1.group(7))
                    rows.append([run_id, leg, net, pf, wr, trades, mdd, score, "-"])
                    continue

                m2 = PAT_BASE.search(s)
                if m2:
                    wL = float(m2.group(1)); wS = float(m2.group(2))
                    net = float(m2.group(3))
                    pf = float(m2.group(4))
                    wr = float(m2.group(5))
                    trades = int(m2.group(6))
                    mdd = float(m2.group(7))
                    score = float(m2.group(8))
                    rows.append([run_id, "Baseline", net, pf, wr, trades, mdd, score, f"{wL:.2f}/{wS:.2f}"])
                    continue

                m3 = PAT_SEL.search(s)
                if m3:
                    wL = float(m3.group(1)); wS = float(m3.group(2))
                    net = float(m3.group(3))
                    pf = float(m3.group(4))
                    wr = float(m3.group(5))
                    trades = int(m3.group(6))
                    mdd = float(m3.group(7))
                    score = float(m3.group(8))
                    rows.append([run_id, "Selected", net, pf, wr, trades, mdd, score, f"{wL:.2f}/{wS:.2f}"])
                    continue
    except FileNotFoundError:
        pass

    return rows

def main():
    if len(sys.argv) < 2:
        print("Uso: python tools/log2csv_perla_v2.py logs/perla_negra_v2_*.log > reports/perla_negra_v2_summary.csv", file=sys.stderr)
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        paths.extend(glob.glob(arg))

    # Ordenar por nombre (timestamp incluido)
    paths = sorted(set(paths))

    out = csv.writer(sys.stdout)
    out.writerow(["run","leg","net","pf","wr","trades","mdd","score","weights"])

    for p in paths:
        for row in parse_file(p):
            out.writerow(row)

if __name__ == "__main__":
    main()

