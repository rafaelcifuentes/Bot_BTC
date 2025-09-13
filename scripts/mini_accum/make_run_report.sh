#!/usr/bin/env bash
set -Eeuo pipefail

KPI=$(ls -t reports/mini_accum/*_kpis__*.csv | head -1)
FLP=$(ls -t reports/mini_accum/*_flips__*.csv | head -1)
[ -f "$KPI" ] && [ -f "$FLP" ] || { echo "No encuentro KPIs/FLIPS en reports/mini_accum"; exit 1; }

TAG=$(basename "$KPI" | sed -E 's/^base_v0_1_([0-9]{8}_[0-9]{4})_kpis__.*/\1/')
HASH=$(git rev-parse --short=12 HEAD 2>/dev/null || echo "NO-GIT")
OUT="docs/runs/${TAG}_PASS.md"

python3 - "$KPI" "$FLP" "$OUT" "$HASH" <<'PY'
import sys, csv, os, pandas as pd
kpi, flp, out, gh = sys.argv[1:5]

def sniff(p):
    import csv
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        s=f.read(4096)
    try: d=csv.Sniffer().sniff(s, delimiters=[',',';','\t','|']); return d.delimiter
    except: return ','

sep_k, sep_f = sniff(kpi), sniff(flp)
K = pd.read_csv(kpi, sep=sep_k)
F = pd.read_csv(flp, sep=sep_f)

row = K.iloc[-1]
def get(name, idx): 
    return float(row.get(name, row.iloc[idx]))

net = get('net_btc_ratio', 0)
mdd = get('mdd_vs_hodl_ratio', 3)
fpy = get('flips_per_year', 6)

# flips compactos (últimos 12)
last = F.tail(12)

md = []
md += [f"# PASS report — {os.path.basename(kpi)}"]
md += [f"- Commit: `{gh}`"]
md += [""]
md += [f"**KPIs**  netBTC=`{net:.4f}`  mdd_vs_HODL=`{mdd:.3f}`  fpy=`{fpy:.2f}`"]
md += [""]
md += ["**Últimos flips** (máx 12):"]
if not last.empty:
    tcol = last.columns[0]
    cols = last.columns[:4]
    md += ["```"]
    for _, r in last.iterrows():
        md.append(",".join(str(r[c]) for c in cols))
    md += ["```"]
else:
    md += ["(no flips)"]
md += [""]
md += ["**Archivos**:"]
md += [f"- KPI: `{kpi}`"]
md += [f"- FLIPS: `{flp}`"]
open(out, 'w', encoding='utf-8').write("\n".join(md))
print(f"[OK] {out}")
PY
echo "[OK] $OUT"
