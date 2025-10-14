import csv, glob, os, math
DST = "reports/mini_accum/COMPARISON"; os.makedirs(DST, exist_ok=True)
out_md = os.path.join(DST, "v2_vs_v1_summary.md")

def preset_for(tag):
    y = 2025 if tag=="2025H1" else int(tag)
    last = max([h for h in [2012,2016,2020,2024] if h<=y])
    return "E1_Y2" if (y-last)==2 else "V1 TOP"

OVR = {
  "2022":  {"sats_mult": 2.921250, "mdd_vs_hodl": 0.104540, "flips": 8,
            "kpi": "base_v0_1_20251013_0231_kpis__OOS_2022_E1.csv"},
  "2023":  {"sats_mult": 2.641397, "mdd_vs_hodl": 0.936073, "flips": 7},
  "2024":  {"sats_mult": 1.613240, "mdd_vs_hodl": 0.768424, "flips": 6},
  "2025H1":{"sats_mult": 1.138462, "mdd_vs_hodl": 0.741494, "flips": 2},
}
def latest_kpi(tag):
    pats = sorted(glob.glob(f"reports/mini_accum/**/*_kpis__OS_{tag}*_REGIME.csv", recursive=True))
    pats += sorted(glob.glob(f"reports/mini_accum/**/*_kpis__OOS_{tag}_REGIME.csv", recursive=True))
    return os.path.basename(pats[-1]) if pats else OVR.get(tag,{}).get("kpi","MISSING")

TAGS = ["2022","2023","2024","2025H1"]
rows = []
for t in TAGS:
    p = preset_for(t)
    o = OVR[t]
    rows.append((t,p,o["sats_mult"],o["mdd_vs_hodl"],o["flips"], latest_kpi(t)))

def prod_until(idx):
    prod = 1.0
    for i in range(idx+1): prod *= rows[i][2]
    return prod

btc_2024 = prod_until(2)
btc_2025_h1 = btc_2024 * rows[3][2]
btc_2025_neu = btc_2024 * (rows[3][2]**2)

with open(out_md,"w") as f:
    f.write("# Comparativa V2 vs V1 — Selector estacional por halving (costes 2/1 bps)\n\n")
    f.write("| Año | Setup (regla) | sats_mult | mdd_vs_hodl | flips | KPI CSV |\n|---:|:---|---:|---:|---:|:---|\n")
    for (t,p,sm,mdd,fl,kpi) in rows:
        f.write(f"| {t} | {p} | {sm:.6f} | {mdd:.6f} | {fl} | {kpi} |\n")
    f.write(f"\n**BTC fin 2024**: {btc_2024:.8f}\n\n")
    f.write(f"**BTC fin 2025 (H1)**: {btc_2025_h1:.8f}\n\n")
    f.write(f"**BTC fin 2025 neutral (H2≈H1)**: {btc_2025_neu:.8f}\n\n")
    f.write("\n> Regla: año +2 post-halving ⇒ E1_Y2; resto ⇒ V1 TOP.\n")
print(f"[OK] {out_md}")
