import csv, glob, os, sys

TAGS = ["OOS_2023_REGIME","OOS_2024_REGIME","OOS_2025H1_REGIME"]

def last(path_glob):
    xs = sorted(glob.glob(path_glob))
    return xs[-1] if xs else None

def pick(row, *cands, default=None):
    for k in cands:
        if k in row and row[k] not in ("", None):
            return row[k]
    return default

def read_kpis(path):
    if not path or os.path.getsize(path)==0:
        return {}
    rows = list(csv.DictReader(open(path, newline="")))
    return rows[-1] if rows else {}

def count_flips(path):
    if not path or not os.path.exists(path) or os.path.getsize(path)==0:
        return None
    with open(path, "r") as fh:
        n = sum(1 for _ in fh) - 1
    return max(n,0)

print("== CONSOLIDADO OOS ==")
for tag in TAGS:
    k = last(f"reports/mini_accum/*kpis__{tag}.csv")
    f = last(f"reports/mini_accum/*flips__{tag}.csv")
    e = last(f"reports/mini_accum/*equity__{tag}.csv")
    s = last(f"reports/mini_accum/*summary__{tag}.md")

    r = read_kpis(k)
    sats_mult   = pick(r, "sats_mult","sats_mult_btc","btc_mult","sats_mult_ratio")
    netbtc      = pick(r, "netBTC","net_btc","net_btc_ratio","net_btc_mult")
    mdd_vs_hodl = pick(r, "mdd_vs_hodl","mdd_vs_hodl_ratio","mdd_ratio")
    fpy         = pick(r, "fpy","flips_per_year")
    flips_kpis  = pick(r, "flips_total","flips")
    flips_csv   = count_flips(f)

    print(f"\n[{tag}]")
    print("KPIs   :", k or "<no encontrado>")
    print("Flips  :", f or "<no encontrado>")
    print("Equity :", e or "<no encontrado>")
    print("Summary:", s or "<no encontrado>")
    print("  sats_mult   =", sats_mult)
    print("  netBTC      =", netbtc)
    print("  mdd_vs_hodl =", mdd_vs_hodl)
    print("  fpy         =", fpy)
    if flips_kpis is not None:
        print("  flips_total(KPIs) =", flips_kpis)
    if flips_csv is not None:
        print("  flips_total(CSV)  =", flips_csv)
    # Semáforo flips
    flips_ok = (flips_csv or 0) > 0
    print("  CHECK flips>0     =", "PASS" if flips_ok else "FAIL")
