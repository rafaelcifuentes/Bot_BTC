#!/usr/bin/env python3
import sys, glob, os, pathlib
import pandas as pd

def read_kpi(path: str) -> dict:
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()
    return {
        "file": path,
        "net_btc_ratio": float(row.get("net_btc_ratio", "nan")),
        "mdd_vs_hodl_ratio": float(row.get("mdd_vs_hodl_ratio", "nan")),
        "flips_per_year": float(row.get("flips_per_year", "nan")),
        "flips_total": float(row.get("flips_total", "nan")) if "flips_total" in row else float("nan"),
    }

def main():
    if len(sys.argv) < 3:
        print("Usage: collect_v2_gate.py <BASE_KPI_CSV> <REPORTS_DIR>", file=sys.stderr)
        sys.exit(2)

    base_csv = sys.argv[1]
    reports_dir = sys.argv[2]

    base = read_kpi(base_csv)
    base_sats = base["net_btc_ratio"]
    base_mdd  = base["mdd_vs_hodl_ratio"]
    base_fpy  = base["flips_per_year"]

    pat = os.path.join(reports_dir, "*_kpis__*.csv")
    files = sorted(glob.glob(pat))
    rows = []
    for f in files:
        try:
            cand = read_kpi(f)
        except Exception as e:
            print(f"[SKIP] {f} ({e})", file=sys.stderr); continue
        name = pathlib.Path(f).name
        suffix = name.split("__", 1)[1].replace(".csv", "") if "__" in name else name.replace(".csv", "")
        lift = (cand["net_btc_ratio"] / base_sats - 1.0) * 100.0
        mdd_delta = cand["mdd_vs_hodl_ratio"] - base_mdd
        fpy_ok = (cand["flips_per_year"] <= 2 * base_fpy) or (lift >= 5.0)
        risk_ok = cand["mdd_vs_hodl_ratio"] <= base_mdd
        pass_gate = (lift >= 5.0) and risk_ok and fpy_ok

        rows.append({
            "suffix": suffix,
            "netbtc_cand": cand["net_btc_ratio"],
            "netbtc_base": base_sats,
            "lift_pct": lift,
            "mdd_cand": cand["mdd_vs_hodl_ratio"],
            "mdd_base": base_mdd,
            "mdd_delta": mdd_delta,
            "fpy_cand": cand["flips_per_year"],
            "fpy_base": base_fpy,
            "risk_ok": risk_ok,
            "fpy_ok": fpy_ok,
            "GATE_PASS": pass_gate,
            "file": f,
        })

    os.makedirs(reports_dir, exist_ok=True)
    out_csv = os.path.join(reports_dir, "v2_h1_gate_summary.csv")
    out_md  = os.path.join(reports_dir, "v2_h1_gate_summary.md")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    rows_sorted = sorted(rows, key=lambda r: r["lift_pct"], reverse=True)
    with open(out_md, "w") as md:
        md.write("# V2.0 — Gate Summary (H1 2025)\n\n")
        md.write(f"- BASE: {base_csv}\n")
        md.write(f"- Reports dir: {reports_dir}\n\n")
        md.write("| Suffix | NetBTC CAND | NetBTC BASE | Lift % | MDD CAND | MDD BASE | ΔMDD | FPY CAND | FPY BASE | Risk OK | FPY OK | PASS |\n")
        md.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|\n")
        for r in rows_sorted:
            md.write(f"| {r['suffix']} | {r['netbtc_cand']:.6f} | {r['netbtc_base']:.6f} | {r['lift_pct']:.2f}% | {r['mdd_cand']:.6f} | {r['mdd_base']:.6f} | {r['mdd_delta']:.6f} | {r['fpy_cand']:.2f} | {r['fpy_base']:.2f} | {'✅' if r['risk_ok'] else '❌'} | {'✅' if r['fpy_ok'] else '❌'} | {'✅' if r['GATE_PASS'] else '❌'} |\n")
        md.write("\n")
    print(f"[OK] Wrote {out_csv}\n[OK] Wrote {out_md}")

if __name__ == "__main__":
    main()
