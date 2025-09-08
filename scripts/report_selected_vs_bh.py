#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toma microgrid_selection.json y kpis_grid.csv y arma un informe por ventana
para (horizon, threshold) elegido: ROI, B&H, Edge, PF, WR, Sortino, MDD.
"""

import argparse, json, os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kpis_csv", required=True)
    ap.add_argument("--selection_json", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    with open(args.selection_json) as f:
        sel = json.load(f)

    if sel.get("status") not in ("full_coverage","no_full_coverage"):
        raise SystemExit("No hay selección válida en el JSON (probablemente no hubo candidatos).")

    chosen = sel["chosen"]
    H = int(chosen["horizon"])
    T = float(chosen["threshold"])
    asset = sel.get("asset","BTC-USD")

    df = pd.read_csv(args.kpis_csv)
    df = df[(df["asset"]==asset) & (df["horizon"]==H) & (df["threshold"]==T)].copy()
    df = df.sort_values("window")

    lines = ["# Desempeño por ventana — Config elegida", ""]
    lines.append(f"- **Asset**: {asset}")
    lines.append(f"- **Horizon**: {H}")
    lines.append(f"- **Threshold**: {T:.2f}")
    lines.append("")
    lines.append("| Ventana | Trades | PF | WR | Sortino | MDD | ROI | B&H | Edge |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['window']} | {int(r['trades'])} | {r['pf']:.2f} | {r['wr']:.2%} | {r['sortino']:.2f} | "
            f"{r['mdd']:.2f} | {r['roi']:.2%} | {r['roi_bh']:.2%} | {r['edge_vs_bh']:.2%} |"
        )

    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines))

    print(f"[OK] Informe escrito en {args.out_md}")

if __name__ == "__main__":
    main()