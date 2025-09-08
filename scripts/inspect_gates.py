#!/usr/bin/env python3
# scripts/inspect_gates.py
import argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kpis_csv", required=True)
    ap.add_argument("--windows", nargs="+", required=True)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--gate_pf", type=float, default=1.6)
    ap.add_argument("--gate_wr", type=float, default=0.60)
    ap.add_argument("--gate_trades", type=int, default=30)
    ap.add_argument("--out_md", default="reports/gates_summary.md")
    args = ap.parse_args()

    df = pd.read_csv(args.kpis_csv)
    df = df[(df["asset"]==args.asset) & (df["window"].isin(args.windows))].copy()

    gates = (df["pf"]>=args.gate_pf) & (df["wr"]>=args.gate_wr) & (df["trades"]>=args.gate_trades)
    ok = df[gates].copy()
    fail = df[~gates].copy()

    # Mejor por ventana (PF máx)
    best_pf = df.loc[df.groupby("window")["pf"].idxmax()].copy()
    best_wr = df.loc[df.groupby("window")["wr"].idxmax()].copy()
    best_edge = df.loc[df.groupby("window")["edge_vs_bh"].idxmax()].copy()

    lines = []
    lines.append(f"# Gates summary — {args.asset}")
    lines.append(f"Gates: PF≥{args.gate_pf} | WR≥{args.gate_wr:.2f} | trades≥{args.gate_trades}")
    lines.append("")
    lines.append(f"Total filas analizadas: {len(df)}  |  Candidatos que pasan gates: {len(ok)}")
    lines.append("")
    for w in args.windows:
        sub = df[df["window"]==w]
        lines.append(f"## {w}")
        lines.append(f"Filas: {len(sub)}  |  Pasa gates: {(sub['pf']>=args.gate_pf).sum() & (sub['wr']>=args.gate_wr).sum()}")
        if not sub.empty:
            bp = best_pf[best_pf["window"]==w].iloc[0]
            bw = best_wr[best_wr["window"]==w].iloc[0]
            be = best_edge[best_edge["window"]==w].iloc[0]
            lines.append("")
            lines.append("**Mejor PF** (aunque falle gates):")
            lines.append(f"- h={bp.horizon} | th={bp.threshold} | trades={bp.trades} | PF={bp.pf:.3f} | WR={bp.wr:.3f} | MDD={bp.mdd:.3f} | Sortino={bp.sortino:.3f} | Edge_vs_BH={bp.edge_vs_bh:.3f}")
            lines.append("**Mejor WR**:")
            lines.append(f"- h={bw.horizon} | th={bw.threshold} | trades={bw.trades} | PF={bw.pf:.3f} | WR={bw.wr:.3f} | Edge_vs_BH={bw.edge_vs_bh:.3f}")
            lines.append("**Mejor Edge_vs_BH**:")
            lines.append(f"- h={be.horizon} | th={be.threshold} | trades={be.trades} | PF={be.pf:.3f} | WR={be.wr:.3f} | Edge_vs_BH={be.edge_vs_bh:.3f}")
        lines.append("")

    with open(args.out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"[OK] Escrito {args.out_md}")

if __name__ == "__main__":
    main()