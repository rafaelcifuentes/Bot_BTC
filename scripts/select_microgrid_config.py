#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selecciona la configuración (horizon, threshold) más estable a través de varias ventanas,
aplicando gates (PF, WR, trades). Produce un JSON con la selección y un MD con el ranking.

Requiere un CSV con columnas:
window,asset,horizon,threshold,trades,pf,wr,mdd,sortino,roi,roi_bh,edge_vs_bh,avg_trade
"""

import argparse, json, os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kpis_csv", required=True)
    ap.add_argument("--windows", nargs="+", required=True, help="Nombres exactos de las ventanas (ej. 2022H2 2023Q4 2024H1)")
    ap.add_argument("--asset", default="BTC-USD")
    ap.add_argument("--gate_pf", type=float, default=1.6)
    ap.add_argument("--gate_wr", type=float, default=0.60)
    ap.add_argument("--gate_trades", type=int, default=30)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--top_n", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.kpis_csv)

    needed = {"window","asset","horizon","threshold","trades","pf","wr","mdd","sortino","roi","roi_bh","edge_vs_bh","avg_trade"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {args.kpis_csv}: {sorted(missing)}")

    # Filtrar por activo y ventanas de interés
    df = df[(df["asset"] == args.asset) & (df["window"].isin(args.windows))].copy()

    # Aplicar gates por fila (ventana)
    filt = (df["pf"] >= args.gate_pf) & (df["wr"] >= args.gate_wr) & (df["trades"] >= args.gate_trades)
    df_g = df[filt].copy()

    # Cobertura por (horizon, threshold): debe cubrir TODAS las ventanas requeridas
    group_cols = ["horizon","threshold"]
    coverage = (df_g.groupby(group_cols)["window"]
                  .nunique()
                  .reset_index(name="n_windows"))
    coverage_full = coverage[coverage["n_windows"] == len(args.windows)]
    if coverage_full.empty:
        # Si nada pasa los gates en TODAS las ventanas, intentamos al menos 2/3, etc. (degradación suave)
        best_cover = coverage["n_windows"].max() if not coverage.empty else 0
        status = "no_full_coverage"
        candidates = coverage[coverage["n_windows"] == best_cover][group_cols] if best_cover>0 else pd.DataFrame(columns=group_cols)
    else:
        status = "full_coverage"
        candidates = coverage_full[group_cols]

    # Calcular métricas agregadas por (horizon, threshold)
    agg = (df[df[group_cols].apply(tuple, axis=1).isin(candidates.apply(tuple, axis=1))]
           if not candidates.empty else df.iloc[0:0]).copy()

    if not agg.empty:
        summary = (agg.groupby(group_cols)
                       .agg(n_windows=("window","nunique"),
                            trades_sum=("trades","sum"),
                            pf_med=("pf","median"),
                            pf_min=("pf","min"),
                            wr_med=("wr","median"),
                            sortino_med=("sortino","median"),
                            mdd_med=("mdd","median"),
                            edge_med=("edge_vs_bh","median"),
                            roi_med=("roi","median"))
                       .reset_index())
        # Ranking: estabilidad (pf_med) → calidad (sortino_med) → edge vs B&H
        summary = summary.sort_values(
            ["n_windows","pf_med","sortino_med","edge_med"],
            ascending=[False, False, False, False]
        )
    else:
        summary = pd.DataFrame(columns=group_cols + ["n_windows","trades_sum","pf_med","pf_min","wr_med","sortino_med","mdd_med","edge_med","roi_med"])

    selection = {}
    if not summary.empty:
        top = summary.iloc[0].to_dict()
        selection = {
            "status": status,
            "asset": args.asset,
            "windows": args.windows,
            "gate_pf": args.gate_pf,
            "gate_wr": args.gate_wr,
            "gate_trades": args.gate_trades,
            "chosen": {
                "horizon": int(top["horizon"]),
                "threshold": float(top["threshold"]),
                "n_windows": int(top["n_windows"]),
                "trades_sum": int(top["trades_sum"]),
                "pf_med": float(top["pf_med"]),
                "pf_min": float(top["pf_min"]),
                "wr_med": float(top["wr_med"]),
                "sortino_med": float(top["sortino_med"]),
                "mdd_med": float(top["mdd_med"]),
                "edge_med": float(top["edge_med"]),
                "roi_med": float(top["roi_med"]),
            }
        }
    else:
        selection = {
            "status": "no_candidates",
            "reason": "Ninguna config pasó gates ni siquiera parcialmente.",
            "asset": args.asset,
            "windows": args.windows,
            "gate_pf": args.gate_pf,
            "gate_wr": args.gate_wr,
            "gate_trades": args.gate_trades,
        }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(selection, f, indent=2)

    # MD corto
    lines = ["# Selección Micro-grid", ""]
    lines.append(f"- **Asset**: {args.asset}")
    lines.append(f"- **Ventanas**: {', '.join(args.windows)}")
    lines.append(f"- **Gates**: PF≥{args.gate_pf}, WR≥{args.gate_wr:.0%}, trades≥{args.gate_trades}")
    lines.append(f"- **Cobertura requerida**: {len(args.windows)} ventanas")
    lines.append("")
    lines.append(f"**Status**: {selection['status']}")
    lines.append("")
    if selection.get("chosen"):
        c = selection["chosen"]
        lines += [
            "## Config elegida",
            f"- `horizon` = **{c['horizon']}**",
            f"- `threshold` = **{c['threshold']:.2f}**",
            f"- Cobertura = **{c['n_windows']}** ventanas",
            f"- Trades (suma) = **{c['trades_sum']}**",
            f"- PF (med) = **{c['pf_med']:.2f}**  | PF (min) = **{c['pf_min']:.2f}**",
            f"- WR (med) = **{c['wr_med']:.2%}**",
            f"- Sortino (med) = **{c['sortino_med']:.2f}**",
            f"- MDD (med) = **{c['mdd_med']:.2f}**",
            f"- ROI (med) = **{c['roi_med']:.2%}**  | Edge vs B&H (med) = **{c['edge_med']:.2%}**",
            ""
        ]
        lines.append("## Top candidatos")
        topN = summary.head(args.top_n)
        for _, r in topN.iterrows():
            lines.append(
                f"- H={int(r['horizon'])} thr={r['threshold']:.2f} | cov={int(r['n_windows'])} | "
                f"PF_med={r['pf_med']:.2f} (min={r['pf_min']:.2f}) | WR_med={r['wr_med']:.2%} | "
                f"Sortino_med={r['sortino_med']:.2f} | Edge_med={r['edge_med']:.2%} | "
                f"MDD_med={r['mdd_med']:.2f} | Trades_sum={int(r['trades_sum'])}"
            )
    else:
        lines.append("> No hubo candidatos con cobertura suficiente. Considera relajar gates o revisar señales.")

    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines))

    print(f"[OK] Selección escrita en {args.out_json} y {args.out_md}")

if __name__ == "__main__":
    main()