#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplica stress de fricciones ±(bps por lado) a sats_mult por ventana/config usando un modelo multiplicativo:
sats_mult_stressed = sats_mult * (1 - 2*Δbps/10000) ** flips_total

Asume que el run ya incluye una fricción base; aquí probamos Δbps alrededor (positivo/negativo).
"""

import argparse, pandas as pd
from pathlib import Path

def stress(df: pd.DataFrame, delta_bps: int) -> pd.DataFrame:
    d = df.copy()
    # Δbps por lado → por roundtrip son 2*Δbps (buy+sell).
    factor = (1.0 - (2.0 * delta_bps) / 10000.0)
    d["sats_mult_stress"] = d["sats_mult"] * (factor ** d["flips_total"].clip(lower=0))
    d["delta_bps_side"] = delta_bps
    return d

def appendix_md(stressed: pd.DataFrame) -> str:
    # Resumen por config (medianas OOS)
    grp = stressed.groupby(["config_id","delta_bps_side"])["sats_mult_stress"].median().reset_index()
    # Tabla pivot con columnas por Δbps
    piv = grp.pivot(index="config_id", columns="delta_bps_side", values="sats_mult_stress")
    piv = piv.sort_index(axis=1)
    md = ["## Stress de costes (mediana sats_mult por config)"]
    md.append(piv.to_markdown())
    md.append("")
    # Robustez de decisión: signo y orden relativo preservados
    # (Chequeo simple: ranking por Δbps vs ranking base Δ=0)
    if 0 in piv.columns:
        base_rank = piv[0].rank(ascending=False, method="average")
        ok_rows = []
        for col in piv.columns:
            r = piv[col].rank(ascending=False, method="average")
            spearman_like = r.corr(base_rank, method="spearman")
            ok_rows.append((col, spearman_like))
        md.append("**Correlación de ranking (Spearman) vs Δbps=0**")
        md.append(pd.DataFrame(ok_rows, columns=["delta_bps_side","spearman_corr"]).to_markdown(index=False))
        md.append("")
    return "\n".join(md)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_md_append", required=True)
    ap.add_argument("--bps", type=int, nargs="+", default=[5,10,20])
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    need = {"config_id","sats_mult","flips_total"}
    if not need.issubset(df.columns):
        raise ValueError(f"Faltan columnas {need - set(df.columns)} en {args.summary_csv}")

    # Añadimos Δ=0 para referencia
    outs = [stress(df, 0)]
    for b in args.bps:
        outs += [stress(df, +b), stress(df, -b)]
    all_s = pd.concat(outs, ignore_index=True)

    md = appendix_md(all_s)
    p = Path(args.out_md_append)
    txt = p.read_text(encoding="utf-8") if p.exists() else ""
    p.write_text(txt + ("\n\n" if txt else "") + md, encoding="utf-8")
    print("[OK] Stress de costes añadido al Roadmap.")

if __name__ == "__main__":
    main()