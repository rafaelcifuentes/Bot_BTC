#!/usr/bin/env python3
"""
Comparador de KPIs (PF, MDD, Vol) entre dos runs (p.ej., ATR 0.07 vs 0.08).

Uso:
  python scripts/mini_accum/quick_compare_kpis.py <kpis_07.csv> <kpis_08.csv>

También funciona si el orden es inverso; puedes especificar con --old y --new.

Salida:
  Tabla con PF_base/overlay, MDD_base/overlay, Vol_base/overlay de cada run,
  y las diferencias (Δ) y variaciones (%) tomando NEW - OLD.
"""
import argparse
import os
import pandas as pd

def load_metrics(path: str) -> dict:
    k = pd.read_csv(path).iloc[0]
    # Normaliza keys a minúscula para evitar problemas
    cols = {c.lower(): c for c in k.index}
    g = lambda name: float(k[cols[name]])
    return {
        "pf_base": g("pf_base"),
        "pf_overlay": g("pf_overlay"),
        "mdd_base": abs(g("mdd_base")),
        "mdd_overlay": abs(g("mdd_overlay")),
        "vol_base": g("vol_base"),
        "vol_overlay": g("vol_overlay"),
        "net_base": g("net_base"),
        "net_overlay": g("net_overlay"),
        "_rows": int(k.get("rows", 0)) if "rows" in cols else None,
        "_file": os.path.basename(path),
    }

def fmt_pct(x):
    return f"{x*100:.4f}%"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old", nargs="?", help="CSV KPIs 'OLD' (p.ej., atr0.07)")
    ap.add_argument("new", nargs="?", help="CSV KPIs 'NEW' (p.ej., atr0.08)")
    ap.add_argument("--old", dest="old_kw", help="Alias explícito para OLD")
    ap.add_argument("--new", dest="new_kw", help="Alias explícito para NEW")
    args = ap.parse_args()

    old_path = args.old_kw or args.old
    new_path = args.new_kw or args.new
    if not old_path or not new_path:
        ap.error("Debes pasar dos archivos: quick_compare_kpis.py <old.csv> <new.csv>")

    m_old = load_metrics(old_path)
    m_new = load_metrics(new_path)

    # Deltas (NEW - OLD)
    def delta(a, b): return b - a
    def delta_pct(a, b):
        if a == 0: 
            return float("inf") if b != 0 else 0.0
        return (b - a) / abs(a)

    rows = []
    # PF (overlay)
    rows.append(["PF (overlay)", f"{m_old['pf_overlay']:.6f}", f"{m_new['pf_overlay']:.6f}",
                 f"{delta(m_old['pf_overlay'], m_new['pf_overlay']):+.6f}",
                 f"{delta_pct(m_old['pf_overlay'], m_new['pf_overlay'])*100:+.2f}%"])
    # PF (base) solo informativo
    rows.append(["PF (base)", f"{m_old['pf_base']:.6f}", f"{m_new['pf_base']:.6f}",
                 f"{delta(m_old['pf_base'], m_new['pf_base']):+.6f}",
                 f"{delta_pct(m_old['pf_base'], m_new['pf_base'])*100:+.2f}%"])

    # MDD (overlay) — menor es mejor
    rows.append(["MDD (overlay)", fmt_pct(m_old["mdd_overlay"]), fmt_pct(m_new["mdd_overlay"]),
                 fmt_pct(delta(m_old["mdd_overlay"], m_new["mdd_overlay"])),
                 f"{delta_pct(m_old['mdd_overlay'], m_new['mdd_overlay'])*100:+.2f}%"])
    rows.append(["MDD (base)", fmt_pct(m_old["mdd_base"]), fmt_pct(m_new["mdd_base"]),
                 fmt_pct(delta(m_old["mdd_base"], m_new["mdd_base"])),
                 f"{delta_pct(m_old['mdd_base'], m_new['mdd_base'])*100:+.2f}%"])

    # Vol (overlay) — menor es mejor
    rows.append(["Vol (overlay)", f"{m_old['vol_overlay']:.12f}", f"{m_new['vol_overlay']:.12f}",
                 f"{delta(m_old['vol_overlay'], m_new['vol_overlay']):+.12f}",
                 f"{delta_pct(m_old['vol_overlay'], m_new['vol_overlay'])*100:+.2f}%"])
    rows.append(["Vol (base)", f"{m_old['vol_base']:.12f}", f"{m_new['vol_base']:.12f}",
                 f"{delta(m_old['vol_base'], m_new['vol_base']):+.12f}",
                 f"{delta_pct(m_old['vol_base'], m_new['vol_base'])*100:+.2f}%"])

    # Net (overlay) — informativo (no decide)
    rows.append(["Net (overlay)", f"{m_old['net_overlay']:.12f}", f"{m_new['net_overlay']:.12f}",
                 f"{delta(m_old['net_overlay'], m_new['net_overlay']):+.12f}",
                 f"{delta_pct(m_old['net_overlay'], m_new['net_overlay'])*100:+.2f}%"])
    rows.append(["Net (base)", f"{m_old['net_base']:.12f}", f"{m_new['net_base']:.12f}",
                 f"{delta(m_old['net_base'], m_new['net_base']):+.12f}",
                 f"{delta_pct(m_old['net_base'], m_new['net_base'])*100:+.2f}%"])

    # Render
    print("\n== quick_compare_kpis ==")
    print(f"OLD: {m_old['_file']}")
    print(f"NEW: {m_new['_file']}\n")

    # ancho de columnas
    colw = [18, 18, 18, 16, 12]
    header = ["Métrica", "OLD", "NEW", "Δ (NEW-OLD)", "Δ%"]
    def fmt_row(cells):
        return "".join(str(cells[i]).ljust(colw[i]) for i in range(len(colw)))

    print(fmt_row(header))
    print("-" * sum(colw))
    for r in rows:
        print(fmt_row(r))

    # Reglas PASS informativas para NEW (no decide cambio productivo aquí)
    pass_cond = (m_new["pf_overlay"] >= 0.90 * m_new["pf_base"]
                 and m_new["mdd_overlay"] <= m_new["mdd_base"]
                 and m_new["vol_overlay"] <= m_new["vol_base"])
    print("\nRegla PASS NEW (informativa):",
          "PASS" if pass_cond else "FAIL")
