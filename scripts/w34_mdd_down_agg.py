#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob
import pandas as pd
import numpy as np

# === Config ===
RAW_GLOBS = [
    "reports/w34_mdd_down/raw2/*.csv",   # preferido (tu última corrida)
    "reports/w34_mdd_down/raw/*.csv"     # fallback
]
OUT_DIR = "../reports/w34_mdd_down"

# Gates/umbrales (mismos que venimos usando)
CAP_MDD   = 0.0187     # 1.87%
PF_GATE   = 1.50       # para pass_both (freeze-level)
RELAX_PF  = 1.40       # para shortlist_relaxed
WR_GATE   = 60.0
TR_GATE   = 30

# === Utilidades ===
PAT = re.compile(r"""
    (?P<prefix>.*?)            # cualquier prefijo
    _?btc_
    T(?P<t>\d+(?:\.\d+)?)      # T0.60
    _SL(?P<sl>\d+(?:\.\d+)?)   # SL1.3
    _TP1(?P<tp1>\d+(?:\.\d+)?) # TP10.7 (-> 0.7)
    _P(?P<p>\d+(?:\.\d+)?)     # P0.8
    (?:_plus)?\.csv$
""", re.X)

def find_files():
    files = []
    for g in RAW_GLOBS:
        files.extend(glob.glob(g))
        if files:  # usamos el primer glob que tenga archivos
            break
    return sorted(files)

def parse_meta(path: str):
    base = os.path.basename(path)
    m = PAT.search(base)
    if not m:
        return None
    t   = float(m.group("t"))
    sl  = float(m.group("sl"))
    tp1 = float(m.group("tp1"))
    p   = float(m.group("p"))
    # freeze = prefijo hasta '_btc_'
    pref = m.group("prefix")
    # Normaliza freeze: quita trailing '_' si quedó
    freeze = pref.rstrip('_')
    return dict(file=base, freeze=freeze, t=t, sl=sl, tp1=tp1, p=p)

def pick60(path: str):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    # Esperamos: days, net, pf, win_rate, trades, mdd
    if 'days' in cols:
        sub = df[df[cols['days']] == 60]
        r = sub.iloc[0] if len(sub) else df.iloc[-1]
    else:
        r = df.iloc[-1]
    pf     = float(r[cols.get('pf','pf')])
    wr     = float(r[cols.get('win_rate','win_rate')])
    trades = int(float(r[cols.get('trades','trades')]))
    mdd    = abs(float(r[cols.get('mdd','mdd')]))
    return dict(pf=pf, wr=wr, trades=trades, mdd=mdd)

def fmt_pct(x):
    return f"{x*100:.2f}%"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = find_files()
    if not files:
        print("[WARN] Sin filas en raw; revisa la corrida anterior.")
        return

    rows = []
    bad  = 0
    for f in files:
        meta = parse_meta(f)
        if not meta:
            bad += 1
            continue
        try:
            r = pick60(f)
            rows.append({**meta, **r})
        except Exception:
            bad += 1

    if not rows:
        print("[WARN] No se pudieron leer métricas de 60d.")
        return

    df = pd.DataFrame(rows)
    # De-dup por (freeze,t,sl,tp1,p) para evitar duplicados _plus
    df = df.sort_values("file").drop_duplicates(subset=["freeze","t","sl","tp1","p"], keep="first")

    # pass_both por freeze (PF y MDD simultáneamente)
    df["pass_both"] = (df["pf"] >= PF_GATE) & (df["mdd"] <= CAP_MDD)

    # ---- Agregado por combo ----
    grp = df.groupby(["t","sl","tp1","p"]).agg(
        pf_med  = ("pf", "median"),
        wr_med  = ("wr", "median"),
        tr_med  = ("trades", "median"),
        mdd_med = ("mdd", "median"),
        pf_min  = ("pf", "min"),
        n       = ("freeze", "nunique"),
        pass_rate_both = ("pass_both", "mean"),
    ).reset_index()

    grp["pass_rate_both"] = grp["pass_rate_both"] * 100.0

    # Top 12 por MDD (desempate PF/WR/Trades desc)
    top = grp.sort_values(
        by=["mdd_med","pf_med","wr_med","tr_med"],
        ascending=[True, False, False, False]
    ).head(12)

    # Shortlists
    relaxed = grp[
        (grp["mdd_med"] <= CAP_MDD) &
        (grp["pf_med"]  >= RELAX_PF) &
        (grp["wr_med"]  >= WR_GATE) &
        (grp["tr_med"]  >= TR_GATE)
    ].copy()

    final = grp[
        (grp["pass_rate_both"] >= 60.0) &
        (grp["mdd_med"] <= CAP_MDD) &
        (grp["pf_med"]  >= RELAX_PF) &
        (grp["wr_med"]  >= WR_GATE) &
        (grp["tr_med"]  >= TR_GATE)
    ].copy()

    # Guardar CSVs
    grp.to_csv(f"{OUT_DIR}/agg_by_combo.csv", index=False)
    relaxed.to_csv(f"{OUT_DIR}/shortlist_relaxed.csv", index=False)
    final.to_csv(f"{OUT_DIR}/shortlist_final.csv", index=False)

    # ---- Console summary “bonito” ----
    def pretty(df0, cols_pct=("wr_med","mdd_med","pass_rate_both")):
        df = df0.copy()
        for c in cols_pct:
            if c in df.columns:
                df[c] = np.where(df[c].notna(), df[c] if c=="pass_rate_both" else df[c],
                                 df[c])
        # Formateo simple
        if "wr_med" in df.columns:  df["wr_med"]  = df["wr_med"].map(lambda v: f"{v:.1f}%")
        if "mdd_med" in df.columns: df["mdd_med"] = df["mdd_med"].map(lambda v: f"{v*100:.2f}%")
        if "pass_rate_both" in df.columns: df["pass_rate_both"] = df["pass_rate_both"].map(lambda v: f"{v:.1f}%")
        return df

    print("\n# MDD-down · resumen compacto\n")

    print("## Top 12 por MDD (desempate PF/WR/Trades)")
    cols_show = ["t","sl","tp1","p","pf_med","wr_med","tr_med","mdd_med","n"]
    print(pretty(top[cols_show]).to_string(index=False))

    if len(relaxed):
        print("\n## Shortlist RELAJADA")
        print(pretty(relaxed[cols_show]).to_string(index=False))
    else:
        print("\n## Shortlist RELAJADA\n[vacío]")

    if len(final):
        print("\n## Shortlist FINAL (≥60% freezes pasan PF&MDD)")
        cols_final = ["t","sl","tp1","p","pf_med","wr_med","tr_med","mdd_med","pf_min","n","pass_rate_both"]
        print(pretty(final[cols_final]).to_string(index=False))
    else:
        print("\n## Shortlist FINAL (≥60% freezes pasan PF&MDD)\n[vacío]")

    print(f"\n**Archivos**:\n- `{OUT_DIR}/agg_by_combo.csv`\n- `{OUT_DIR}/shortlist_relaxed.csv`\n- `{OUT_DIR}/shortlist_final.csv`")

    # ---- Confirmación por-freeze (si hay final) ----
    if len(final):
        t0, sl0, tp10, p0 = final.iloc[0][["t","sl","tp1","p"]]
        dff = df[(df["t"]==t0)&(df["sl"]==sl0)&(df["tp1"]==tp10)&(df["p"]==p0)].copy()
        dff = dff.sort_values("freeze")
        dff_out = dff[["freeze","pf","wr","trades","mdd","pass_both"]].rename(
            columns={"wr":"WR","trades":"trades","mdd":"MDD"}
        )
        dff_out.to_csv(f"{OUT_DIR}/confirm_by_freeze.csv", index=False)

        # después de filtrar dff = df[(t,sl,tp1,p)]
        dff = dff.drop_duplicates(subset=["freeze"]).sort_values("freeze")

        summ = pd.DataFrame([{
            "PF_med": dff["pf"].median(),
            "WR_med": dff["wr"].median(),
            "Trades_med": dff["trades"].median(),
            "MDD_med": dff["mdd"].median(),
            "pass_rate_both": 100.0 * dff["pass_both"].mean(),
            "n": dff["freeze"].nunique()
        }])
        summ.to_csv(f"{OUT_DIR}/confirm_summary.csv", index=False)

        # MD
        md = []
        md.append("# MDD-down · confirmación")
        md.append(f"- Combo: T={t0:.2f}, SL={sl0:.2f}, TP1={tp10:.2f}, P={p0:.2f}  | cap MDD={CAP_MDD*100:.2f}%")
        md.append(f"- Freezes: {int(dff['freeze'].nunique())}")
        md.append("\n## Medianas 60d\n```")
        md.append(summ.assign(
            WR_med = summ.WR_med.map(lambda v: f"{v:.2f}%"),
            MDD_med = summ.MDD_med.map(lambda v: f"{v*100:.2f}%"),
            pass_rate_both = summ.pass_rate_both.map(lambda v: f"{v:.1f}%"),
        ).to_string(index=False))
        md.append("```")
        md.append("## Por freeze (60d)\n```")
        md.append(dff_out.assign(
            WR = dff_out.WR.map(lambda v: f"{v:.2f}%"),
            MDD = dff_out.MDD.map(lambda v: f"{v*100:.2f}%"),
        ).to_string(index=False))
        md.append("```")
        md.append("**Archivos**:\n- `reports/w34_mdd_down/confirm_by_freeze.csv`\n- `reports/w34_mdd_down/confirm_summary.csv`")
        open(f"{OUT_DIR}/confirm_summary.md","w").write("\n".join(md))
        print(f"OK -> {OUT_DIR}/confirm_summary.md")
    else:
        print("OK -> sin confirmación (no hubo final)")

if __name__ == "__main__":
    main()