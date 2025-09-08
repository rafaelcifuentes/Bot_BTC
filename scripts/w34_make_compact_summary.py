import os, glob, pandas as pd, numpy as np
# --- Pct helpers (autodetect: 0–1 vs 0–100) ---
def _pct_auto(x, digs=1):
    if x is None: return ""
    try:
        x = float(x)
    except Exception:
        return ""
    if x > 2.0:   # heurística: ya en 0–100
        return f"{x:.{digs}f}%"
    return f"{x*100:.{digs}f}%"

def apply_pct(df, col, digs=1):
    if col in df.columns:
        df[col] = df[col].map(lambda v: _pct_auto(v, digs))

def apply_pct_many(df, mapping):
    # mapping: {"WR":1, "MDD":2, ...}
    for c, d in mapping.items():
        apply_pct(df, c, d)
ROOT = "reports/w34_oos_btc"
OUT_BY = os.path.join(ROOT, "by_freeze_60d_compact.csv")
OUT_SUM = os.path.join(ROOT, "summary_60d_compact.csv")
OUT_MD = os.path.join(ROOT, "summary_60d_compact.md")

BASELINE_MDD = float(os.environ.get("BASELINE_MDD", "0.0170"))  # 1.70%
CAP_MDD = float(os.environ.get("CAP_MDD", "0.0187"))            # 1.87%

def parse_T_from_sub(sub):
    # sub e.g. "t058" -> 0.58
    try:
        n = sub[1:]
        return float(f"0.{n}")
    except Exception:
        return np.nan

def collect_rows():
    rows=[]
    if not os.path.isdir(ROOT):
        return pd.DataFrame(rows)
    # subcarpetas t0xx
    subs = [d for d in os.listdir(ROOT) if d.startswith("t0") and os.path.isdir(os.path.join(ROOT,d))]
    for sub in subs:
        T = parse_T_from_sub(sub)
        for f in sorted(glob.glob(os.path.join(ROOT, sub, "*.csv"))):
            try:
                df = pd.read_csv(f)
                if "days" not in df.columns: 
                    continue
                r = df.loc[df["days"]==60]
                if r.empty:
                    continue
                r = r.iloc[0]
                base=os.path.basename(f)
                freeze_tag = base.replace("_plus","")
                rows.append({
                    "T": T,
                    "freeze": freeze_tag,
                    "pf": float(r.get("pf", np.nan)),
                    "wr": float(r.get("win_rate", np.nan)),
                    "trades": int(r.get("trades", np.nan)) if not pd.isna(r.get("trades", np.nan)) else np.nan,
                    "mdd": abs(float(r.get("mdd", np.nan))),
                    "file": base
                })
            except Exception:
                pass
    return pd.DataFrame(rows)

def fmt_pct(x):
    return f"{x:.1f}%" if pd.notna(x) else ""

def fmt_pct2(x):
    return f"{x:.2%}" if pd.notna(x) else ""

def main():
    df = collect_rows()
    if df.empty:
        print("[WARN] No se encontraron CSV 60d en reports/w34_oos_btc/t0*/")
        # escribe archivos vacíos para no romper pipelines
        pd.DataFrame().to_csv(OUT_BY, index=False)
        pd.DataFrame().to_csv(OUT_SUM, index=False)
        open(OUT_MD,"w").write("# W3–W4 · OOS 4y (60d) — Resumen compacto\n\n[sin filas]\n")
        return

    # Guardar by_freeze “compacto”
    df_by = df[["T","freeze","pf","wr","trades","mdd","file"]].copy()
    df_by.to_csv(OUT_BY, index=False)
    print(f"OK -> {OUT_BY}")

    # Summary por T (medianas + min PF)
    grp = df_by.groupby("T")
    summ = pd.DataFrame({
        "PF_med": grp["pf"].median(),
        "WR_med": grp["wr"].median(),
        "Trades_med": grp["trades"].median(),
        "MDD_med": grp["mdd"].median(),
        "PF_min": grp["pf"].min(),
        "N": grp.size()
    }).reset_index()

    # Gate
    summ["GATE"] = (
        (summ["PF_med"] >= 1.50) &
        (summ["WR_med"] >= 60.0) &
        (summ["Trades_med"] >= 30.0) &
        (summ["MDD_med"] <= CAP_MDD) &
        (summ["PF_min"] >= 1.30)
    )

    summ.to_csv(OUT_SUM, index=False)
    print(f"OK -> {OUT_SUM}")

    # Markdown compacto
    total_rows = len(df_by)
    md = []
    md.append("# W3–W4 · OOS 4y (60d) — Resumen compacto\n")
    md.append(f"- Baseline MDD S1(D4)= **{BASELINE_MDD:.2%}**  | Cap dinámico= **{CAP_MDD:.2%}**")
    md.append(f"- Muestras: **{total_rows}** filas\n")
    md.append("## Medianas por T (gate PF≥1.5, WR≥60, Trades≥30, MDD≤cap, PFmin≥1.30)\n")
    summ_fmt = summ.copy()
    summ_fmt["WR_med"] = summ_fmt["WR_med"].map(fmt_pct)
    summ_fmt["MDD_med"] = summ_fmt["MDD_med"].map(fmt_pct2)
    md.append("```\n"+summ_fmt[["T","PF_med","WR_med","Trades_med","MDD_med","PF_min","N","GATE"]].to_string(index=False)+"\n```")

    # Últimos 6 freezes por T (usar mediana de duplicados plus/base)
    for Tval in sorted(df_by["T"].unique()):
        sub = df_by[df_by["T"]==Tval].copy()
        # consolida por freeze
        g = sub.groupby("freeze").agg(pf=("pf","median"), wr=("wr","median"), trades=("trades","median"), mdd=("mdd","median")).reset_index()
        tags = sorted(g["freeze"].unique())
        last6 = g[g["freeze"].isin(tags[-6:])].copy()
        last6["wr"] = last6["wr"].map(fmt_pct)
        last6["mdd"] = last6["mdd"].map(fmt_pct2)
        md.append(f"\n### T={Tval:.2f} — últimos 6 freezes")
        md.append("```\n"+last6[["freeze","pf","wr","trades","mdd"]].to_string(index=False)+"\n```")

    # Top/Bottom 5 PF global
    top5 = df_by.sort_values("pf", ascending=False).head(5).copy()
    bot5 = df_by.sort_values("pf", ascending=True).head(5).copy()
    for dd in (top5, bot5):
        dd["wr"] = dd["wr"].map(fmt_pct)
        dd["mdd"] = dd["mdd"].map(fmt_pct2)
    md.append("\n## Top 5 PF (global)\n")
    md.append("```\n"+top5[["T","freeze","pf","wr","trades","mdd"]].to_string(index=False)+"\n```")
    md.append("\n## Bottom 5 PF (global)\n")
    md.append("```\n"+bot5[["T","freeze","pf","wr","trades","mdd"]].to_string(index=False)+"\n```")

    md.append("\n**Archivos generados**:")
    md.append(f"- `{OUT_BY}`")
    md.append(f"- `{OUT_SUM}`")
    md.append(f"- `{OUT_MD}`")

    open(OUT_MD,"w").write("\n".join(md))
    print(f"OK -> {OUT_MD}")

if __name__ == "__main__":
    main()
