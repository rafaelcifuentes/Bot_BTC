#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
OUT="$ROOT/reports/mini_accum/walkforward"
A="${1:-$OUT/wf_summary_kpis.csv}"   # baseline A por defecto
B="${2:-}"                           # candidato B (obligatorio)
LABEL="${3:-candidate}"
export FREEZE_DATE="${FREEZE_DATE:-$(TZ=America/New_York date +%F)}"

if [[ -z "$B" ]]; then
  echo "Uso: $(basename "$0") [csv_A (opt)] <csv_B> [label]"
  exit 1
fi
[[ -f "$A" ]] || { echo "⛔ No existe A=$A"; exit 1; }
[[ -f "$B" ]] || { echo "⛔ No existe B=$B"; exit 1; }
mkdir -p "$OUT"

python - "$A" "$B" "$OUT" "$LABEL" <<'PY'
import pandas as pd, sys, os, datetime as dt
A_path, B_path, OUT, label = sys.argv[1:5]
freeze = os.environ.get("FREEZE_DATE") or dt.datetime.now().strftime("%Y-%m-%d")
cols = ["window","config","sats_mult","mdd_vs_hodl","fpy","flips","passed","src"]

def load_csv(p):
    df = pd.read_csv(p, header=None, names=cols)
    # numéricos
    for c in ["sats_mult","mdd_vs_hodl","fpy"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["flips"] = pd.to_numeric(df["flips"], errors="coerce").astype("Int64")
    # boolean sin FutureWarning
    df["passed"] = df["passed"].astype(str).str.strip().str.lower().isin(
        {"true","1","yes","y","t"}
    ).astype(bool)
    return df.dropna(subset=["sats_mult","mdd_vs_hodl","fpy"])

A = load_csv(A_path)
B = load_csv(B_path)

def summarize(df):
    return (df.groupby("window", dropna=False)
              .agg(median_sats_mult=("sats_mult","median"),
                   median_mdd_vs_hodl=("mdd_vs_hodl","median"),
                   median_fpy=("fpy","median"),
                   fail_rate=("passed", lambda s: 1 - s.mean()))
              .reset_index())

Sa, Sb = summarize(A), summarize(B)

# Alinear ventanas y columnas planas
W = sorted(set(Sa["window"]).union(Sb["window"]))
Sa = Sa.set_index("window").reindex(W); Sa.index.name = "window"
Sb = Sb.set_index("window").reindex(W); Sb.index.name = "window"

Delta = pd.DataFrame(index=W)
Delta["d_sats"] = Sb["median_sats_mult"] - Sa["median_sats_mult"]
Delta["d_mdd"]  = Sb["median_mdd_vs_hodl"] - Sa["median_mdd_vs_hodl"]
Delta["d_fpy"]  = Sb["median_fpy"] - Sa["median_fpy"]

summary = pd.concat([Sa.add_prefix("A_"),
                     Sb.add_prefix("B_"),
                     Delta.add_prefix("Δ_")], axis=1).reset_index()

# Ventanas ACTIVAS (si A o B operaron)
active = (summary["A_median_fpy"] > 0) | (summary["B_median_fpy"] > 0)
summary_active = summary[active].copy()

# Gates por ventana (activas)
summary_active["gate_sats"] = summary_active["Δ_d_sats"] >= 0.02
summary_active["gate_mdd"]  = summary_active["B_median_mdd_vs_hodl"] <= (summary_active["A_median_mdd_vs_hodl"] + 0.05)
summary_active["gate_fail"] = summary_active["B_fail_rate"] <= 0.25
summary_active["gate_fpy"]  = summary_active["B_median_fpy"] <= 26

# Veredicto global
g_sats = bool(summary_active["gate_sats"].all()) if len(summary_active) else False
g_mdd  = bool(summary_active["gate_mdd"].all())  if len(summary_active) else False
g_fail = bool(summary_active["gate_fail"].all()) if len(summary_active) else False
g_fpy  = bool(summary_active["gate_fpy"].all())  if len(summary_active) else False
status = "PASS" if (g_sats and g_mdd and g_fail and g_fpy) else "CHECK"

def fmt(df): return df.to_csv(index=False)

md = []
md.append(f"# Mini-Accum KISS v1 — A/B semanal en sombra ({freeze})\n")
md.append(f"**A**: {os.path.basename(A_path)}  \n**B**: {os.path.basename(B_path)}  \n**Label**: {label}\n")
md.append("## Resumen por ventana (medianas)\n"); md.append(fmt(summary.fillna("")))
md.append("\n## Ventanas ACTIVAS (median_fpy>0 en A o B)\n")
md.append(fmt(summary_active.fillna("")) if len(summary_active) else "_No hubo ventanas activas_")
md.append("\n## Gates (sobre activas)\n")
md.append(f"- Δ median_sats_mult ≥ +0.02: {g_sats}\n- mdd_vs_hodl(B) ≤ mdd_vs_hodl(A)+0.05: {g_mdd}\n- fail_rate_B ≤ 0.25: {g_fail}\n- median_fpy_B ≤ 26: {g_fpy}\n")
md.append(f"\n**VEREDICTO (A/B)**: {status}\n")

out_md = os.path.join(OUT, "ab_latest.md")
with open(out_md, "w") as f: f.write("\n".join(md))
print(f"[OK] A/B escrito → {out_md}")
print(f"[VERDICT] {status}")
PY
