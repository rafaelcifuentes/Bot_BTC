#!/usr/bin/env bash
# scripts/append_decision_diamante.sh
set -euo pipefail

# ---------- args ----------
CSV_GLOB=""
SELECTED_YAML="configs/diamante_selected.yaml"
ALLOC_YAML=""
DEC_FILE="decisiones.md"
LABEL_DATE="$(date -u +%F)"
PF_MIN="1.50"
WR_MIN="0.60"
TR_MIN="30"
MDD_BASE=""   # opcional (decimal, ej 0.0187)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv) CSV_GLOB="$2"; shift 2;;
    --selected_yaml) SELECTED_YAML="$2"; shift 2;;
    --config_yml|--alloc_yaml) ALLOC_YAML="$2"; shift 2;;
    --dec) DEC_FILE="$2"; shift 2;;
    --date) LABEL_DATE="$2"; shift 2;;
    --pf_min) PF_MIN="$2"; shift 2;;
    --wr_min) WR_MIN="$2"; shift 2;;
    --trades_min) TR_MIN="$2"; shift 2;;
    --mdd_base) MDD_BASE="$2"; shift 2;;
    *) echo "Arg no reconocido: $1"; exit 1;;
  esac
done

# ---------- helpers ----------
read_fees() {
python3 - <<'PY'
import sys, yaml
fee=slip=6.0
cfg = sys.stdin.read().strip()
if cfg:
    try:
        y=yaml.safe_load(open(cfg))
        fee=float(y.get('costs',{}).get('fee_bps',6))
        slip=float(y.get('costs',{}).get('slip_bps',6))
    except Exception:
        pass
print(f"{fee},{slip}")
PY
}

read_selected() {
python3 - <<'PY'
import sys,yaml,json
p=sys.stdin.read().strip()
try:
    y=yaml.safe_load(open(p))
    print(json.dumps(y, ensure_ascii=False))
except Exception:
    print("{}")
PY
}

summarize_csvs() {
python3 - <<'PY'
import sys,glob,statistics as st
import pandas as pd, json, math, os

glob_pat = sys.stdin.read().strip()
files=sorted(glob.glob(glob_pat)) if glob_pat else []
if not files:
    print("{}"); sys.exit(0)

rows=[]
for f in files:
    try:
        df=pd.read_csv(f)
    except Exception:
        continue
    # heurística de columnas
    def pick(*prefs):
        for p in prefs:
            c=[x for x in df.columns if x.lower().startswith(p)]
            if c: return c[0]
        return None
    c_pf  = pick("pf","oos_pf","pf_")
    c_wr  = pick("wr","oos_wr","wr_")
    c_tr  = pick("trades","ntr","tr")
    c_mdd = pick("mdd","oos_mdd","maxdd")
    c_net = pick("net","oos_net","ret","pnl")

    if not (c_pf and c_wr and c_tr and c_mdd):
        continue
    d=df[[c for c in [c_pf,c_wr,c_tr,c_mdd,c_net] if c in df.columns]].copy()
    d.columns=["pf","wr","trades","mdd"] + (["net"] if c_net in d.columns else [])
    d=d.apply(pd.to_numeric, errors="coerce")
    # normalizaciones
    if (d["wr"]>1).mean()>0.5: d["wr"]=d["wr"]/100.0
    d["mdd"]=d["mdd"].abs()
    rows.append(d)

if not rows:
    print("{}"); sys.exit(0)

import pandas as pd
D=pd.concat(rows, ignore_index=True).dropna(subset=["pf","wr","trades","mdd"], how="all")

def med(s):
    s=s.dropna()
    return float(st.median(s)) if len(s) else float("nan")
def mmin(s):
    s=s.dropna()
    return float(s.min()) if len(s) else float("nan")

out={
  "pf_med": med(D.get("pf",pd.Series(dtype=float))),
  "wr_med": med(D.get("wr",pd.Series(dtype=float))),
  "trades_med": med(D.get("trades",pd.Series(dtype=float))),
  "mdd_med": med(D.get("mdd",pd.Series(dtype=float))),
  "pf_min": mmin(D.get("pf",pd.Series(dtype=float))),
}
if "net" in D:
  out["net_med"]=med(D["net"])

print(json.dumps(out))
PY
}

append_md() {
python3 - <<'PY'
import sys,json
dec      = sys.argv[1]
label_dt = sys.argv[2]
fee      = float(sys.argv[3])
slip     = float(sys.argv[4])
sel_js   = sys.argv[5]
sum_js   = sys.argv[6]
pf_min   = sys.argv[7]
wr_min   = sys.argv[8]
tr_min   = sys.argv[9]
mdd_base = sys.argv[10] if len(sys.argv)>10 else ""

sel = json.loads(sel_js) if sel_js.strip() else {}
s   = json.loads(sum_js) if sum_js.strip() else {}

def fmt(v, nd=3, pct=False):
    try:
        x=float(v)
        return (f"{x*100:.2f}%" if pct else f"{x:.{nd}f}")
    except: return "n/a"

with open(dec,"a",encoding="utf-8") as f:
    f.write(f"### 🟦 Diamante ({label_dt}) — **Cierre semanal OOS**\n")
    if sel:
        f.write("**Selected YAML**: `" + ", ".join(f"{k}={v}" for k,v in sel.items()) + "`\n")
    f.write(f"**Fees:** {fee:.0f} bps + {slip:.0f} bps — **{int(round(fee+slip))} bps** totales\n\n")

    if not s:
        f.write("_Aviso_: no se encontró CSV/columnas válidas en el patrón solicitado. Revisa la ruta o columnas esperadas (pf/wr/trades/mdd[/net]).\n\n")
    else:
        f.write("**Resumen (mediana OOS sobre ventanas)**\n")
        f.write(f"- PF_med: **{fmt(s.get('pf_med'))}**\n")
        f.write(f"- WR_med: **{fmt(s.get('wr_med'))}**\n")
        f.write(f"- Trades_med: **{fmt(s.get('trades_med'),1)}**\n")
        f.write(f"- MDD_med: **{fmt(s.get('mdd_med'),4)}**\n")
        if "net_med" in s:
            f.write(f"- NET_med: **{fmt(s.get('net_med'),6)}**\n")
        f.write("\n**Gates (objetivo)**\n")
        f.write(f"- PF_med ≥ {pf_min}\n- WR_med ≥ {wr_min}\n- Trades_med ≥ {tr_min}\n")
        if mdd_base:
            try:
                mb=float(mdd_base)
                f.write(f"- MDD_med ≤ 1.1×MDD_base ({mb:.4f})\n")
            except: pass
        f.write("\n**Decisión preliminar:** mantener o ajustar según gates; si falla, auditoría/fine-tuning.\n\n")
print("[OK] Append Diamante")
PY
}

# ---------- main ----------
# Fees
IFS=, read -r FEE_BPS SLIP_BPS < <(printf "%s" "${ALLOC_YAML:-}" | read_fees)

# Selected YAML → JSON
SELECTED_JSON="$(printf "%s" "$SELECTED_YAML" | read_selected || echo "{}")"
[[ -z "$SELECTED_JSON" ]] && SELECTED_JSON="{}"

# CSV summary → JSON (o "{}" si no hay match)
SUMMARY_JSON="{}"
if [[ -n "${CSV_GLOB}" ]]; then
  SUMMARY_JSON="$(printf "%s" "$CSV_GLOB" | summarize_csvs || echo "{}")"
  [[ -z "$SUMMARY_JSON" ]] && SUMMARY_JSON="{}"
fi

# Append seguro (aunque falte CSV)
append_md "$DEC_FILE" "$LABEL_DATE" "$FEE_BPS" "$SLIP_BPS" "$SELECTED_JSON" "$SUMMARY_JSON" "$PF_MIN" "$WR_MIN" "$TR_MIN" "$MDD_BASE"
