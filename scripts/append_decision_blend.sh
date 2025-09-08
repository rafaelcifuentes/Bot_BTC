#!/usr/bin/env bash
# scripts/append_decision_blend.sh
set -euo pipefail
ALLOC_YAML="configs/allocator_sombra.yaml"
KPIS_MD="reports/allocator/sombra_kpis.md"
DEC_FILE="decisiones.md"
LABEL_DATE="$(date -u +%F)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) ALLOC_YAML="$2"; shift 2;;
    --kpis) KPIS_MD="$2"; shift 2;;
    --dec) DEC_FILE="$2"; shift 2;;
    --date) LABEL_DATE="$2"; shift 2;;
    *) echo "Arg no reconocido: $1"; exit 1;;
  esac
done

read_fees() {
python3 - <<'PY'
import sys,yaml
fee=slip=6.0
cfg=sys.stdin.read().strip()
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

parse_kpis() {
python3 - <<'PY'
import sys,re,json
p=sys.stdin.read().strip()
try:
    txt=open(p,encoding="utf-8").read()
except Exception:
    print("{}"); raise SystemExit

rows=[r.strip() for r in txt.splitlines() if r.strip().startswith("|")]
def parse_row(line):
    return [x.strip() for x in line.strip("|").split("|")]
base=overlay={}
for line in rows:
    cols=parse_row(line)
    if len(cols)<6:
        continue
    if cols[0].lower()=="serie":
        continue
    name=cols[0].lower()
    row=dict(vol=cols[1], mdd=cols[2], pf=cols[3], wr=cols[4], sortino=cols[5])
    if name.startswith("base"):
        base=row
    if name.startswith("overlay"):
        overlay=row
print(json.dumps(dict(base=base, overlay=overlay)))
PY
}

run_breakdown() {
python3 - <<'PY'
import re,subprocess,sys,json
try:
    cp = subprocess.run([sys.executable,"tests_overlay_check.py"], capture_output=True, text=True)
    out=cp.stdout
except Exception:
    out=""

def grab(key):
    m=re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", out, flags=re.M)
    return m.group(1).strip() if m else ""
keys=[
"NET por trayectoria","Gross - Σcostes","Gross (sin costes)",
"Costes totales","Turnover total","Coste D / P","Cost share D/P","Diff (calc-curve)"
]
d={k:grab(k) for k in keys}
print(json.dumps(d, ensure_ascii=False))
PY
}

append_md() {
python3 - <<'PY'
import sys,json,math
dec,label_dt,fee,slip,kpis_js,bd_js = sys.argv[1:7]
fee=float(fee); slip=float(slip); ttl=int(round(fee+slip))
k = json.loads(kpis_js) if kpis_js.strip() else {}
b = json.loads(bd_js) if bd_js.strip() else {}

def num(x):
    x=(x or "").replace("%","").strip()
    try: return float(x)
    except: return float("nan")

base=k.get("base",{}) or {}
over=k.get("overlay",{}) or {}

vol_base=num(base.get("vol","nan"))
vol_over=num(over.get("vol","nan"))

def parse_mdd(v):
    s=(v or "").strip()
    if not s: return float("nan")
    if s.endswith("%"):
        try: return float(s[:-1])/100.0
        except: return float("nan")
    try: return float(s)
    except: return float("nan")

mdd_base=parse_mdd(base.get("mdd"))
mdd_over=parse_mdd(over.get("mdd"))

pf_base=num(base.get("pf","nan")); pf_over=num(over.get("pf","nan"))

impr_vol = (vol_base - vol_over)/vol_base*100 if vol_base==vol_base and vol_base!=0 else float("nan")
impr_mdd = ((abs(mdd_base)-abs(mdd_over))/abs(mdd_base))*100 if mdd_base==mdd_base and mdd_base!=0 else float("nan")
delta_pf = pf_over - pf_base if pf_over==pf_over and pf_base==pf_base else float("nan")

with open(dec,"a",encoding="utf-8") as f:
    f.write(f"### 🟩 Corazón / Allocator (sombra) ({label_dt}) — **Blend semanal**\n")
    f.write(f"**Fees:** {fee:.0f} bps + {slip:.0f} bps — **{ttl} bps** totales\n\n")
    if base and over:
        f.write("**KPIs (tabla)**\n")
        f.write(f"- Base → Vol: {base.get('vol','n/a')}, MDD: {base.get('mdd','n/a')}, PF: {base.get('pf','n/a')}, WR: {base.get('wr','n/a')}\n")
        f.write(f"- Overlay → Vol: {over.get('vol','n/a')}, MDD: {over.get('mdd','n/a')}, PF: {over.get('pf','n/a')}, WR: {over.get('wr','n/a')}\n")
        f.write(f"- Mejora Vol (↓): **{impr_vol:.2f}%**, Mejora MDD (↓): **{impr_mdd:.2f}%**, ΔPF: **{delta_pf:+.2f}**\n\n")
    else:
        f.write("_Aviso_: no se encontró la tabla en los KPIs; revisa `reports/allocator/sombra_kpis.md`.\n\n")

    if b:
        f.write("**Breakdown (tests_overlay_check.py)**\n")
        for k,v in b.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")
    else:
        f.write("_Aviso_: no se pudo ejecutar/leer `tests_overlay_check.py`.\n\n")

    f.write("**Criterio:** aceptar overlay si MDD ↓ ≥ 15% **o** Vol ↓ ≥ 10% y ΔPF ≥ −0.05. Si no, revisar reglas/pesos.\n\n")
print("[OK] Append Blend")
PY
}

# ---------- main ----------
# Fees
IFS=, read -r FEE_BPS SLIP_BPS < <(printf "%s" "$ALLOC_YAML" | read_fees)

# KPIs → JSON (o "{}")
KPIS_JSON="$(printf "%s" "$KPIS_MD" | parse_kpis || echo "{}")"
[[ -z "$KPIS_JSON" ]] && KPIS_JSON="{}"

# Breakdown → JSON (o "{}")
BD_JSON="$(run_breakdown || echo "{}")"
[[ -z "$BD_JSON" ]] && BD_JSON="{}"

# Append
append_md "$DEC_FILE" "$LABEL_DATE" "$FEE_BPS" "$SLIP_BPS" "$KPIS_JSON" "$BD_JSON"