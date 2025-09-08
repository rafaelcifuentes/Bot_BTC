#!/usr/bin/env bash
set -euo pipefail

# ========= CONFIG =========
W_TAG="w6"                  # Cambia si usas otra semana
SYMBOL="BTC-USD"
PERIOD="1460d"
HORIZONS="30,60,90"
MAX_BARS="975"

# Vecino A (alterno robusto)
A_T="0.60"; A_SL="1.25"; A_TP1="0.65"; A_P="0.80"
# Vecino B (candidato actual)
B_T="0.58"; B_SL="1.25"; B_TP1="0.70"; B_P="0.80"

# Costes acordados (ambos vecinos)
BASE_SLIP="0.0002"; BASE_COST="0.0004"
STRESS_SLIP="0.0003"; STRESS_COST="0.0005"

# ========= CARPETAS =========
mkdir -p "reports/${W_TAG}_oos_btc/a"  "reports/${W_TAG}_oos_btc/b"
mkdir -p "reports/${W_TAG}_stress_btc/a" "reports/${W_TAG}_stress_btc/b"
mkdir -p "reports/${W_TAG}_summary"

# ========= FREEZES (dos últimos miércoles 00:00 UTC) =========
FREEZES=()
while IFS= read -r ln; do
  [ -n "$ln" ] && FREEZES+=("$ln")
done < <(python - <<'PY'
from datetime import datetime, timedelta, timezone
def last_wednesday_utc(k=0):
    now = datetime.now(timezone.utc)
    delta_days = (now.isoweekday() - 3) % 7  # miércoles=3
    d = (now - timedelta(days=delta_days+7*k)).replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"{d.strftime('%Y-%m-%d')} 00:00")
last_wednesday_utc(0)
last_wednesday_utc(1)
PY
)

echo "FREEZES detectados:"
for f in "${FREEZES[@]}"; do echo " - $f"; done

# ========= Helper para correr (BASE o STRESS) =========
run_pair() {
  local MODE="$1" T="$2" SL="$3" TP1="$4" P="$5" SUBDIR="$6" SLIP="$7" COST="$8"
  for F in "${FREEZES[@]}"; do
    TAG="$(echo "$F" | tr ':- ' '_')"   # 2025_09_03_00_00
    OUT="reports/${W_TAG}_${SUBDIR}/${TAG}.csv"
    echo ">> [$MODE] T=$T SL=$SL TP1=$TP1 P=$P freeze=$F -> $OUT"
    python swing_4h_forward_diamond.py --skip_yf \
      --symbol "$SYMBOL" --period "$PERIOD" --horizons "$HORIZONS" \
      --freeze_end "$F" --max_bars "$MAX_BARS" \
      --threshold "$T" --sl_atr_mul "$SL" --tp1_atr_mul "$TP1" --partial_pct "$P" \
      --slip "$SLIP" --cost "$COST" \
      --out_csv "$OUT"
  done
}

# ========= 2) BASE (A y B) =========
run_pair "BASE-A" "$A_T" "$A_SL" "$A_TP1" "$A_P" "oos_btc/a" "$BASE_SLIP" "$BASE_COST"
run_pair "BASE-B" "$B_T" "$B_SL" "$B_TP1" "$B_P" "oos_btc/b" "$BASE_SLIP" "$BASE_COST"

# ========= 3) STRESS (A y B) =========
run_pair "STRESS-A" "$A_T" "$A_SL" "$A_TP1" "$A_P" "stress_btc/a" "$STRESS_SLIP" "$STRESS_COST"
run_pair "STRESS-B" "$B_T" "$B_SL" "$B_TP1" "$B_P" "stress_btc/b" "$STRESS_SLIP" "$STRESS_COST"

# ========= 4) COMPACTA & DECISIÓN =========
python - <<PY
import glob, os, pandas as pd, numpy as np
W_TAG = "${W_TAG}"

def pick60(path):
    df=pd.read_csv(path)
    r=df.loc[df['days']==60].iloc[0]
    return dict(pf=float(r['pf']), wr=float(r['win_rate']), tr=int(r['trades']), mdd=abs(float(r['mdd'])))

def load_block(root,label):
    rows=[]
    for f in glob.glob(os.path.join(root,"*.csv")):
        try:
            r=pick60(f); r['file']=os.path.basename(f); r['label']=label; r['freeze']=r['file'].replace(".csv","")
            rows.append(r)
        except: pass
    return pd.DataFrame(rows)

BASEA=f"reports/{W_TAG}_oos_btc/a"; BASEB=f"reports/{W_TAG}_oos_btc/b"
STRA =f"reports/{W_TAG}_stress_btc/a"; STRB=f"reports/{W_TAG}_stress_btc/b"
base   = pd.concat([load_block(BASEA,"A_base"), load_block(BASEB,"B_base")], ignore_index=True)
stress = pd.concat([load_block(STRA,"A_stress"), load_block(STRB,"B_stress")], ignore_index=True)
if base.empty or stress.empty: raise SystemExit("[ERROR] No hay CSVs para compactar.")

base['vec']   = base['label'].str[0]
stress['vec'] = stress['label'].str[0]
base['freeze_key'] = base['freeze']; stress['freeze_key'] = stress['freeze']
joined = pd.merge(
    base[['freeze_key','vec','pf','wr','tr','mdd']],
    stress[['freeze_key','vec','pf','wr','tr','mdd']],
    on=['freeze_key','vec'], suffixes=('_base','_stress')
)
joined['pf_ratio']  = joined['pf_stress']/(joined['pf_base']+1e-9)
joined['mdd_ratio'] = np.where(joined['mdd_base']>0, joined['mdd_stress']/joined['mdd_base'], np.inf)
joined['pass_both'] = (joined['pf_ratio']>=0.90)&(joined['mdd_ratio']<=1.20)

def neighbor_name(v):
    return "A (T=0.60 SL=1.25 TP1=0.65 P=0.80)" if v=="A" else "B (T=0.58 SL=1.25 TP1=0.70 P=0.80)"

comp=[]
for vec in ["A","B"]:
    sub=joined[joined["vec"]==vec]
    if len(sub)==0: continue
    comp.append({
      "neighbor": neighbor_name(vec),
      "pf_med_base": sub["pf_base"].median(),
      "pf_med_stress": sub["pf_stress"].median(),
      "wr_med_base": sub["wr_base"].median(),
      "wr_med_stress": sub["wr_stress"].median(),
      "tr_med_base": sub["tr_base"].median(),
      "tr_med_stress": sub["tr_stress"].median(),
      "mdd_med_base": sub["mdd_base"].median(),
      "mdd_med_stress": sub["mdd_stress"].median(),
      "pf_ratio_med": sub["pf_ratio"].median(),
      "mdd_ratio_med": sub["mdd_ratio"].median(),
      "pass_rate_both": 100.0*sub["pass_both"].mean(),
      "n_freezes": len(sub),
    })
comp_df = pd.DataFrame(comp)
out_csv=f"reports/{W_TAG}_summary/compact_base_vs_stress.csv"
comp_df.to_csv(out_csv, index=False)

def pct(x): return f"{x:.2f}%"
tbl = comp_df.assign(
    wr_med_base=lambda d: d.wr_med_base.map(pct),
    wr_med_stress=lambda d: d.wr_med_stress.map(pct),
    mdd_med_base=lambda d: d.mdd_med_base.map(lambda v:pct(v*100)),
    mdd_med_stress=lambda d: d.mdd_med_stress.map(lambda v:pct(v*100)),
    pass_rate_both=lambda d: d.pass_rate_both.map(lambda v:f"{v:.1f}%"),
)[["neighbor","pf_med_base","pf_med_stress","wr_med_base","wr_med_stress",
   "tr_med_base","tr_med_stress","mdd_med_base","mdd_med_stress",
   "pf_ratio_med","mdd_ratio_med","pass_rate_both","n_freezes"]]

md=[
"# W6 · Resumen & decisión preliminar",
"",
"## Compacto por vecino (medianas 60d)",
tbl.to_string(index=False),
"",
"**Gate stress**: PF_ratio ≥ 0.90 y MDD_ratio ≤ 1.20 (por freeze; se reporta tasa de pases).",
"",
"**Archivos**:",
f"- reports/{W_TAG}_oos_btc/(a|b)/*.csv",
f"- reports/{W_TAG}_stress_btc/(a|b)/*.csv",
f"- reports/{W_TAG}_summary/compact_base_vs_stress.csv"
]
open(f"reports/{W_TAG}_summary/decision.md","w").write("\n".join(md))
print(f"OK -> reports/{W_TAG}_summary/(compact_base_vs_stress.csv|decision.md)")
PY

# ========= Verificación rápida =========
echo
echo "== Compacto =="
head -n 3 "reports/${W_TAG}_summary/compact_base_vs_stress.csv" || true
echo
echo "== Decision =="
sed -n '1,80p' "reports/${W_TAG}_summary/decision.md" || true
echo
echo "[DONE] Semana ${W_TAG} completada."
# ========= Weekly summary (segunda opción) + nota "qué vigilar" =========
python - <<'PY'
import os, pandas as pd

w_tag   = os.environ.get('W_TAG','wX')
root    = f"reports/{w_tag}_summary"
comp    = os.path.join(root, "compact_base_vs_stress.csv")
out_md  = os.path.join(root, "weekly_summary.md")
dec_md  = os.path.join(root, "decision.md")

df = pd.read_csv(comp)

def status_row(r):
    return "✅ PASS" if (r['pf_ratio_med']>=0.90 and r['mdd_ratio_med']<=1.20 and r['pass_rate_both']>=60.0) else "⚠️ FAIL"

view = df.copy()
view['status'] = [status_row(r) for _, r in df.iterrows()]
view['wr_med_base']    = view['wr_med_base'].map(lambda v: f"{v:.2f}%")
view['wr_med_stress']  = view['wr_med_stress'].map(lambda v: f"{v:.2f}%")
view['mdd_med_base']   = view['mdd_med_base'].map(lambda v: f"{v*100:.2f}%")
view['mdd_med_stress'] = view['mdd_med_stress'].map(lambda v: f"{v*100:.2f}%")
view['pass_rate_both'] = view['pass_rate_both'].map(lambda v: f"{v:.1f}%")

lines = []
lines.append(f"W summary: {w_tag}\n")
lines.append(view[['neighbor','status','pf_med_base','pf_med_stress','wr_med_base','wr_med_stress','tr_med_base','tr_med_stress','mdd_med_base','mdd_med_stress','pf_ratio_med','mdd_ratio_med','pass_rate_both','n_freezes']].to_string(index=False))
lines.append("\n\nGate stress: PF_ratio ≥ 0.90, MDD_ratio ≤ 1.20, pass_rate_both ≥ 60%")

os.makedirs(root, exist_ok=True)
open(out_md, "w").write("\n".join(lines))
print(f"OK -> {out_md}")

# --- Anexar "Qué vigilar la próxima semana" en decision.md (con valores actuales de B si existe)
try:
    b = df[df['neighbor'].str.startswith('B (')].iloc[0]
    note = []
    note.append("\n\n---\n### Qué vigilar la próxima semana")
    note.append(f"- En **B**, PF_ratio_med = {b['pf_ratio_med']:.3f}, MDD_ratio_med = {b['mdd_ratio_med']:.2f}, pass_rate_both = {b['pass_rate_both']:.1f}%.")
    note.append("- Si PF_ratio_med < 0.90 o pass_rate_both < 60%, **pausar B**.")
    note.append("- Mantener la brecha de MDD (límite 1.20).")
    with open(dec_md, "a") as f:
        f.write("\n".join(note))
    print(f"[append] Nota 'Qué vigilar' → {dec_md}")
except Exception:
    pass
PY

# ========= Vista rápida integrada =========
echo
echo "== Semana ${W_TAG} =="
echo
echo "-- Compacto --"
head -n 3 "reports/${W_TAG}_summary/compact_base_vs_stress.csv" || true
echo
echo "-- Weekly summary (segunda opción) --"
sed -n '1,80p' "reports/${W_TAG}_summary/weekly_summary.md" || true
echo
echo "-- Decisión (primeras 80 líneas) --"
sed -n '1,80p' "reports/${W_TAG}_summary/decision.md" || true