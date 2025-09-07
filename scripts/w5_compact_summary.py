# W5 compact + stress summary
import glob, os, pandas as pd, numpy as np
BASEA="reports/w5_oos_btc/a"; BASEB="reports/w5_oos_btc/b"
STRA="reports/w5_stress_btc/a"; STRB="reports/w5_stress_btc/b"
os.makedirs("reports/w5_summary", exist_ok=True)

def pick60(path):
    df=pd.read_csv(path)
    r=df.loc[df['days']==60].iloc[0]
    return dict(pf=float(r['pf']),
                wr=float(r['win_rate']),
                tr=int(r['trades']),
                mdd=abs(float(r['mdd'])))

def load_block(root,label):
    rows=[]
    for f in glob.glob(os.path.join(root,"*.csv")):
        try:
            r=pick60(f); r['file']=os.path.basename(f); r['label']=label
            r['freeze']=r['file'].replace(".csv","")
            rows.append(r)
        except: pass
    return pd.DataFrame(rows)

base = pd.concat([load_block(BASEA,"A_base"), load_block(BASEB,"B_base")], ignore_index=True)
stress = pd.concat([load_block(STRA,"A_stress"), load_block(STRB,"B_stress")], ignore_index=True)

# Join base vs stress por freeze/vecino (A/B)
def lab(x): return "A" if "_oos_btc/a/" in x or "_stress_btc/a/" in x else "B"
base['vec'] = base['label'].str[0]
stress['vec'] = stress['label'].str[0]
base['freeze_key'] = base['freeze']
stress['freeze_key'] = stress['freeze']

joined = pd.merge(
    base[['freeze_key','vec','pf','wr','tr','mdd']],
    stress[['freeze_key','vec','pf','wr','tr','mdd']],
    on=['freeze_key','vec'], suffixes=('_base','_stress')
)

# Ratios + gates
joined['pf_ratio']  = joined['pf_stress']/(joined['pf_base']+1e-9)
joined['mdd_ratio'] = np.where(joined['mdd_base']>0, joined['mdd_stress']/joined['mdd_base'], np.inf)
joined['pass_both'] = (joined['pf_ratio']>=0.90)&(joined['mdd_ratio']<=1.20)

# Compact por vecino
def fmtpct(x): return f"{x*100:.2f}%"
compact=[]
for vec in ['A','B']:
    sub = joined[joined['vec']==vec]
    if len(sub)==0: continue
    row = {
      'neighbor': 'A (T=0.60 SL=1.25 TP1=0.65 P=0.80)' if vec=='A' else 'B (T=0.58 SL=1.25 TP1=0.70 P=0.80)',
      'pf_med_base': sub['pf_base'].median(),
      'pf_med_stress': sub['pf_stress'].median(),
      'wr_med_base': sub['wr_base'].median(),
      'wr_med_stress': sub['wr_stress'].median(),
      'tr_med_base': sub['tr_base'].median(),
      'tr_med_stress': sub['tr_stress'].median(),
      'mdd_med_base': sub['mdd_base'].median(),
      'mdd_med_stress': sub['mdd_stress'].median(),
      'pf_ratio_med': sub['pf_ratio'].median(),
      'mdd_ratio_med': sub['mdd_ratio'].median(),
      'pass_rate_both': 100.0*sub['pass_both'].mean(),
      'n_freezes': len(sub)
    }
    compact.append(row)

comp = pd.DataFrame(compact)
comp.to_csv("reports/w5_summary/compact_base_vs_stress.csv", index=False)

# Decision md
lines=[
"# W5 · Resumen & decisión preliminar",
"",
"## Compacto por vecino (medianas 60d)",
comp.assign(
    wr_med_base=lambda d: d.wr_med_base.map(lambda v:f"{v:.2f}%"),
    wr_med_stress=lambda d: d.wr_med_stress.map(lambda v:f"{v:.2f}%"),
    mdd_med_base=lambda d: d.mdd_med_base.map(lambda v:f"{v*100:.2f}%"),
    mdd_med_stress=lambda d: d.mdd_med_stress.map(lambda v:f"{v*100:.2f}%"),
    pass_rate_both=lambda d: d.pass_rate_both.map(lambda v:f"{v:.1f}%"),
)[['neighbor','pf_med_base','pf_med_stress','wr_med_base','wr_med_stress',
   'tr_med_base','tr_med_stress','mdd_med_base','mdd_med_stress',
   'pf_ratio_med','mdd_ratio_med','pass_rate_both','n_freezes']].to_string(index=False),
"",
"**Gate stress**: PF_ratio ≥ 0.90 y MDD_ratio ≤ 1.20 (por freeze; se reporta tasa de pases).",
"",
"**Archivos**:",
"- reports/w5_oos_btc/(a|b)/*.csv",
"- reports/w5_stress_btc/(a|b)/*.csv",
"- reports/w5_summary/compact_base_vs_stress.csv"
]
open("reports/w5_summary/decision.md","w").write("\n".join(lines))
print("OK -> reports/w5_summary/(compact_base_vs_stress.csv|decision.md)")