#!/usr/bin/env python3
# Compactador semanal: base vs stress por vecino (A/B)
import argparse, glob, os, pandas as pd, numpy as np

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
            r=pick60(f)
            r['file']=os.path.basename(f)
            r['freeze']=os.path.splitext(r['file'])[0]
            r['label']=label  # A_base, B_base, A_stress, B_stress
            rows.append(r)
        except Exception:
            pass
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseA", required=True)
    ap.add_argument("--baseB", required=True)
    ap.add_argument("--stressA", required=True)
    ap.add_argument("--stressB", required=True)
    ap.add_argument("--outdir", required=True)
    args=ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    baseA = load_block(args.baseA, "A_base")
    baseB = load_block(args.baseB, "B_base")
    stressA = load_block(args.stressA, "A_stress")
    stressB = load_block(args.stressB, "B_stress")

    # Debug breve
    print(f"[debug] found baseA={len(baseA)} baseB={len(baseB)}")
    print(f"[debug] found stressA={len(stressA)} stressB={len(stressB)}")

    base = pd.concat([baseA, baseB], ignore_index=True)
    stress = pd.concat([stressA, stressB], ignore_index=True)
    if base.empty or stress.empty:
        raise SystemExit("[error] No hay archivos base o stress; revisa rutas y corridas previas.")

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
    rows=[]
    for vec in ['A','B']:
        sub = joined[joined['vec']==vec]
        if len(sub)==0: continue
        rows.append({
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
        })
    comp = pd.DataFrame(rows)
    comp_path = os.path.join(args.outdir, "compact_base_vs_stress.csv")
    comp.to_csv(comp_path, index=False)

    # Decision MD
    def pct(x): return f"{x:.2f}%"
    def pct100(x): return f"{x*100:.2f}%"
    md = [
        "# W* · Resumen & decisión preliminar",
        "",
        "## Compacto por vecino (medianas 60d)",
        comp.assign(
            wr_med_base=lambda d: d.wr_med_base.map(pct),
            wr_med_stress=lambda d: d.wr_med_stress.map(pct),
            mdd_med_base=lambda d: d.mdd_med_base.map(pct100),
            mdd_med_stress=lambda d: d.mdd_med_stress.map(pct100),
            pass_rate_both=lambda d: d.pass_rate_both.map(lambda v:f\"{v:.1f}%\"),
        )[['neighbor','pf_med_base','pf_med_stress','wr_med_base','wr_med_stress',
           'tr_med_base','tr_med_stress','mdd_med_base','mdd_med_stress',
           'pf_ratio_med','mdd_ratio_med','pass_rate_both','n_freezes']].to_string(index=False),
        "",
        "**Gate stress**: PF_ratio ≥ 0.90 y MDD_ratio ≤ 1.20 (por freeze; se reporta tasa de pases).",
        "",
        "**Archivos**:",
        f"- {args.baseA} | {args.baseB}",
        f"- {args.stressA} | {args.stressB}",
        f"- {comp_path}"
    ]
    with open(os.path.join(args.outdir, "decision.md"), "w") as fh:
        fh.write("\n".join(md))
    print(f"OK -> {args.outdir}/(compact_base_vs_stress.csv|decision.md)")

if __name__ == "__main__":
    main()