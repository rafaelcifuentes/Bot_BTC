import os, re, pandas as pd, datetime as dt
BASE='DD15_RB1_H30_G200_BULL0'

def write_ab(summary, out_path, legend=True):
    lines=[]
    ts = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    lines.append(f"# A/B KISS v1 — {ts}")
    try:
        df = pd.read_csv(summary)
    except Exception as e:
        lines += ["", f"**No pude leer el summary:** `{summary}` — {e}"]
    else:
        df['config_id'] = df['config_id'].astype(str).str.strip()
        req = ['sats_mult','mdd_vs_hodl','fpy','config_id','window']
        miss = [c for c in req if c not in df.columns]
        if miss:
            lines += ["", f"**Faltan columnas en summary:** {', '.join(miss)}. A/B omitido."]
        else:
            wins = sorted(df['window'].astype(str).unique())
            lines.append(f"Ventanas en summary: {', '.join(wins)}")
            base = df[df['config_id']==BASE]
            if base.empty:
                base = df[df['config_id'].str.contains(r'^DD15_.*_H30_.*_G200_.*BULL0$', na=False)]
            nb = df[df['config_id'].str.match(r'^DD(14|15|16)_RB(1|2)_H(30|31|32)_G200_BULL0$', na=False) & (df['config_id']!=BASE)].copy()
            if base.empty:
                lines += ["", f"**Baseline no encontrado (ni fuzzy)**: `{BASE}`. A/B omitido."]
            elif nb.empty:
                lines += ["", "**Vecindario vacío** (DD14/15/16 × RB1/2 × H30/31/32 × G200 × BULL0). A/B omitido."]
            else:
                def kpis(d):
                    return {
                        'median_sats': float(d['sats_mult'].median()),
                        'fail_rate': float((d['sats_mult']<1.0).mean()) if len(d)>0 else float('nan'),
                        'median_mdd': float(d['mdd_vs_hodl'].median()),
                        'median_fpy': float(d['fpy'].median()),
                    }
                bk = kpis(base)
                cands = [(cid, kpis(g)) for cid,g in nb.groupby('config_id')]
                cands.sort(key=lambda x:(-x[1]['median_sats'], x[1]['median_mdd'], x[1]['median_fpy']))
                cont, ck = cands[0]
                lines += [
                    "",
                    f"**Baseline:** `{BASE}`  |  **Contender (auto):** `{cont}`",
                    "",
                    "| KPI | Baseline | Contender | Δ (cont - base) | Mejor |",
                    "|:----|---------:|----------:|---------------:|:------:|",
                    f"| median(sats_mult) | {bk['median_sats']:.3f} | {ck['median_sats']:.3f} | {ck['median_sats']-bk['median_sats']:+.3f} | ↑ |",
                    f"| fail_rate (sats<1) | {bk['fail_rate']:.2%} | {ck['fail_rate']:.2%} | {ck['fail_rate']-bk['fail_rate']:+.2%} | ↓ |",
                    f"| median(mdd_vs_hodl) | {bk['median_mdd']:.3f} | {ck['median_mdd']:.3f} | {ck['median_mdd']-bk['median_mdd']:+.3f} | ↓ |",
                    f"| median(FPY) | {bk['median_fpy']:.2f} | {ck['median_fpy']:.2f} | {ck['median_fpy']-bk['median_fpy']:+.2f} | ↓ |",
                ]
                if legend:
                    lines += [
                        "",
                        "> **Leyenda KISS de lectura A/B**:",
                        "> Promueve a *revisión* si **Δ median(sats_mult) ≥ +0.02** y **no empeora** *median(mdd_vs_hodl)* ni *median(FPY)*.",
                        "> En otro caso: **mantener baseline**.",
                    ]
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write("\n".join(lines) + "\n")

write_ab('reports/mini_accum/walkforward/wf_summary_kpis.csv', 'reports/mini_accum/ab_latest.md')
