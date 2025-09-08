#!/usr/bin/env python3
import os, glob, pandas as pd, numpy as np

RAW_DIR = "reports/w34_neighbors/raw"
OUT_DIR = "reports/w34_neighbors"
os.makedirs(OUT_DIR, exist_ok=True)

# Gates (ajústalos si hace falta)
PF_RATIO_MIN = float(os.environ.get("PF_RATIO_MIN", "0.90"))
MDD_RATIO_MAX = float(os.environ.get("MDD_RATIO_MAX", "1.20"))

def pick60(path):
    df = pd.read_csv(path)
    # Toma fila days==60 (robusto a int/str)
    if 'days' not in df.columns:
        raise ValueError(f"[faltante 'days'] {path}")
    # castea por si viene como str:
    try:
        df['days'] = df['days'].astype(int)
    except Exception:
        # si trae '60.0' como str
        df['days'] = df['days'].astype(float).round().astype(int)
    row = df.loc[df['days']==60]
    if row.empty:
        raise ValueError(f"[sin 60d] {path}")
    r = row.iloc[0]
    # nombres robustos
    pf = float(r.get('pf', np.nan))
    wr = float(r.get('win_rate', np.nan))
    tr = int(float(r.get('trades', np.nan))) if not pd.isna(r.get('trades', np.nan)) else np.nan
    mdd = abs(float(r.get('mdd', np.nan)))
    return dict(pf=pf, wr=wr, trades=tr, mdd=mdd)

def parse_fn(fn):
    """
    Espera: {TAG}_{LABEL}_{KIND}.csv
      TAG = 2025_07_01_00_00
      LABEL = T0.60_SL1.25_TP10.65_P0.80
      KIND = base | stress
    """
    base = os.path.basename(fn)
    name = os.path.splitext(base)[0]
    parts = name.split('_')
    if parts[-1] not in ('base','stress'):
        return None
    kind = parts[-1]
    tag = "_".join(parts[0:4])  # YYYY_MM_DD_00_00
    label_tokens = parts[4:-1]
    label_key = "_".join(label_tokens)
    # parsear numéricos para columnas bonitas
    t = sl = tp1 = p = np.nan
    for tok in label_tokens:
        if tok.startswith('T') and not tok.startswith('TP1'):
            t = float(tok[1:])
        elif tok.startswith('SL'):
            sl = float(tok[2:])
        elif tok.startswith('TP1'):
            tp1 = float(tok[3:])   # 'TP1' + '0.65'
        elif tok.startswith('P'):
            p = float(tok[1:])
    return dict(tag=tag, label_key=label_key, kind=kind, t=t, sl=sl, tp1=tp1, p=p)

# Carga métricas 60d
rows = []
for f in glob.glob(os.path.join(RAW_DIR, "*.csv")):
    meta = parse_fn(f)
    if not meta:
        continue
    try:
        m = pick60(f)
        rows.append({**meta, **m, "file": os.path.basename(f)})
    except Exception as e:
        # silente pero útil en debug
        # print(f"[skip] {f} -> {e}")
        pass

df = pd.DataFrame(rows)
if df.empty:
    print("[WARN] No se encontraron CSV en", RAW_DIR)
    raise SystemExit(0)

# pivot base vs stress por (freeze, label)
key_cols = ['tag','label_key','t','sl','tp1','p']
base_df = df[df.kind=='base'][key_cols + ['pf','wr','trades','mdd']].rename(
    columns={'pf':'pf_base','wr':'wr_base','trades':'trades_base','mdd':'mdd_base'}
)
stress_df = df[df.kind=='stress'][key_cols + ['pf','wr','trades','mdd']].rename(
    columns={'pf':'pf_stress','wr':'wr_stress','trades':'trades_stress','mdd':'mdd_stress'}
)

merged = pd.merge(base_df, stress_df, on=key_cols, how='inner')

if merged.empty:
    print("[WARN] No hay pares base/stress para comparar")
    raise SystemExit(0)

# ratios y gates
merged['pf_ratio']  = merged['pf_stress'] / (merged['pf_base'] + 1e-9)
merged['mdd_ratio'] = np.where(merged['mdd_base']>0, merged['mdd_stress']/merged['mdd_base'], np.inf)
merged['pass_pf']   = merged['pf_ratio']  >= PF_RATIO_MIN
merged['pass_mdd']  = merged['mdd_ratio'] <= MDD_RATIO_MAX
merged['pass_both'] = merged['pass_pf'] & merged['pass_mdd']

# salida por freeze
by_freeze = merged.copy()
by_freeze = by_freeze.rename(columns={'tag':'freeze'})
by_freeze['wr_base']    = (by_freeze['wr_base']   ).round(6)
by_freeze['wr_stress']  = (by_freeze['wr_stress'] ).round(6)
by_freeze['pf_ratio']   = by_freeze['pf_ratio'].round(3)
by_freeze['mdd_ratio']  = by_freeze['mdd_ratio'].round(3)
by_freeze.to_csv(os.path.join(OUT_DIR, "summary_by_freeze.csv"), index=False)

# mediana por combo
agg = merged.groupby(['label_key','t','sl','tp1','p'], as_index=False).agg(
    pf_base_med=('pf_base','median'),
    pf_stress_med=('pf_stress','median'),
    wr_base_med=('wr_base','median'),
    wr_stress_med=('wr_stress','median'),
    trades_base_med=('trades_base','median'),
    trades_stress_med=('trades_stress','median'),
    mdd_base_med=('mdd_base','median'),
    mdd_stress_med=('mdd_stress','median'),
    pf_ratio_med=('pf_ratio','median'),
    mdd_ratio_med=('mdd_ratio','median'),
    pass_rate_both=('pass_both','mean')
)
agg['pass_rate_both'] = (agg['pass_rate_both']*100).round(1)
agg['pass_gate'] = (agg['pf_ratio_med']>=PF_RATIO_MIN) & (agg['mdd_ratio_med']<=MDD_RATIO_MAX)
agg.sort_values(['mdd_stress_med','pf_stress_med'], ascending=[True,False], inplace=True)
agg.to_csv(os.path.join(OUT_DIR, "summary_by_combo.csv"), index=False)

# MD corto
def pct(x):
    return f"{x*100:.2f}%"

top = agg.head(12).copy()
top_disp = top[['t','sl','tp1','p','pf_stress_med','wr_stress_med','trades_stress_med','mdd_stress_med','pass_gate']].rename(
    columns={'pf_stress_med':'pf_med_stress','wr_stress_med':'wr_med_stress',
             'trades_stress_med':'tr_med_stress','mdd_stress_med':'mdd_med_stress'}
)
top_disp['wr_med_stress']  = top_disp['wr_med_stress'].map(lambda v: f"{v:.1f}%")
top_disp['mdd_med_stress'] = top_disp['mdd_med_stress'].map(pct)

md = [
"# Stress de fricción — Resumen vecinos (60d)",
f"- Gate: PF_ratio ≥ {PF_RATIO_MIN:.2f}  |  MDD_ratio ≤ {MDD_RATIO_MAX:.2f}",
"",
"## Top 12 por MDD_stress (desempate PF/WR)",
"```",
top_disp.to_string(index=False),
"```",
"",
"**Archivos**:",
"- reports/w34_neighbors/summary_by_freeze.csv",
"- reports/w34_neighbors/summary_by_combo.csv",
]
open(os.path.join(OUT_DIR,"summary_neighbors_stress.md"), "w").write("\n".join(md))

print("OK -> reports/w34_neighbors/summary_by_freeze.csv")
print("OK -> reports/w34_neighbors/summary_by_combo.csv")
print("OK -> reports/w34_neighbors/summary_neighbors_stress.md")