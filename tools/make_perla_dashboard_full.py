from pathlib import Path
import pandas as pd

# Rutas
weekly_csv = Path("reports/perla_negra_v2_summary.csv")
annual_csv = Path("reports/perla_money_view.csv")
out_csv    = Path("reports/perla_dashboard_full.csv")

# Cargar semanal
w = pd.read_csv(weekly_csv)

# Asegurar tipos
w['run'] = w['run'].astype(str)
w['leg'] = w['leg'].astype(str)

# Tomar última corrida por (run, leg) por si hay duplicados
w = w.sort_values(['run','leg']).drop_duplicates(subset=['run','leg'], keep='last').copy()

# Derivar 'date' (UTC) desde 'run' (YYYYMMDD_HHMM -> YYYYMMDD)
w['date'] = w['run'].str.slice(0,8)

# Elegimos por cada fecha las filas Baseline y Selected, y las combinamos lado a lado
def pick_one(df, legname):
    df_leg = df[df['leg'].str.lower() == legname.lower()].copy()
    # Columnas renombradas por prefijo
    cols_map = {
        'net': f'{legname.lower()}_net',
        'mdd': f'{legname.lower()}_mdd',
        'score': f'{legname.lower()}_score',
        'pf': f'{legname.lower()}_pf',
        'wr': f'{legname.lower()}_wr',
        'trades': f'{legname.lower()}_trades',
        'weights': f'{legname.lower()}_weights'
    }
    return df_leg[['date','net','mdd','score','pf','wr','trades','weights']].rename(columns=cols_map)

base = pick_one(w, 'Baseline')
sel  = pick_one(w, 'Selected')

weekly = pd.merge(base, sel, on='date', how='outer')

# Deltas (Selected - Baseline)
weekly['delta_net']   = weekly['selected_net'] - weekly['baseline_net']
weekly['delta_mdd']   = weekly['selected_mdd'] - weekly['baseline_mdd']
weekly['delta_score'] = weekly['selected_score'] - weekly['baseline_score']

# Veredicto simple
def verdict_row(r):
    try:
        ok_score = (r['delta_score'] or 0) > 0
        ok_mdd   = (r['delta_mdd'] or 0) <= 0
        return "✅ Mejor riesgo-ajustado" if (ok_score and ok_mdd) else "➖ Sin mejora clara"
    except Exception:
        return "➖ Sin datos"
weekly['verdict'] = weekly.apply(verdict_row, axis=1)

# Cargar anual (puede no existir la primera vez)
annual = None
if annual_csv.exists():
    annual = pd.read_csv(annual_csv)
    # Normalizar timestamp a datetime
    for col in ['stamp_utc']:
        if col in annual.columns:
            annual[col] = pd.to_datetime(annual[col], errors='coerce', utc=True)
    annual = annual.sort_values('stamp_utc')

# Vincular anual más reciente a cada 'date'
if annual is not None and not annual.empty:
    # Para cada 'date' semanal, buscamos el último stamp_utc anterior o igual a ese día (UTC 23:59)
    weekly['_date_ts'] = pd.to_datetime(weekly['date'], format='%Y%m%d', utc=True) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    merged = pd.merge_asof(
        weekly.sort_values('_date_ts'),
        annual.sort_values('stamp_utc'),
        left_on='_date_ts',
        right_on='stamp_utc',
        direction='backward'
    )
    # Renombrar columnas anuales
    rename_map = {
        'annual_net':'ann_net',
        'annual_ret_pct':'ann_ret_pct',
        'annual_pf':'ann_pf',
        'annual_mdd':'ann_mdd',
        'annual_score':'ann_score',
        'band_40_60':'ann_band'
    }
    for k,v in rename_map.items():
        if k in merged.columns:
            merged.rename(columns={k:v}, inplace=True)
    weekly = merged.drop(columns=['_date_ts','stamp_utc'], errors='ignore')
else:
    # Si no hay anual, rellenamos columnas anuales como vacías
    weekly['ann_net']    = pd.NA
    weekly['ann_ret_pct']= pd.NA
    weekly['ann_pf']     = pd.NA
    weekly['ann_mdd']    = pd.NA
    weekly['ann_score']  = pd.NA
    weekly['ann_band']   = pd.NA

# Orden final de columnas (amigable)
cols = [
    'date',
    'baseline_net','baseline_mdd','baseline_score','baseline_pf','baseline_wr','baseline_trades','baseline_weights',
    'selected_net','selected_mdd','selected_score','selected_pf','selected_wr','selected_trades','selected_weights',
    'delta_net','delta_mdd','delta_score','verdict',
    'ann_net','ann_ret_pct','ann_pf','ann_mdd','ann_score','ann_band'
]
# Asegurar que existan
cols = [c for c in cols if c in weekly.columns]
weekly = weekly[cols].sort_values('date')

# Guardar
out_csv.parent.mkdir(parents=True, exist_ok=True)
weekly.to_csv(out_csv, index=False)

print(f"✓ Guardado → {out_csv}")
print("\nÚltimas filas:")
print(weekly.tail(6).to_string(index=False))
