#!/usr/bin/env zsh
emulate -L zsh
set -euo pipefail

# Carga helpers si existen
[[ -f scripts/mini_accum/helpers.zsh ]] && source scripts/mini_accum/helpers.zsh

# Toma KPI por argumento o el último encontrado
KPI="${1:-}"
if [[ -z "$KPI" ]]; then
  if typeset -f pick_latest >/dev/null 2>&1; then
    KPI=$(pick_latest 'reports/mini_accum/*_kpis__*.csv')
  else
    setopt local_options null_glob
    local arr=(reports/mini_accum/*_kpis__*.csv)
    KPI="${arr[-1]:-}"
  fi
fi

[[ -n "${KPI:-}" ]] || { echo "[CI] No hay KPIs"; exit 1; }

# Usa la aserción de helpers o un fallback en Python
if typeset -f assert_kpi_has_sats >/dev/null 2>&1; then
  assert_kpi_has_sats "$KPI"
else
  python3 - "$KPI" <<'PY'
import pandas as pd, numpy as np, re, sys
df=pd.read_csv(sys.argv[1], nrows=1)
def to_float(x):
    if x is None: return np.nan
    try: return float(x)
    except: 
        try: return float(re.sub(r'[,\s%]', '', str(x)))
        except: return np.nan
r=df.iloc[0].to_dict() if len(df) else {}
keys=['sats_mult','net_btc_vs_hodl','net_btc_ratio','net_btc','netBTC','net_btc_oos','sats_vs_hodl','roi_sats','roi_vs_hodl']
vals=[to_float(r.get(k)) for k in keys]
ok=any(np.isfinite(v) and not np.isnan(v) for v in vals)
print("[ASSERT] OK: KPI con métrica de sats." if ok else "[ASSERT] FAIL: KPI sin métrica de sats (todas NaN).")
sys.exit(0 if ok else 1)
PY
fi
