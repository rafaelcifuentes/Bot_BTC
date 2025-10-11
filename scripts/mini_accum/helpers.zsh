kiss_top(){ local p="$1" top="${2:-20}";
python3 - "$p" "$top" <<'PY'
import sys,glob,pandas as pd
pat=sys.argv[1]; top=int(sys.argv[2])
rows=[]
for f in sorted(glob.glob(pat)):
    try: df=pd.read_csv(f,nrows=1)
    except Exception as e: 
        print(f"[WARN] no pude leer {f}: {e}", file=sys.stderr); 
        continue
    r=df.iloc[0].to_dict()
    def g(*ks, default=None):
        for k in ks:
            if k in r: return r[k]
        return default
    d={
      'sats_mult': float(g('sats_mult','net_btc_vs_hodl', default='nan')),
      'mdd_vs_hodl': float(g('mdd_vs_hodl','mdd_vs_hodl_ratio', default='nan')),
      'fpy': float(g('flips_per_year', default='nan')),
      'flips_total': float(g('flips_total', default='nan')),
      'file': f
    }
    rows.append(d)
if not rows:
    print('[ERR] No se pudieron extraer KPIs', file=sys.stderr); sys.exit(1)
T=pd.DataFrame(rows).sort_values('sats_mult',ascending=False).head(top).reset_index(drop=True)
print(T.to_string(index=False))
PY
}

kiss_rank_micro(){ local base="${1:-}"; setopt local_options null_glob
  [[ -z "$base" ]] && { local a=(reports/mini_accum/kiss_v1/*_kpis__PT_G200_DD15_RB*_H*_BULL0*.csv)
                         base=$([[ ${#a} -gt 0 ]] && echo reports/mini_accum/kiss_v1 || echo reports/mini_accum); }
  kiss_top "${base}/*_kpis__PT_G200_DD15_RB*_H*_BULL0*.csv" 20
}

kiss_kpi(){ local f="$1"; [[ -f "$f" ]] || { echo "uso: kiss_kpi path/to/_kpis__.csv"; return 1; }
python3 - "$f" <<'PY'
import pandas as pd, sys
df=pd.read_csv(sys.argv[1], nrows=1); r=df.iloc[0]
gm=lambda *k: next((r[x] for x in k if x in r), None)
sats=float(gm('sats_mult','net_btc_vs_hodl'))
mdd=float(gm('mdd_vs_hodl','mdd_vs_hodl_ratio'))
flips=int(gm('flips_total') or 0)
print(f"sats_mult={sats:.6f}  mdd_vs_hodl={mdd:.6f}  flips_total={flips}  file={sys.argv[1]}")
PY
}

# OOS 2025H1 — usa MODO POSICIONAL para evitar el bug del env inline
kiss_oos_2025H1(){ local h="${1:-30}" rb="${2:-1}" dd="${3:-15}" gate=200
  local preset="configs/mini_accum/presets/CORE_2025.yaml"
  local suf="G${gate}_DD${dd}_RB${rb}_H${h}_BULL0"
  bash scripts/mini_accum/run_oos.sh 2025-01-01 2025-06-30 "$preset" "$suf"
}

# Gate de lift y riesgo (+robustez opcional)
kiss_gate_lift(){ local base_glob="$1" cand_glob="$2" min_lift="${3:-5}" strict="${4:-0}"
  setopt local_options null_glob
  [[ -z "$base_glob" || -z "$cand_glob" ]] && { echo "uso: kiss_gate_lift BASE_GLOB CAND_GLOB [min_lift%] [strict 0|1]"; return 1; }
  local base_files=($~base_glob) cand_files=($~cand_glob)
  [[ ${#base_files} -eq 0 ]] && { echo "[ERR] no hay matches para BASE_GLOB: $base_glob"; return 1; }
  [[ ${#cand_files} -eq 0 ]] && { echo "[ERR] no hay matches para CAND_GLOB: $cand_glob"; return 1; }
  local base="${base_files[-1]}" cand="${cand_files[-1]}"
  python3 - "$base" "$cand" "$min_lift" "$strict" <<'PY'
import os, sys, pandas as pd, numpy as np
base, cand, min_lift, strict = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])

def read_kpis(path):
    df = pd.read_csv(path, nrows=1)
    r = df.iloc[0]
    def g(*ks):
        for k in ks:
            if k in r and pd.notna(r[k]): return r[k]
    return dict(
        sats=float(g('sats_mult','net_btc_vs_hodl') or np.nan),
        mdd=float(g('mdd_vs_hodl','mdd_vs_hodl_ratio') or np.nan),
        flips=int(g('flips_total') or 0),
        file=path
    )

B = read_kpis(base); C = read_kpis(cand)
if np.isnan(B['sats']) or np.isnan(C['sats']):
    print("[ERR] KPI inválido (sats_mult) en base o candidato", file=sys.stderr); sys.exit(1)
if np.isnan(B['mdd']) or np.isnan(C['mdd']):
    print("[ERR] KPI inválido (mdd_vs_hodl) en base o candidato", file=sys.stderr); sys.exit(1)

lift = (C['sats']/B['sats'] - 1.0)*100.0
mdd_delta = C['mdd'] - B['mdd']
risk_ok = mdd_delta <= 0.0 + 1e-12  # menor o igual MDD vs base (menor es mejor)

print(f"[BASE] sats={B['sats']:.6f}  mdd={B['mdd']:.6f}  flips={B['flips']}  file={B['file']}")
print(f"[CAND] sats={C['sats']:.6f}  mdd={C['mdd']:.6f}  flips={C['flips']}  file={C['file']}")
print(f"[DIFF] lift={lift:+.2f}%  mdd_delta={mdd_delta:+.6f}")

# Robustez opcional
spearman_ok = None
spearman_csv = os.environ.get("SPEARMAN_CSV")
if spearman_csv and os.path.isfile(spearman_csv):
    try:
        D = pd.read_csv(spearman_csv)
        numcols = [c for c in D.columns if pd.api.types.is_numeric_dtype(D[c])]
        okcols = [c for c in numcols if D[c].notna().sum()>=2]
        if len(okcols) >= 2:
            s = D[okcols].rank(method="average")
            rho = s[okcols[0]].corr(s[okcols[1]], method="spearman")
            spearman_ok = (rho >= 0.95)
            print(f"[ROBUST] spearman_rho={rho:.3f} ({'OK≥0.95' if spearman_ok else 'FAIL'})")
        else:
            print("[ROBUST] spearman SKIP (insuficientes columnas numéricas)")
    except Exception as e:
        print(f"[ROBUST] spearman SKIP ({e})")
else:
    print("[ROBUST] spearman SKIP (no SPEARMAN_CSV)")

pbo_ok = None
pbo_max = os.environ.get("PBO_MAX")
pbo_val = os.environ.get("PBO_VAL")
if pbo_max and pbo_val:
    try:
        pbo_ok = float(pbo_val) <= float(pbo_max)
        print(f"[ROBUST] PBO={float(pbo_val):.3f} ({'OK≤'+str(pbo_max) if pbo_ok else 'FAIL'})")
    except Exception as e:
        print(f"[ROBUST] PBO SKIP ({e})")
elif pbo_max and not pbo_val:
    print("[ROBUST] PBO SKIP (define PBO_VAL)")
else:
    print("[ROBUST] PBO SKIP (no PBO_MAX)")

def decide():
    ok_lift = (lift >= min_lift - 1e-12)
    if strict:
        if spearman_ok is False: return (1,"FAIL: spearman < 0.95")
        if pbo_ok is False: return (1,"FAIL: PBO > max")
        if spearman_ok is None or pbo_ok is None: return (2,"WARN: robustez incompleta (falta spearman o PBO)")
        if not ok_lift: return (1,"FAIL: lift < threshold")
        if not risk_ok: return (1,"FAIL: MDD empeora vs base")
        return (0,"PASS (estricto)")
    else:
        if not ok_lift: return (1,"FAIL: lift < threshold")
        if not risk_ok: return (1,"FAIL: MDD empeora vs base")
        return (0,"PASS")

code,msg = decide()
print(f"[GATE] {msg} | lift≥{min_lift:.2f}%  risk_ok={risk_ok}  spearman_ok={spearman_ok}  pbo_ok={pbo_ok}")
sys.exit(code)
PY
}
