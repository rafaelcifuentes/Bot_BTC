if [[ -n "${BASH_VERSION:-}" ]]; then
  echo "[WARN] helpers.zsh requiere zsh; omitiendo porque estás en bash" >&2
  return 0 2>/dev/null || exit 0
fi
emulate -L zsh
kiss_kpi(){ local f="$1"; [[ -f "$f" ]] || { echo "uso: kiss_kpi path/to/_kpis__.csv"; return 1; }
python3 - "$f" <<'PY'
import pandas as pd, sys
df=pd.read_csv(sys.argv[1], nrows=1); r=df.iloc[0].to_dict()

def get_first(*keys):
    for k in keys:
        if k in r and pd.notna(r[k]):
            v = r[k]
            try:
                return float(v)
            except Exception:
                try:
                    return float(str(v).replace(',', '').strip())
                except Exception:
                    return float('nan')
    return float('nan')

sats  = get_first('sats_mult','net_btc_vs_hodl','net_btc_ratio','net_btc','netBTC','net_btc_oos','sats_vs_hodl','roi_sats','roi_vs_hodl')
mdd   = get_first('mdd_vs_hodl','mdd_vs_hodl_ratio')
flips = r.get('flips_total', r.get('flips', r.get('trades_total', 0)))
try:
    flips = int(flips) if pd.notna(flips) and str(flips).strip()!='' else 0
except Exception:
    flips = 0

print(f"sats_mult={sats:.6f}  mdd_vs_hodl={mdd:.6f}  flips_total={flips}  file={sys.argv[1]}")
PY
}


# --- Assert: KPI tiene alguna métrica de sats válida ---
# uso: assert_kpi_has_sats path/to/_kpis__.csv
assert_kpi_has_sats () {
  local f="$1"
  [[ -f "$f" ]] || { echo "[ASSERT] KPI no existe: $f"; return 1; }
  python3 - "$f" <<'PY'
import pandas as pd, numpy as np, sys, re
path=sys.argv[1]
df=pd.read_csv(path, nrows=1)
if df.shape[0]==0:
    print("[ASSERT] FAIL: KPI vacío.")
    sys.exit(1)
r=df.iloc[0].to_dict()

def to_float(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        try:
            return float(re.sub(r'[,\s%]', '', str(x)))
        except Exception:
            return np.nan

keys = ['sats_mult','net_btc_vs_hodl','net_btc_ratio','net_btc','netBTC','net_btc_oos','sats_vs_hodl','roi_sats','roi_vs_hodl']
vals = [to_float(r.get(k)) for k in keys]
ok = any(np.isfinite(v) and not np.isnan(v) for v in vals)

print("[ASSERT] OK: KPI con métrica de sats." if ok else "[ASSERT] FAIL: KPI sin métrica de sats (todas NaN).")
sys.exit(0 if ok else 1)
PY
}

# --- Utilidad: tomar el último archivo que matchea un glob ---
# uso: pick_latest "reports/mini_accum/*_kpis__OOS_*.csv"
pick_latest () {
  local pattern="$1"
  setopt local_options null_glob
  local arr=($~pattern)
  if [[ ${#arr} -eq 0 ]]; then
    echo ""
  else
    echo "${arr[-1]}"
  fi
}

# --- Gate seguro (verifica sats antes de evaluar lift) ---
# uso: safe_gate BASE.csv CAND.csv [label]  (usa umbral 5% y strict=0)
safe_gate () {
  local BASE="$1" CAND="$2" LABEL="${3:-CAND}"
  [[ -f "$BASE" ]] || { echo "[GATE:${LABEL}] base no existe: $BASE"; return 1; }
  [[ -f "$CAND" ]] || { echo "[GATE:${LABEL}] cand no existe: $CAND"; return 1; }

  # Aserciones de métricas de sats (evita falsos positivos y NaN)
  assert_kpi_has_sats "$BASE"  >/dev/null || { echo "[GATE:${LABEL}] skip (BASE sin sats)"; return 1; }
  assert_kpi_has_sats "$CAND"  >/dev/null || { echo "[GATE:${LABEL}] skip (KPI sin sats)";  return 1; }

  # Trazabilidad básica
  kiss_kpi "$BASE"
  kiss_kpi "$CAND"

  # Gate KISS (≥ +5%, no estricto por defecto)
  kiss_gate_lift "$BASE" "$CAND" 5 0
}

# Gate de lift entre dos KPIs (BASE vs CAND)
# uso: kiss_gate_lift BASE_GLOB CAND_GLOB [min_lift%=5] [strict=0|1]
kiss_gate_lift () {
  local base_glob="$1" cand_glob="$2" min_lift="${3:-5}" strict="${4:-0}"
  setopt local_options null_glob

  [[ -z "$base_glob" || -z "$cand_glob" ]] && {
    echo "uso: kiss_gate_lift BASE_GLOB CAND_GLOB [min_lift%] [strict 0|1]"
    return 1
  }

  local base_files=($~base_glob) cand_files=($~cand_glob)
  [[ ${#base_files} -eq 0 ]] && { echo "[ERR] no hay matches para BASE_GLOB: $base_glob"; return 1; }
  [[ ${#cand_files} -eq 0 ]] && { echo "[ERR] no hay matches para CAND_GLOB: $cand_glob"; return 1; }

  local base="${base_files[-1]}" cand="${cand_files[-1]}"

  python3 - "$base" "$cand" "$min_lift" "$strict" <<'PY'
import os, sys, re, pandas as pd, numpy as np

base, cand, min_lift, strict = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])

def to_float(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        try:
            return float(re.sub(r'[,\s%]', '', str(x)))
        except Exception:
            return np.nan

def read_kpis(path):
    df = pd.read_csv(path, nrows=1)
    r = df.iloc[0].to_dict()
    def g(keys):
        for k in keys:
            if k in r and pd.notna(r[k]):
                return r[k]
        return None

    # Campos alternativos (robustos) para cada métrica
    sats  = to_float(g(['sats_mult','net_btc_vs_hodl','net_btc_ratio','net_btc','netBTC','net_btc_oos','sats_vs_hodl','roi_sats','roi_vs_hodl']))
    mdd   = to_float(g(['mdd_vs_hodl','mdd_vs_hodl_ratio','mdd_ratio','max_drawdown_vs_hodl']))
    flips_raw = g(['flips_total','flips','trades_total'])
    try:
        flips = int(float(str(flips_raw).strip())) if flips_raw not in (None,'') else 0
    except Exception:
        flips = 0

    return dict(sats=sats, mdd=mdd, flips=flips, file=path)

B = read_kpis(base)
C = read_kpis(cand)

# Prints básicos para trazabilidad
print(f"[BASE] sats={B['sats'] if not np.isnan(B['sats']) else float('nan'):.6f}  mdd={B['mdd'] if not np.isnan(B['mdd']) else float('nan'):.6f}  flips={B['flips']}  file={B['file']}")
print(f"[CAND] sats={C['sats'] if not np.isnan(C['sats']) else float('nan'):.6f}  mdd={C['mdd'] if not np.isnan(C['mdd']) else float('nan'):.6f}  flips={C['flips']}  file={C['file']}")

# Validaciones mínimas
if np.isnan(B['sats']) or np.isnan(C['sats']):
    # Sin métrica de sats no hay decisión de lift — mantenemos FAIL pero sin traceback
    print("[GATE] FAIL: candidato/base sin métricas de sats (sats_mult/net_btc_vs_hodl). No trazable para promoción.")
    sys.exit(1)
if np.isnan(B['mdd']) or np.isnan(C['mdd']):
    print("[GATE] FAIL: candidato/base sin métrica de MDD. No trazable para control de riesgo.")
    sys.exit(1)

lift = (C['sats']/B['sats'] - 1.0) * 100.0
mdd_delta = C['mdd'] - B['mdd']
risk_ok = mdd_delta <= 1e-12  # menor o igual MDD vs base

print(f"[DIFF] lift={lift:+.2f}%  mdd_delta={mdd_delta:+.6f}")

# Robustez opcional
spearman_ok = None
spearman_csv = os.environ.get("SPEARMAN_CSV")
if spearman_csv and os.path.isfile(spearman_csv):
    try:
        D = pd.read_csv(spearman_csv)
        numcols = [c for c in D.columns if pd.api.types.is_numeric_dtype(D[c])]
        okcols = [c for c in numcols if D[c].notna().sum() >= 2]
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

code, msg = decide()
print(f"[GATE] {msg} | lift≥{min_lift:.2f}%  risk_ok={risk_ok}  spearman_ok={spearman_ok}  pbo_ok={pbo_ok}")
sys.exit(code)
PY
}
# --- renombra el último batch de artefactos añadiendo un sufijo seguro ---
rename_last_reports(){ # uso: rename_last_reports "__WF_2025_v1_2"
  local suf="$1"
  local base="$(ls -1t reports/mini_accum/*_equity.csv 2>/dev/null | head -n1)"
  [[ -z "$base" ]] && { echo "[WARN] No hay artefactos para renombrar"; return 1; }
  local pfx="${base%_equity.csv}"
  for kind in equity kpis summary flips; do
    local src="${pfx}_${kind}.csv"; [[ "$kind" == "summary" ]] && src="${pfx}_${kind}.md"
    [[ -f "$src" ]] || continue
    local ext=$([[ "$kind" == "summary" ]] && echo md || echo csv)
    local dst="${pfx}_${kind}${suf}.${ext}"
    mv "$src" "$dst" && echo "[OK] ${kind} → ${dst}"
  done
}
# --- Gate y log: no promueve si falla el guardián ---
# uso: gate_or_log BASE.csv CAND.csv "label"
gate_or_log () {
  local BASE="$1" CAND="$2" LABEL="$3"
  if [[ -z "$CAND" ]]; then
    echo "[INFO] No hay candidato para $LABEL"
    return 0
  fi
  if ! safe_gate "$BASE" "$CAND" "$LABEL"; then
    echo "[LOG] $LABEL : FAIL gate (mantener BASE)"
  fi
}
