#!/usr/bin/env bash
set -euo pipefail
ORIG_ARGS=("$@")

# Defaults
LIFT_MIN="${LIFT_MIN:-5}"
ALLOW_MIXED="${ALLOW_MIXED:-0}"

BASE_KPI=""; CAND_KPI=""; BASE_EQ=""; CAND_EQ=""

usage(){
  cat <<USG
Usage: $0 --base-kpi KPI.csv --cand-kpi KPI.csv [--base-eq EQ.csv --cand-eq EQ.csv]
           [--lift-min N] [--allow-mixed]
USG
}

# Arg parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-kpi)  BASE_KPI="$2"; shift 2;;
    --cand-kpi)  CAND_KPI="$2"; shift 2;;
    --base-eq)   BASE_EQ="$2";  shift 2;;
    --cand-eq)   CAND_EQ="$2";  shift 2;;
    --lift-min)  LIFT_MIN="$2"; shift 2;;
    --allow-mixed) ALLOW_MIXED=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "arg desconocido: $1"; usage; exit 2;;
  esac
done

# Si no pasaste args, intenta .env.kpi
if [[ -z "${BASE_KPI}" && -z "${CAND_KPI}" && -f ./.env.kpi ]]; then
  set -a; . ./.env.kpi; set +a
fi

# Debug
[[ "${GATE_DEBUG:-0}" = "1" ]] && {
  echo "[DBG] BASE_KPI=${BASE_KPI:-}"
  echo "[DBG] CAND_KPI=${CAND_KPI:-}"
  echo "[DBG] BASE_EQ=${BASE_EQ:-}"
  echo "[DBG] CAND_EQ=${CAND_EQ:-}"
  echo "[DBG] LIFT_MIN=${LIFT_MIN}"
}

# Helper para etiqueta/ventana desde el nombre de archivo
label_of() {
  local fn; fn="$(basename "$1")"
  local token
  for pat in 'WF_[0-9]{4}' 'oos[0-9]{2}[A-Za-z0-9]+' 'live_[0-9]{4}-[0-9]{2}-[0-9]{2}' 'v[0-9_]+'; do
    token="$(echo "$fn" | grep -Eo "$pat" || true)"
    if [[ -n "$token" ]]; then
      echo "$token"
      return 0
    fi
  done
  # último recurso: sufijo tras "__"
  echo "${fn##*__}" | sed -E 's/\.csv$//'
}

# Apples-to-apples (si hay KPI) salvo override
if [[ -n "${BASE_KPI}" && -n "${CAND_KPI}" ]]; then
  LBASE="$(label_of "$BASE_KPI")"; LCAND="$(label_of "$CAND_KPI")"
  [[ "${GATE_DEBUG:-0}" = "1" ]] && echo "[DBG] LABELS: BASE=$LBASE CAND=$LCAND"
  if [[ "$LBASE" != "$LCAND" && "${ALLOW_MIXED}" != "1" ]]; then
    echo "⛔ Ventanas distintas: BASE=$LBASE vs CAND=$LCAND" >&2
    exit 2
  fi
  [[ "$LBASE" != "$LCAND" && "${ALLOW_MIXED}" = "1" ]] && \
    echo "[WARN] Ventanas distintas: BASE=$LBASE vs CAND=$LCAND (ALLOW_MIXED=1)"
fi

# Core en Python
python3 - "${ORIG_ARGS[@]}" <<'PY'
import os, sys, math, argparse, pandas as pd, numpy as np

# Parse CLI flags (preferred) with env fallback to keep backward-compat
p = argparse.ArgumentParser(add_help=False)
p.add_argument('--base-kpi')
p.add_argument('--cand-kpi')
p.add_argument('--base-eq')
p.add_argument('--cand-eq')
p.add_argument('--lift-min', type=float)
args, _ = p.parse_known_args()

def finite(x):
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False

def _env(k, d=""):
    return os.environ.get(k, d)

BASE_KPI = args.base_kpi or _env("BASE_KPI","")
CAND_KPI = args.cand_kpi or _env("CAND_KPI","")
BASE_EQ  = args.base_eq  or _env("BASE_EQ","")
CAND_EQ  = args.cand_eq  or _env("CAND_EQ","")
LIFT_MIN = args.lift_min if args.lift_min is not None else float(_env("LIFT_MIN","5"))

if os.environ.get("GATE_DEBUG") == "1":
    print("[DBG] PY argv:", " ".join(sys.argv))
    print(f"[DBG] RESOLVED: BASE_KPI={BASE_KPI or '∅'} CAND_KPI={CAND_KPI or '∅'} BASE_EQ={BASE_EQ or '∅'} CAND_EQ={CAND_EQ or '∅'} LIFT_MIN={LIFT_MIN}")


PREF_NET=("sats_mult","net_btc_vs_hodl","equity_mult","net_btc","net_sats","cum_mult","net")
KEYS=("fpy","flips_per_year","flips_total","mdd_vs_hodl","mdd","MDD","mdd_pct","max_drawdown")

def load_kpi(path):
    if not path: return {}
    df=pd.read_csv(path)
    out={}
    # net
    for k in PREF_NET:
        if k in df.columns:
            s=pd.to_numeric(df[k], errors="coerce").dropna()
            if len(s): out["net"]=float(s.iloc[-1]); out["net_key"]=k; break
    # mdd
    for k in ("mdd_vs_hodl","mdd","MDD","mdd_pct","max_drawdown"):
        if k in df.columns:
            s=pd.to_numeric(df[k], errors="coerce").dropna()
            if len(s): out["mdd"]=float(s.iloc[-1]); out["mdd_key"]=k; break
    # fpy/flips
    if "fpy" in df.columns:
        s=pd.to_numeric(df["fpy"], errors="coerce").dropna()
        if len(s): out["fpy"]=float(s.iloc[-1])
    if "flips_total" in df.columns:
        s=pd.to_numeric(df["flips_total"], errors="coerce").dropna()
        if len(s): out["flips"]=int(s.iloc[-1])
    if "flips_per_year" in df.columns:
        s=pd.to_numeric(df["flips_per_year"], errors="coerce").dropna()
        if len(s): out["fpy"]=float(s.iloc[-1])
    return out

def load_eq(path):
    if not path: return None
    df=pd.read_csv(path)
    col = "equity" if "equity" in df.columns else df.columns[-1]
    s=pd.to_numeric(df[col], errors="coerce").dropna()
    return s if len(s) else None

# Helper: check if equity series is valid (≥2 points, last point finite > 0)
def valid_eq(s):
    try:
        return (s is not None) and (len(s) >= 2) and np.isfinite(float(s.iloc[-1])) and (float(s.iloc[-1]) > 0)
    except Exception:
        return False

def mdd_from_equity(eq):
    if eq is None or len(eq)<2: return None
    roll=np.maximum.accumulate(eq.values.astype(float))
    dd = (eq.values/roll) - 1.0
    mdd = -float(np.min(dd))  # positivo
    return mdd

B=load_kpi(BASE_KPI); C=load_kpi(CAND_KPI)
# Equity de respaldo si no hay net en KPI
if "net" not in B or "net" not in C:
    beq = load_eq(BASE_EQ) if BASE_EQ else None
    ceq = load_eq(CAND_EQ) if CAND_EQ else None
    if beq is None or ceq is None:
        print("[GATE] FAIL: KPI sin métrica trazable de equity/net y no se pasó --base-eq/--cand-eq.")
        sys.exit(2)
    if not (valid_eq(beq) and valid_eq(ceq)):
        print("[GATE] FAIL: equity externo inválido (se requieren ≥2 puntos y último > 0).")
        sys.exit(2)
    B["net"]=float(beq.iloc[-1]); C["net"]=float(ceq.iloc[-1])
    B["mdd"]=B.get("mdd", mdd_from_equity(beq))
    C["mdd"]=C.get("mdd", mdd_from_equity(ceq))
    B["net_key"]=B.get("net_key","EQ"); C["net_key"]=C.get("net_key","EQ")

# --- LIFT ---
lift = (C["net"]/B["net"] - 1.0)*100.0
print(f"[NET]  BASE({B.get('net_key','?')})={B['net']:.6f}  CAND({C.get('net_key','?')})={C['net']:.6f}  -> Lift≈ {lift:.2f}%  (min {LIFT_MIN:.2f}%)")
lift_fail = lift < LIFT_MIN
if lift_fail:
    print("[RULE] Lift < min ⇒ FAIL")

# --- MDD ---
mdd_fail=False
if "mdd" in B and "mdd" in C and finite(B["mdd"]) and finite(C["mdd"]):
    print(f"[MDD]  BASE={B['mdd']:.6f}  CAND={C['mdd']:.6f}  -> {'PASS' if C['mdd']<=B['mdd'] else 'FAIL'}")
    mdd_fail = (C["mdd"] > B["mdd"])
else:
    print("[MDD]  N/A")

# --- FRIC ---
fric_fail=False
if "fpy" in B and "fpy" in C and finite(B["fpy"]) and finite(C["fpy"]):
    if C["fpy"] > 2.0*B["fpy"] and lift < LIFT_MIN:
        print(f"[FRIC] FPY_cand={C['fpy']:.3f} > 2× FPY_base={B['fpy']:.3f} y Lift<{LIFT_MIN:.1f}% ⇒ FAIL")
        fric_fail=True
    else:
        print(f"[FRIC] FPY base={B['fpy']:.3f} cand={C['fpy']:.3f} ⇒ OK")
else:
    print("[FRIC] N/A")

fail = lift_fail or mdd_fail or fric_fail
print(f"[GATE] {'PASS' if not fail else 'FAIL'}")
sys.exit(0 if not fail else 2)
PY
