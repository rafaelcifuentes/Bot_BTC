#!/usr/bin/env bash
set -euo pipefail
set -a; . ./.env.kpi; set +a

label_of(){ basename "$1" | sed -E 's/.*__//; s/\.csv$//' ; }
LBASE="$(label_of "$BASE_KPI")"; LCAND="$(label_of "$CAND_KPI")"
[[ "$LBASE" == "$LCAND" ]] || { echo "⛔ Ventanas distintas: $LBASE vs $LCAND"; exit 2; }

python3 - <<'PY'
import os, pandas as pd
PREFS=["sats_mult","net_btc_vs_hodl","equity_mult","net_btc","net_sats","cum_mult","net"]
def pick_eq(path):
    df=pd.read_csv(path)
    for k in PREFS:
        if k in df.columns:
            s=pd.to_numeric(df[k], errors="coerce").dropna()
            if len(s): return s.reset_index(drop=True)
    for col in df.columns[::-1]:
        s=pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s): return s.reset_index(drop=True)
    return None
b=os.environ["BASE_KPI"]; c=os.environ["CAND_KPI"]
beq=pick_eq(b); ceq=pick_eq(c)
if beq is not None:
  pd.DataFrame({"ts":range(len(beq)),"equity":beq}).to_csv("/tmp/base_eq_auto.csv", index=False)
if ceq is not None:
  pd.DataFrame({"ts":range(len(ceq)),"equity":ceq}).to_csv("/tmp/cand_eq_auto.csv", index=False)
PY

args=( --base-kpi "$BASE_KPI" --cand-kpi "$CAND_KPI" --lift-min 5 )
[[ -s /tmp/base_eq_auto.csv && -s /tmp/cand_eq_auto.csv ]] && args+=( --base-eq /tmp/base_eq_auto.csv --cand-eq /tmp/cand_eq_auto.csv )

GATE_DEBUG=1 scripts/mini_accum/gate_pair.sh "${args[@]}"
