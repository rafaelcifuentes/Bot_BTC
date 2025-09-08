#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs reports profiles cache

if [[ $# -lt 1 ]]; then
  echo "Uso: $0 \"YYYY-MM-DD HH:MM\"   # en UTC (12:10 UTC = 08:10 Montreal en verano)"
  exit 1
fi

FREEZE_END_UTC="$1"
ts=$(date -u +"%Y%m%d_%H%M")

./.venv/bin/python runner_conflict_guard_weekly_v2.py \
  --freeze-end "$FREEZE_END_UTC" \
  --cap-mode "min_leg*1.05" \
  --net-tol 0.20 \
  --seed-weights "0.35,0.65;0.40,0.60;0.45,0.55;0.30,0.70;0.50,0.50" \
  2>&1 | tee "logs/perla_negra_v2_${ts}.log"

# ... todo igual hasta el tee del log ...

ASSET="BTC-USD" \
RUN_ID="perla_$(date -u +%Y%m%d)" \
FREEZE_END="$FREEZE_END_UTC" \
./.venv/bin/python - <<'PY'
import os, pathlib, numpy as np, pandas as pd

ASSET=os.environ.get("ASSET","BTC-USD")
RID=os.environ.get("RUN_ID","perla_run")
FREEZE_END=os.environ.get("FREEZE_END","").strip()
freeze_ts = pd.to_datetime(FREEZE_END, utc=True, errors="coerce")
if not pd.notna(freeze_ts):
    freeze_ts = None  # evita el error dtype vs None

BAND=0.995
TTL=2

# ==== fuente Perla ====
cands=[pathlib.Path("reports/Allocator/perla_for_allocator.csv"),
       pathlib.Path("reports/perla_dashboard_full.csv")]
P_PATH=next((p for p in cands if p.exists()), None)
if P_PATH is None:
    raise SystemExit("No encuentro perla_for_allocator.csv ni perla_dashboard_full.csv")

# ==== OHLC 4h → diario ====
ohlc=pd.read_csv(f"data/ohlc/4h/{ASSET}.csv")
ohlc["timestamp"]=pd.to_datetime(ohlc["timestamp"], utc=True, errors="coerce")
ohlc=ohlc.dropna(subset=["timestamp"]).sort_values("timestamp")
if freeze_ts is not None:
    ohlc=ohlc[ohlc["timestamp"]<=freeze_ts]
ohlc=ohlc.set_index("timestamp")
daily=ohlc.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
ema200=daily["close"].ewm(span=200, adjust=False).mean()

# ==== Perla → normaliza a 4h ====
p=pd.read_csv(P_PATH)
p["timestamp"]=pd.to_datetime(p["timestamp"], utc=True, errors="coerce")
p=p.dropna(subset=["timestamp"]).sort_values("timestamp")
if freeze_ts is not None:
    p=p[p["timestamp"]<=freeze_ts]
p=p.set_index("timestamp")
rcol=next((c for c in p.columns if c.lower() in ("retp_btc","ret_perla","ret_4h","ret","r","return")), None)
if rcol is None:
    raise SystemExit(f"Sin columna de retorno en {P_PATH}")
ret=p[rcol].astype(float)

diffs=ret.index.to_series().diff().dropna()
freq_h=(diffs.dt.total_seconds().mode().iloc[0]/3600.0) if len(diffs) else 4.0
idx4=pd.date_range(ret.index.min().floor("4h"), ret.index.max().ceil("4h"), freq="4h", tz=ret.index.tz)
if 3.5<=freq_h<=4.5:
    ret4=ret.reindex(idx4).fillna(0.0)
elif freq_h<3.5:
    ret4=(1.0+ret).resample("4h").prod()-1.0
    ret4=ret4.reindex(idx4).fillna(0.0)
else:
    lr=np.log1p(ret); ret4=pd.Series(0.0, index=idx4)
    ts=list(lr.index)
    for i, t in enumerate(ts):
        lr_i=lr.iloc[i]; t2=ts[i+1] if i+1<len(ts) else idx4[-1]+pd.Timedelta(hours=4)
        sub=idx4[(idx4>t)&(idx4<=t2)]; n=max(len(sub),1)
        ret4.loc[sub]=np.expm1(lr_i/n)

# ==== banda soft + TTL diario ====
cond=(daily["close"]>=BAND*ema200).astype(int)
cond_ttl=cond.rolling(TTL+1, min_periods=TTL+1).min() if TTL>0 else cond
w=pd.DataFrame({"w":cond_ttl}).reindex(idx4, method="ffill")["w"].fillna(0).astype(int)

out=pd.DataFrame({"timestamp": idx4.tz_convert("UTC"),
                  "ret_perla": ret4.values,
                  "w_perla_ema": w.values})
out["ret_perla_ema"]=out["ret_perla"]*out["w_perla_ema"]

pathlib.Path("reports/heart").mkdir(parents=True, exist_ok=True)
pathlib.Path("reports/Allocator").mkdir(parents=True, exist_ok=True)
out.to_csv(f"reports/heart/{RID}_perla_ema_filter.csv", index=False)
pd.DataFrame({"timestamp": out["timestamp"], "retP_btc_ema": out["ret_perla_ema"]})\
  .to_csv("reports/Allocator/perla_for_allocator_ema.csv", index=False)

# KPIs rápidos
def pf(s): pos=s[s>0].sum(); neg=s[s<0].sum(); return (pos/abs(neg)) if neg<0 else float('inf')
def mdd(s): eq=(1+s).cumprod(); dd=eq/eq.cummax()-1.0; return -dd.min()
def eqv(s): return (1+s).prod()
pf_o, pf_f = pf(out["ret_perla"]), pf(out["ret_perla_ema"])
mdd_o, mdd_f = mdd(out["ret_perla"]), mdd(out["ret_perla_ema"])
eq_o,  eq_f  = eqv(out["ret_perla"]), eqv(out["ret_perla_ema"])
act_f = float(out["w_perla_ema"].mean())
print(f"[Perla EMA soft] PF_o={pf_o:.2f} → PF_f={pf_f:.2f}  ΔPF={pf_f-pf_o:+.2f} | MDD_o={mdd_o:.2%} → MDD_f={mdd_f:.2%} | Equity_o={eq_o:.4f} → Equity_f={eq_f:.4f} | Act={act_f:.0%}")
print(f"[OK] Escritos: reports/heart/{RID}_perla_ema_filter.csv y reports/Allocator/perla_for_allocator_ema.csv  (band={BAND}, ttl_days={TTL})")
PY