#!/usr/bin/env bash
# scripts/run_heart_slim_pipeline.sh
# Pipeline SLIM: FREEZE por-barra -> baseline_norm -> weights (EMA/ATR%) -> overlay -> KPIs -> snapshot
# Nota: es un smoke test SLIM (no reemplaza el pipeline Corazón+Diamante para ξ* de producción)

set -euo pipefail

# ====== Inputs con defaults (puedes sobreescribir por ENV) ======
ASSET="${ASSET:-BTC-USD}"
ATR_MAX="${ATR_MAX:-0.07}"                          # umbral ATR% (atr/close)
FREEZE_END="${FREEZE_END:-$(date +%F) 00:00}"       # "YYYY-MM-DD 00:00"
RUN_ID="${RUN_ID:-slim_ema200_atrpct_$(date +%Y%m%d)}"

# Exportar para subshells (Python)
export ASSET ATR_MAX FREEZE_END RUN_ID

echo "[INFO] ASSET=${ASSET}  ATR_MAX=${ATR_MAX}  FREEZE_END='${FREEZE_END}'  RUN_ID=${RUN_ID}"

# ====== Rutas ======
mkdir -p data/ohlc/4h reports/heart
OHLC4H="data/ohlc/4h/${ASSET}.csv"
ALT_OHLC4H="reports/ohlc_4h/${ASSET}.csv"

# ====== Asegurar OHLC 4h ======
if [[ ! -f "$OHLC4H" ]]; then
  if [[ -f "$ALT_OHLC4H" ]]; then
    cp "$ALT_OHLC4H" "$OHLC4H"
  elif [[ -f "data/ohlc/1m/${ASSET}.csv" ]]; then
    # Generar 4h desde 1m si existe el crudo
    python scripts/build_ohlc_4h.py \
      --in_csv  "data/ohlc/1m/${ASSET}.csv" \
      --out_csv "$OHLC4H"
  else
    echo "⛔ No encuentro OHLC 4h para ${ASSET} en '$OHLC4H' ni en '$ALT_OHLC4H'." >&2
    exit 1
  fi
fi

# Reporte de barras disponibles
python - <<'PY'
import pandas as pd, os
p = os.environ["OHLC4H"] if "OHLC4H" in os.environ else "data/ohlc/4h/BTC-USD.csv"
df = pd.read_csv(p)
print(f"[OK] 4h -> {p} ({len(df):,} bars)")
PY

# ====== 1) Baseline SLIM (FREEZE) ======
python - <<'PY'
import os, pandas as pd, numpy as np
from datetime import datetime

OHLC4H = os.environ["OHLC4H"]
FREEZE_END = os.environ["FREEZE_END"]
RID = os.environ["RUN_ID"]

df = pd.read_csv(OHLC4H)
# Detectar timestamp
ts_col = "timestamp" if "timestamp" in df.columns else ("ts" if "ts" in df.columns else None)
if ts_col is None:
    raise SystemExit("⛔ No encuentro columna de tiempo (timestamp/ts) en OHLC 4h.")
df[ts_col] = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(None)
df = df.sort_values(ts_col)
if "close" not in df.columns and "Close" in df.columns:
    df["close"] = df["Close"]

# Recorte FREEZE
fe = pd.to_datetime(FREEZE_END)
df = df[df[ts_col] <= fe].copy()

# Retorno 4h
df["ret_4h"] = df["close"].pct_change()
base = df[[ts_col, "ret_4h"]].dropna().rename(columns={ts_col:"timestamp"})

base_path = f"reports/heart/{RID}_baseline.csv"
base_norm_path = f"reports/heart/{RID}_baseline_norm.csv"
base.to_csv(base_path, index=False)
base[["timestamp","ret_4h"]].to_csv(base_norm_path, index=False)

print(f"[OK] Dummy baseline escrito en {base_path} con {len(base):,} filas")
print(f"[OK] Escrito {base_norm_path} con {len(base):,} filas; columnas: timestamp, ret_4h")
PY

# ====== 2) Weights (EMA200 + ATR%) ======
python - <<'PY'
import os, pandas as pd, numpy as np
import pandas_ta as ta

RID = os.environ["RUN_ID"]
ATR_MAX = float(os.environ["ATR_MAX"])
OHLC4H = os.environ["OHLC4H"]

df = pd.read_csv(OHLC4H, parse_dates=["timestamp", "ts"], infer_datetime_format=True)
ts_col = "timestamp" if "timestamp" in df.columns else ("ts" if "ts" in df.columns else None)
if ts_col is None:
    raise SystemExit("⛔ No encuentro columna de tiempo (timestamp/ts) en OHLC 4h.")
df = df.rename(columns={ts_col:"timestamp"})
for c in ["open","high","low","close"]:
    if c not in df.columns and c.capitalize() in df.columns:
        df[c] = df[c.capitalize()]
df = df[["timestamp","open","high","low","close"]].dropna().sort_values("timestamp")

# Indicadores
df["ema200"] = ta.ema(df["close"], length=200)
df["atr14"]  = ta.atr(df["high"], df["low"], df["close"], length=14)
df["atr_pct"] = df["atr14"] / df["close"]
df["ema_slope"] = df["ema200"].diff()

# Reglas SLIM:
#  - base 1.0
#  - si atr_pct > ATR_MAX → 0.7
#  - si ema_slope < 0 y close < ema200 → 0.8 (tendencia floja)
w = []
for _, r in df.iterrows():
    w_i = 1.0
    if pd.notna(r["atr_pct"]) and r["atr_pct"] > ATR_MAX:
        w_i *= 0.7
    if pd.notna(r["ema_slope"]) and pd.notna(r["ema200"]) and r["ema_slope"] < 0 and r["close"] < r["ema200"]:
        w_i *= 0.8
    # clamp a {0.6, 0.8, 1.0} aprox
    w_i = 1.0 if w_i >= 0.95 else (0.8 if w_i >= 0.75 else 0.6)
    w.append(w_i)
df["w_diamante"] = w

w_out = df[["timestamp","w_diamante"]].dropna()
out_path = f"reports/heart/{RID}_weights.csv"
w_out.to_csv(out_path, index=False)
print(f"[OK] weights -> {out_path} (ATR_MAX={ATR_MAX})")
PY

# ====== 3) Overlay ======
python - <<'PY'
import os, pandas as pd, numpy as np

RID = os.environ["RUN_ID"]
bn = f"reports/heart/{RID}_baseline_norm.csv"
wt = f"reports/heart/{RID}_weights.csv"
ov = f"reports/heart/{RID}_overlay.csv"

b = pd.read_csv(bn, parse_dates=["timestamp"])
w = pd.read_csv(wt, parse_dates=["timestamp"])

b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("4h")
w["timestamp"] = pd.to_datetime(w["timestamp"], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("4h")

m = pd.merge_asof(b.sort_values("timestamp"), w.sort_values("timestamp"),
                  on="timestamp", direction="nearest", tolerance=pd.Timedelta("2h"))
m = m.dropna(subset=["ret_4h","w_diamante"]).copy()

m["ret_base"] = m["ret_4h"]
m["ret_overlay"] = m["ret_4h"] * m["w_diamante"]

def equity(ret):
    return (1.0 + ret.fillna(0.0)).cumprod()

m["eq_base"]    = equity(m["ret_base"])
m["eq_overlay"] = equity(m["ret_overlay"])

m.to_csv(ov, index=False)
print(f"[OK] overlay -> {ov} ({len(m)} rows)")
PY

# ====== 4) KPIs + resumen ======
python - <<'PY'
import os, pandas as pd, numpy as np, json

RID = os.environ["RUN_ID"]
ovp = f"reports/heart/{RID}_overlay.csv"
kp = f"reports/heart/{RID}_overlay_vs_base.csv"
md = f"reports/heart/{RID}_overlay_vs_base.md"

m = pd.read_csv(ovp)

def pf(ret):
    pos = ret[ret > 0].sum()
    neg = -ret[ret < 0].sum()
    return float(pos/neg) if neg > 1e-12 else float("inf")

def mdd(eq):
    peak = eq.cummax()
    dd = (eq/peak - 1.0)
    return float(dd.min())

def vol(ret):
    return float(ret.std())

def wr(ret):
    return float((ret > 0).mean() * 100.0)

row = {
  "pf_base": pf(m["ret_base"]),
  "pf_overlay": pf(m["ret_overlay"]),
  "wr_base": wr(m["ret_base"]),
  "wr_overlay": wr(m["ret_overlay"]),
  "mdd_base": mdd(m["eq_base"]),
  "mdd_overlay": mdd(m["eq_overlay"]),
  "vol_base": vol(m["ret_base"]),
  "vol_overlay": vol(m["ret_overlay"]),
  "net_base": float(m["ret_base"].sum()),
  "net_overlay": float(m["ret_overlay"].sum()),
  "rows": int(len(m)),
}
pd.DataFrame([row]).to_csv(kp, index=False)
print(f"[OK] KPIs → {kp}")

with open(md, "w") as f:
    f.write(f"PF_base={row['pf_base']:.2f}  PF_overlay={row['pf_overlay']:.2f}  ΔPF={row['pf_overlay']-row['pf_base']:+.2f}\n")
    if row["mdd_base"] != 0:
        ratio = abs(row["mdd_overlay"])/abs(row["mdd_base"])
        f.write(f"|MDD_overlay|/|MDD_base| = {ratio:.2f}  (base={row['mdd_base']:.2%}, overlay={row['mdd_overlay']:.2%})\n")
print(f"[OK] Resumen → {md}")
PY

# ====== 5) Snapshot JSON simple ======
python - <<'PY'
import os, json
RID = os.environ["RUN_ID"]
snap = {
  "run_id": RID,
  "files": [
    f"reports/heart/{RID}_baseline_norm.csv",
    f"reports/heart/{RID}_weights.csv",
    f"reports/heart/{RID}_overlay.csv",
    f"reports/heart/{RID}_overlay_vs_base.csv",
    f"reports/heart/{RID}_overlay_vs_base.md",
  ],
}
outj = f"reports/heart/{RID}_snapshot.json"
with open(outj,"w") as f: json.dump(snap, f, indent=2)
print(f"[baseline] escrito {outj} con {len(snap['files'])} archivos")
PY

echo "[DONE] Runner SLIM finalizado. Archivos en reports/heart/"

cat >> "scripts/run_heart_slim_pipeline.sh" <<'BASH'
# ───────────────────────────────────────────────────────────────────────────────
# ATR% DAILY WATCHDOG (info-only; no cambia parámetros)
# ───────────────────────────────────────────────────────────────────────────────
# Desc: calcula ATR% diario a partir de tu OHLC 4h y reporta % de días por encima
#       de umbrales 0.07/0.08/0.09. Útil para decidir si re-abrir grid ATR_MAX.

# 0) ROOT del repo (si no viene seteado)
if [ -z "${ROOT:-}" ]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi

# 1) Python del venv (o fallback a python3 del sistema)
PYEXE="${PYEXE:-$ROOT/.venv/bin/python}"
if [ ! -x "$PYEXE" ]; then
  PYEXE="$(command -v python3 || command -v python)"
fi

# 2) Fuente OHLC 4h (si no viene por env, arma ruta por ASSET)
if [ -z "${OHLC4H:-}" ]; then
  ASSET="${ASSET:-BTC-USD}"
  OHLC4H="$ROOT/data/ohlc/4h/${ASSET}.csv"
fi

if [ -x "$PYEXE" ] && [ -f "$OHLC4H" ]; then
  # 3) Script temporal (idempotente; se reescribe)
  cat > /tmp/atr_watchdog_daily.py <<'PY'
import sys, pandas as pd, numpy as np
if len(sys.argv) < 2:
    raise SystemExit("uso: atr_watchdog_daily.py <ohlc_4h_csv>")
ohlc=sys.argv[1]
df=pd.read_csv(ohlc)

# detectar/normalizar timestamp
t=[c for c in df.columns if c.lower() in ("timestamp","ts","time","datetime","date","dt")]
if not t: raise SystemExit("❌ Falta columna de tiempo (timestamp/ts/...)")
t=t[0]
df[t]=pd.to_datetime(df[t], utc=True, errors="coerce")
df=df.rename(columns={t:"timestamp"}).sort_values("timestamp").set_index("timestamp")

# 4h → D1
d=df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()

# TR clásico + ATR14 (SMA) y ATR%
tr=(d["high"]-d["low"]).to_frame("tr")
tr["x1"]=(d["high"]-d["close"].shift()).abs()
tr["x2"]=(d["low"] -d["close"].shift()).abs()
d["tr"]=tr.max(axis=1)
d["atr"]=d["tr"].rolling(14, min_periods=14).mean()
d["atr_pct"]=d["atr"]/d["close"]

def pct_over(a): return float((d["atr_pct"]>a).mean())

p07, p08, p09 = (pct_over(x) for x in (0.07,0.08,0.09))
print(f"[ATR watchdog D1] 0.07→ {p07:.2%} | 0.08→ {p08:.2%} | 0.09→ {p09:.2%}")

# Sugerencia simple en últimos 180d (no cambia nada automáticamente)
recent=d.tail(180)
r07=float((recent["atr_pct"]>0.07).mean())
if r07 > 0.10:
    print(f"[ATR watchdog D1] Sugerencia: considerar ATR_MAX=0.08 (últimos 180d: 0.07 supera el {r07:.1%})")
PY

  # 4) Ejecuta y loguea (no-fatal)
  mkdir -p "$ROOT/reports/heart"
  "$PYEXE" /tmp/atr_watchdog_daily.py "$OHLC4H" 2>&1 | tee -a "$ROOT/reports/heart/atr_watchdog_daily.log" || \
    echo "[WARN] watchdog diario falló (continuo)"
else
  echo "[WARN] watchdog diario no ejecutado (faltan OHLC4H o Python)"
fi
# ───────────────────────────────────────────────────────────────────────────────
BASH