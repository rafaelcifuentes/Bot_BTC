#!/usr/bin/env bash
set -euo pipefail

# Raíz del repo (ajusta si ejecutas desde otro path)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WF_DIR="$REPO_DIR/reports/mini_accum/walkforward"
KPI_GLOB="$REPO_DIR/reports/mini_accum/kiss_v1/"'*kpis__WF_*.csv'
OUT_SUMMARY="$WF_DIR/wf_summary_kpis.csv"
OUT_BEST="$WF_DIR/wf_best_by_window.csv"
OUT_ROADMAP="$WF_DIR/Roadmap_PDCA.md"

# (A) Consolidar WF y chequear criterios de aceptación
python "$REPO_DIR/tools/mini_accum/wf_consolidate.py" \
  --kpis_glob "$KPI_GLOB" \
  --out_summary "$OUT_SUMMARY" \
  --out_best "$OUT_BEST" \
  --out_md "$OUT_ROADMAP" \
  --candidate "DD15_RB1_H30_G200_BULL0" \
  --accept_median_sats 1.05 \
  --accept_fail_rate_max 0.25 \
  --delta_sats_vs_nbhd_min 0.02

# (B) Stress de costes (±5/10/20 bps por lado) → actualiza Roadmap_PDCA.md (sección stress)
python "$REPO_DIR/tools/mini_accum/stress_costs.py" \
  --summary_csv "$OUT_SUMMARY" \
  --out_md_append "$OUT_ROADMAP" \
  --bps 5 10 20

# (C) Tests anti-overfitting (PBO/CSCV + Reality/SPA; DSR si hay datos de Sharpe o retornos)
python "$REPO_DIR/tools/mini_accum/stats_overfit.py" \
  --summary_csv "$OUT_SUMMARY" \
  --out_md_append "$OUT_ROADMAP" \
  --windows WF_2021 WF_2022 WF_2023 WF_2024 WF_2025H1 \
  --candidate "DD15_RB1_H30_G200_BULL0"

echo "[OK] Pipeline KISS v1 completo → $OUT_ROADMAP"

# ---------------------------------------------
# [Optional] A/B semanal en sombra (condicional)
# Corre SOLO si existe un CSV candidato B.
#   - A = reports/mini_accum/walkforward/wf_summary_kpis.csv
#   - B se toma de:
#       1) $AB_CANDIDATE_CSV si está definido y existe
#       2) auto-descubrimiento si hay EXACTAMENTE uno en:
#          reports/mini_accum/experiments/*/wf_summary_kpis.csv
#   - Etiqueta = $AB_LABEL o el nombre del directorio del candidato.
#   - No falla el pipeline si el A/B falla o no hay candidato.
# ---------------------------------------------
(
  set -e
  ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
  A="$ROOT/reports/mini_accum/walkforward/wf_summary_kpis.csv"
  CAND=""
  LABEL="${AB_LABEL:-candidate}"

  # 1) Candidato explícito por variable de entorno
  if [ -n "${AB_CANDIDATE_CSV:-}" ] && [ -f "${AB_CANDIDATE_CSV}" ]; then
    CAND="${AB_CANDIDATE_CSV}"
    LABEL="${AB_LABEL:-$(basename "$(dirname "${CAND}")")}"
  fi

  # 2) Descubrimiento automático: exactamente un candidato en experiments
  if [ -z "$CAND" ]; then
    CANDS=$(find "$ROOT/reports/mini_accum/experiments" -maxdepth 2 -type f -name "wf_summary_kpis.csv" 2>/dev/null | sed -e '/^$/d')
    CNT=$(printf "%s\n" "$CANDS" | wc -l | tr -d ' ')
    if [ "$CNT" -eq 1 ]; then
      CAND=$(printf "%s\n" "$CANDS" | head -n 1)
      LABEL="${AB_LABEL:-$(basename "$(dirname "${CAND}")")}"
    fi
  fi

  if [ -n "$CAND" ] && [ -f "$CAND" ]; then
    echo "[A/B] Ejecutando en sombra → A=$(basename "$A") vs B=$(basename "$CAND") | label=$LABEL"
    if [ -x "$ROOT/scripts/mini_accum/ab_weekly.sh" ]; then
      "$ROOT/scripts/mini_accum/ab_weekly.sh" "$A" "$CAND" "$LABEL" || echo "[WARN] A/B falló (continuo pipeline)."
    else
      echo "[WARN] No existe o no es ejecutable: $ROOT/scripts/mini_accum/ab_weekly.sh (saltando A/B)."
    fi
  else
    echo "[A/B] No se encontró candidato; se omite. (Define AB_CANDIDATE_CSV o coloca exactamente un CSV en reports/mini_accum/experiments/*/wf_summary_kpis.csv)"
  fi
) || echo "[WARN] Bloque A/B condicional encontró un error (continuo pipeline)."

  # ---------------------------------------------
# Tracking Error semanal (vs referencia backtest)
#   - Periodo: lunes pasado 00:00 ET → este lunes 00:00 ET
#   - OHLC: reports/ohlc_4h/BTC-USD.csv  (timestamp,close)
#   - Shadow: reports/mini_accum/exec/orders_preview.csv
#   - Ref: signals/mini_accum/history.csv (si no existe → usa shadow)
#   - Coste por flip: COST_PER_FLIP (default 0.0006 = 0.06%)
#   - Criterio: |TE| ≤ 0.03 (±3%)
# ---------------------------------------------
(
  set -e
  ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
  OUT="$ROOT/reports/mini_accum/walkforward"
  FD="${FREEZE_DATE:-$(TZ=America/New_York date +%F)}"
  WFM="$OUT/freezes/$FD/weekly_freeze_summary.$FD.md"

  OHLC="$ROOT/reports/ohlc_4h/BTC-USD.csv"
  SHADOW="$ROOT/reports/mini_accum/exec/orders_preview.csv"
  REFHIST="$ROOT/signals/mini_accum/history.csv"
  COST_PER_FLIP="${COST_PER_FLIP:-0.0006}"

  if [ ! -f "$OHLC" ] || [ ! -f "$SHADOW" ]; then
    echo "python "$ROOT/scripts/common/normalize_csv.py" --schema ohlc_4h --in "$ROOT/reports/ohlc_4h/BTC-USD.csv" --out "$ROOT/reports/ohlc_4h/BTC-USD.csv" || echo "[TE] WARN normalize OHLC"
python "$ROOT/scripts/common/normalize_csv.py" --schema orders_preview --in "$ROOT/reports/mini_accum/exec/orders_preview.csv" --out "$ROOT/reports/mini_accum/exec/orders_preview.csv" || echo "[TE] WARN normalize orders"
[TE] SKIP (faltan OHLC=$OHLC o SHADOW=$SHADOW)"; exit 0
  fi

  python - "$OHLC" "$SHADOW" "$REFHIST" "$WFM" "$FD" "$COST_PER_FLIP" <<'PY'
import sys, os, pandas as pd, numpy as np
from datetime import timedelta
OHLC, SHADOW, REFHIST, WFM, FD, COST = sys.argv[1:7]
COST=float(COST)

ohlc=pd.read_csv(OHLC)
if "timestamp" not in ohlc.columns or "close" not in ohlc.columns:
    raise SystemExit(f"[TE] CSV OHLC debe tener columnas 'timestamp' y 'close': {OHLC}")
ohlc["timestamp"]=pd.to_datetime(ohlc["timestamp"], utc=True)
t1=pd.Timestamp(FD+" 00:00:00", tz="America/New_York").tz_convert("UTC")
t0=t1-timedelta(days=7)
df=ohlc[(ohlc["timestamp"]>=t0)&(ohlc["timestamp"]<t1)].copy()
if df.empty: print("[TE] SKIP (sin barras en la semana)"); sys.exit(0)
df["ret"]=df["close"].pct_change().fillna(0.0)

def load_pos(path_fallback):
    if os.path.isfile(REFHIST):
        src=REFHIST; pos=pd.read_csv(src)
        if "ts" not in pos.columns or "decision" not in pos.columns: src=path_fallback; pos=pd.read_csv(src)
    else:
        src=path_fallback; pos=pd.read_csv(src)
    pos["ts"]=pd.to_datetime(pos["ts"], utc=True)
    return pos, src

sh=pd.read_csv(SHADOW); sh["ts"]=pd.to_datetime(sh["ts"], utc=True)
rf, src_ref = load_pos(SHADOW)

    def weekly(s, w):
    # [TE KISS] Guard: índice UTC, único y ordenado antes de reindex
    import pandas as pd
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    s.index = pd.to_datetime(s.index, utc=True, errors='coerce')
    s = s[~s.index.duplicated(keep='last')].sort_index()
    ev=net_src[net_src["ts"]<=t1].copy().sort_values("ts")
    prev=net_src[net_src["ts"]<t0].tail(1)
    if len(prev)==1: ev=pd.concat([prev,ev],ignore_index=True)
    if ev.empty: ev=pd.DataFrame([{"ts":t0,"decision":0}])
    pos=ev.set_index("ts")["decision"].astype(int).reindex(df["timestamp"]).ffill().bfill().astype(int)
    flips=(pos!=pos.shift(1).fillna(pos.iloc[0])).sum()
    gross=(1.0+(pos.shift(1).fillna(pos.iloc[0])*df["ret"])).prod()-1.0
    return float(gross - flips*COST), int(flips)

r_shadow, f_shadow = weekly(sh)
r_ref,    f_ref    = weekly(rf)
te=r_shadow - r_ref
status = "PASS" if abs(te) <= 0.03 else "FAIL"
line=(f"\n> Tracking Error (semana {t0.date()}→{t1.date()}, ref={os.path.basename(src_ref)}) — "
      f"shadow={r_shadow:+.2%}, ref={r_ref:+.2%}, TE={te:+.2%} → **{status}**\n")
try:
    with open(WFM,"a") as f: f.write(line)
    print("[TE] anotado en", os.path.basename(WFM), "|", line.strip())
except Exception as e:
    print("[TE] WARN no pude escribir:", e, "|", line.strip())
PY
) || echo "[TE] WARN: bloque TE falló (continuo pipeline)."