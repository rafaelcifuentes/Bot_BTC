# corazon_cmds.zsh (versión completa, segura)
emulate -L zsh
setopt no_aliases

: ${EXCHANGE:=binanceus}
: ${FREEZE:="2025-09-01 00:00"}
: ${FG:=./data/sentiment/fear_greed.csv}
: ${FU:=./data/sentiment/funding_rates.csv}
: ${OUT:=reports}
: ${MAX_BARS:=975}
: ${THRESH:=0.60}
: ${ADX1D_LEN:=14}
: ${ADX1D_MIN:=22}
: ${ADX4_MIN:=12}
: ${FG_LONG_MIN:=-0.15}
: ${FG_SHORT_MAX:=0.15}
: ${FUNDING_BIAS:=0.005}

_repo_check() { [[ -d scripts && -d reports ]] || echo "[WARN] Ejecuta desde la raíz del repo."; }
_need() { local p="$1" h="$2"; [[ -e "$p" ]] || { echo "⛔ Falta: $p"; echo "   Pista: $h"; return 1; }; }
_py() { command -v python >/dev/null || { echo "⛔ python no encontrado"; return 1; }; python "$@"; }

runC_ping() { echo "[Corazon] functions loaded OK"; }

runC_status() {
  _repo_check
  cat <<EOF

[Corazón :: estado]
EXCHANGE=$EXCHANGE
FREEZE=$FREEZE
FG=$FG
FU=$FU
OUT=$OUT
THRESH=$THRESH  MAX_BARS=$MAX_BARS
ADX1D_LEN=$ADX1D_LEN  ADX1D_MIN=$ADX1D_MIN  ADX4_MIN=$ADX4_MIN
FG_LONG_MIN=$FG_LONG_MIN  FG_SHORT_MAX=$FG_SHORT_MAX  FUNDING_BIAS=$FUNDING_BIAS

Funciones disponibles:
  runC_all_freeze
  runC_long_freeze_v2
  runC_auto_daily
  runC_auto_status
  runH_build_weights
  runH_apply_overlay <csv_diamante>
  runH_report [baseline_csv overlay_csv]
EOF
}

# Runners (Corazón)
runC_all_freeze() {
  _repo_check
  _need scripts/corazon_auto.py "Requiere scripts/corazon_auto.py"
  _py scripts/corazon_auto.py \
    --exchange "$EXCHANGE" --fg_csv "$FG" --funding_csv "$FU" \
    --max_bars "$MAX_BARS" --freeze_end "$FREEZE" \
    --no_gates --label ALL_FREEZE \
    --report_csv "$OUT/corazon_metrics_auto_all.csv"
}
runC_long_freeze_v2() {
  _repo_check
  _need scripts/corazon_auto.py "Requiere scripts/corazon_auto.py"
  _py scripts/corazon_auto.py \
    --exchange "$EXCHANGE" --fg_csv "$FG" --funding_csv "$FU" \
    --max_bars "$MAX_BARS" --freeze_end "$FREEZE" \
    --adx1d_len "$ADX1D_LEN" --adx1d_min "$ADX1D_MIN" --adx4_min "$ADX4_MIN" \
    --fg_long_min "$FG_LONG_MIN" --fg_short_max "$FG_SHORT_MAX" \
    --funding_bias "$FUNDING_BIAS" --threshold "$THRESH" \
    --label LONG_V2_FREEZE \
    --report_csv "$OUT/corazon_metrics_auto_longv2.csv"
}
runC_auto_daily() {
  _repo_check
  _need scripts/corazon_auto.py "Requiere scripts/corazon_auto.py"
  _py scripts/corazon_auto.py \
    --exchange "$EXCHANGE" --fg_csv "$FG" --funding_csv "$FU" \
    --max_bars "$MAX_BARS" --freeze_end "$FREEZE" \
    --compare_both \
    --adx1d_len "$ADX1D_LEN" --adx1d_min "$ADX1D_MIN" --adx4_min "$ADX4_MIN" \
    --fg_long_min "$FG_LONG_MIN" --fg_short_max "$FG_SHORT_MAX" \
    --funding_bias "$FUNDING_BIAS" --threshold "$THRESH" \
    --report_csv "$OUT/corazon_auto_daily.csv"
}
runC_auto_status() {
  _repo_check
  local f="$OUT/corazon_auto_daily.csv"
  [[ -f "$f" ]] && { echo; tail -n 20 "$f"; echo; } || echo "Aún no existe $f. Ejecuta runC_auto_daily primero."
}

# Modo sombra (overlay)
runH_build_weights() {
  _repo_check
  _need scripts/heart_overlay.py "Falta scripts/heart_overlay.py"
  _need configs/heart_rules.yaml "Falta configs/heart_rules.yaml"
  _need "$FG" "Falta fear_greed.csv (FG)"
  _need "$FU" "Falta funding_rates.csv (FU)"
  mkdir -p "$OUT/heart" "$OUT/ohlc_4h"
  _py scripts/heart_overlay.py \
    --prices_csv "$OUT/ohlc_4h/BTC-USD.csv" \
    --fg_csv "$FG" --funding_csv "$FU" \
    --rules_yaml configs/heart_rules.yaml \
    --out_csv "$OUT/heart/w_diamante.csv"
}
runH_apply_overlay() {
  _repo_check
  _need scripts/apply_heart_overlay.py "Falta scripts/apply_heart_overlay.py"
  _need "$OUT/heart/w_diamante.csv" "Ejecuta runH_build_weights primero"
  local WEEK_CSV=${1:-"$OUT/diamante_btc_week1.csv"}
  _need "$WEEK_CSV" "Provee el CSV de Diamante (p.ej. $OUT/diamante_btc_week1.csv)"
  mkdir -p "$OUT/heart"
  _py scripts/apply_heart_overlay.py \
    --diamante_csv "$WEEK_CSV" \
    --weights_csv  "$OUT/heart/w_diamante.csv" \
    --out_csv      "$OUT/heart/diamante_overlay_$(basename "$WEEK_CSV")"
}
runH_report() {
  _repo_check
  _need scripts/report_heart_vs_baseline.py "Falta scripts/report_heart_vs_baseline.py"
  local BASE=${1:-"$OUT/diamante_btc_week1.csv"}
  local OVER=${2:-"$OUT/heart/diamante_overlay_$(basename "$BASE")"}
  _need "$BASE" "CSV baseline Diamante no encontrado"
  _need "$OVER" "CSV overlay no encontrado"
  mkdir -p "$OUT/heart"
  _py scripts/report_heart_vs_baseline.py \
    --baseline_csv "$BASE" \
    --overlay_csv  "$OVER" \
    --out_md       "$OUT/heart/summary_$(basename "$BASE" .csv).md" \
    --out_csv      "$OUT/heart/kpis_$(basename "$BASE")"
}
# --- Corazón: helpers de monitoreo diario (LIVE) ---

runC_export_live() {
  set -euo pipefail
  local dt="${1:-$(date +%F)}"
  export CORAZON_EXPORT_BARS=1
  EXCHANGE="${EXCHANGE:-binanceus}" python swing_4h_forward_diamond.py --skip_yf \
    --symbol BTC-USD --period 730d --horizons 30,60,90 \
    --slip 0.0002 --cost 0.0004 \
    --out_csv "reports/diamante_btc_live_${dt}.csv"
  echo "[OK] export LIVE -> reports/diamante_btc_live_${dt}_bars.csv"
}

runC_shadow_daily() {
  set -euo pipefail
  local dt="${1:-$(date +%F)}"
  local freeze_date="${FREEZE%% *:-}"
  local rules_snap="configs/heart_rules_${freeze_date}.yaml"
  [[ -f "$rules_snap" ]] || { echo "⚠️  snapshot YAML no encontrado (${rules_snap}), usando configs/heart_rules.yaml"; rules_snap="configs/heart_rules.yaml"; }

  # 1) Pesos LIVE (con YAML congelado)
  python scripts/corazon_weights_generator.py \
    --rules "$rules_snap" \
    --ohlc  reports/ohlc_4h/BTC-USD.csv \
    --diamante signals/diamante.csv \
    --perla    signals/perla.csv \
    --out_weights "corazon/weights_live_${dt}.csv" \
    --out_lq     "corazon/lq_live_${dt}.csv"

  mkdir -p reports/heart
  cp "corazon/weights_live_${dt}.csv" reports/heart/w_diamante.csv

  # 2) Baseline por-barra LIVE
  local base="reports/diamante_btc_live_${dt}_bars.csv"
  if [[ ! -f "$base" ]]; then
    if [[ -f reports/diamante_btc_costes_week1_bars.csv ]]; then
      cp reports/diamante_btc_costes_week1_bars.csv "$base"
      echo "[alias] $base <- reports/diamante_btc_costes_week1_bars.csv"
    else
      echo "⛔ No encuentro baseline por-barra: $base"; return 1
    fi
  fi

  # 3) Overlay
  runH_apply_overlay "$base"

  # 4) KPIs + registro PASS/FAIL y ξ*
  python scripts/report_heart_vs_baseline.py \
    --baseline_csv "$base" \
    --overlay_csv  "reports/heart/diamante_overlay_diamante_btc_live_${dt}_bars.csv" \
    --out_md  "reports/heart/summary_diamante_btc_live_${dt}_bars.md" \
    --out_csv "reports/heart/kpis_diamante_btc_live_${dt}_bars.csv" \
    --ts_col timestamp

  python - "$dt" <<'PY'
import pandas as pd, sys
dt = sys.argv[1]
k = pd.read_csv(f"reports/heart/kpis_diamante_btc_live_{dt}_bars.csv").iloc[0]
pf_ok  = (k['pf_overlay'] >= 0.90*k['pf_base']) and (k['pf_overlay'] <= 1.10*k['pf_base'])
mdd_ok = abs(k['mdd_overlay']) <= abs(k['mdd_base'])
vol_ok = k['vol_overlay'] <= k['vol_base']
status = "PASS" if (pf_ok and mdd_ok and vol_ok) else "FAIL"

mdd_ratio = abs(k["mdd_base"]) / max(1e-12, abs(k["mdd_overlay"]))
vol_ratio = k["vol_base"] / max(1e-12, k["vol_overlay"])
xi = min(mdd_ratio, vol_ratio) * 0.85

# Log status
rowS = pd.DataFrame([{"ts":dt,
  "pf_b":k['pf_base'],"pf_o":k['pf_overlay'],
  "mdd_b":k['mdd_base'],"mdd_o":k['mdd_overlay'],
  "vol_b":k['vol_base'],"vol_o":k['vol_overlay'],
  "net_b":k['net_base'],"net_o":k['net_overlay'],
  "status":status
}])
try:
    ds = pd.read_csv("corazon/daily_status.csv"); ds = pd.concat([ds,rowS], ignore_index=True)
except FileNotFoundError:
    ds = rowS
ds.to_csv("corazon/daily_status.csv", index=False)

# Log xi*
rowX = pd.DataFrame([{"ts":dt,
  "pf_b":k['pf_base'],"pf_o":k['pf_overlay'],
  "mdd_b":k['mdd_base'],"mdd_o":k['mdd_overlay'],
  "vol_b":k['vol_base'],"vol_o":k['vol_overlay'],
  "net_b":k['net_base'],"net_o":k['net_overlay'],
  "xi_star":xi
}])
try:
    dx = pd.read_csv("corazon/daily_xi.csv"); dx = pd.concat([dx,rowX], ignore_index=True)
except FileNotFoundError:
    dx = rowX
dx.to_csv("corazon/daily_xi.csv", index=False)

print(f"[{status}] KPIs LIVE {dt} | xi*={xi:.4f}x")
PY
}
# ===========================
# Corazón: automations útiles
# ===========================

# Exporta baseline por-barra LIVE (hoy o fecha pasada como YYYY-MM-DD)
runC_export_live() {
  set -euo pipefail
  local dt="${1:-$(date +%F)}"
  export CORAZON_EXPORT_BARS=1
  EXCHANGE="${EXCHANGE:-binanceus}" python swing_4h_forward_diamond.py --skip_yf \
    --symbol BTC-USD --period 730d --horizons 30,60,90 \
    --slip 0.0002 --cost 0.0004 \
    --out_csv "reports/diamante_btc_live_${dt}.csv"
  echo "[OK] export LIVE -> reports/diamante_btc_live_${dt}_bars.csv"
}

# LUNES — FREEZE semanal end-to-end (un solo comando)
# Requiere: FREEZE="YYYY-MM-DD 00:00"
runC_shadow_weekly_freeze() {
  set -euo pipefail
  if [[ -z "${FREEZE:-}" ]]; then
    echo "⛔ Define FREEZE primero. Ej: export FREEZE=\"$(date +%F) 00:00\""; return 1
  fi
  local fd="${FREEZE%% *}"             # YYYY-MM-DD
  local rules_snap="configs/heart_rules_${fd}.yaml"

  echo "[1] Snapshot de reglas -> ${rules_snap}"
  cp configs/heart_rules.yaml "$rules_snap" || true
  mkdir -p corazon
  (git rev-parse --short HEAD || echo unknown) > "corazon/RULES_${fd}.txt"

  echo "[2] Pesos FREEZE (semaforo/LQ/corr-gate)"
  python scripts/corazon_weights_generator.py \
    --rules "$rules_snap" \
    --ohlc  reports/ohlc_4h/BTC-USD.csv \
    --diamante signals/diamante.csv \
    --perla    signals/perla.csv \
    --out_weights "corazon/weights_${fd}.csv" \
    --out_lq     "corazon/lq_${fd}.csv"

  mkdir -p reports/heart
  cp "corazon/weights_${fd}.csv" reports/heart/w_diamante.csv

  echo "[3] Export baseline por-barra con FREEZE=$FREEZE"
  export CORAZON_EXPORT_BARS=1
  EXCHANGE="${EXCHANGE:-binanceus}" python swing_4h_forward_diamond.py --skip_yf \
    --symbol BTC-USD --period 730d --horizons 30,60,90 \
    --freeze_end "$FREEZE" \
    --slip 0.0002 --cost 0.0004 \
    --out_csv "reports/diamante_btc_costes_freeze_${fd}.csv"
  local base="reports/diamante_btc_costes_freeze_${fd}_bars.csv"
  [[ -f "$base" ]] || { echo "⛔ No encontré $base"; return 1; }

  echo "[4] Overlay + KPIs (FREEZE)"
  runH_apply_overlay "$base"
  python scripts/report_heart_vs_baseline.py \
    --baseline_csv "$base" \
    --overlay_csv  "reports/heart/diamante_overlay_diamante_btc_costes_freeze_${fd}_bars.csv" \
    --out_md       "reports/heart/summary_diamante_btc_costes_freeze_${fd}_bars.md" \
    --out_csv      "reports/heart/kpis_diamante_btc_costes_freeze_${fd}_bars.csv" \
    --ts_col timestamp

  echo "[5] PASS/FAIL + ξ* (FREEZE)"
  python - "$fd" <<'PY'
import pandas as pd, sys
fd = sys.argv[1]
k = pd.read_csv(f"reports/heart/kpis_diamante_btc_costes_freeze_{fd}_bars.csv").iloc[0]
pf_ok  = (k['pf_overlay'] >= 0.90*k['pf_base']) and (k['pf_overlay'] <= 1.10*k['pf_base'])
mdd_ok = abs(k['mdd_overlay']) <= abs(k['mdd_base'])
vol_ok = k['vol_overlay'] <= k['vol_base']
status = "PASS" if (pf_ok and mdd_ok and vol_ok) else "FAIL"

mdd_ratio = abs(k["mdd_base"]) / max(1e-12, abs(k["mdd_overlay"]))
vol_ratio = k["vol_base"] / max(1e-12, k["vol_overlay"])
xi = min(mdd_ratio, vol_ratio) * 0.85

# Log semanal (FREEZE)
import time
row = pd.DataFrame([{
  "ts": fd, "freeze": fd + " 00:00",
  "pf_b":k['pf_base'],"pf_o":k['pf_overlay'],
  "mdd_b":k['mdd_base'],"mdd_o":k['mdd_overlay'],
  "vol_b":k['vol_base'],"vol_o":k['vol_overlay'],
  "net_b":k['net_base'],"net_o":k['net_overlay'],
  "status":status, "xi_star":xi
}])
for path in ["corazon/weekly_status.csv","corazon/weekly_xi.csv"]:
    try:
        df = pd.read_csv(path); df = pd.concat([df,row], ignore_index=True)
    except FileNotFoundError:
        df = row
    df.to_csv(path, index=False)

print(f"[{status}] FREEZE {fd} | xi*={xi:.4f}x")
PY
}

# DIARIO — snapshot LIVE end-to-end (export + pesos LIVE + overlay + KPIs + logs)
runC_shadow_daily() {
  set -euo pipefail
  local dt="${1:-$(date +%F)}"
  local freeze_day="${FREEZE%% *-fallback}"
  # preferir snapshot de reglas del FREEZE si existe
  local rules_snap="configs/heart_rules_${freeze_day}.yaml"
  [[ -f "$rules_snap" ]] || rules_snap="configs/heart_rules.yaml"

  echo "[1] Export LIVE por-barra"
  runC_export_live "$dt"

  echo "[2] Pesos LIVE (YAML snapshot si existe)"
  python scripts/corazon_weights_generator.py \
    --rules "$rules_snap" \
    --ohlc  reports/ohlc_4h/BTC-USD.csv \
    --diamante signals/diamante.csv \
    --perla    signals/perla.csv \
    --out_weights "corazon/weights_live_${dt}.csv" \
    --out_lq     "corazon/lq_live_${dt}.csv"
  mkdir -p reports/heart
  cp "corazon/weights_live_${dt}.csv" reports/heart/w_diamante.csv

  echo "[3] Overlay + [4] KPIs LIVE"
  local base="reports/diamante_btc_live_${dt}_bars.csv"
  runH_apply_overlay "$base"
  python scripts/report_heart_vs_baseline.py \
    --baseline_csv "$base" \
    --overlay_csv  "reports/heart/diamante_overlay_diamante_btc_live_${dt}_bars.csv" \
    --out_md  "reports/heart/summary_diamante_btc_live_${dt}_bars.md" \
    --out_csv "reports/heart/kpis_diamante_btc_live_${dt}_bars.csv" \
    --ts_col timestamp

  echo "[5] PASS/FAIL + ξ* (LIVE)"
  python - "$dt" <<'PY'
import pandas as pd, sys, time
dt = sys.argv[1]
k = pd.read_csv(f"reports/heart/kpis_diamante_btc_live_{dt}_bars.csv").iloc[0]
pf_ok  = (k['pf_overlay'] >= 0.90*k['pf_base']) and (k['pf_overlay'] <= 1.10*k['pf_base'])
mdd_ok = abs(k['mdd_overlay']) <= abs(k['mdd_base'])
vol_ok = k['vol_overlay'] <= k['vol_base']
status = "PASS" if (pf_ok and mdd_ok and vol_ok) else "FAIL"

mdd_ratio = abs(k["mdd_base"]) / max(1e-12, abs(k["mdd_overlay"]))
vol_ratio = k["vol_base"] / max(1e-12, k["vol_overlay"])
xi = min(mdd_ratio, vol_ratio) * 0.85

# Log diarios
rowS = pd.DataFrame([{"ts":dt,
  "pf_b":k['pf_base'],"pf_o":k['pf_overlay'],
  "mdd_b":k['mdd_base'],"mdd_o":k['mdd_overlay'],
  "vol_b":k['vol_base'],"vol_o":k['vol_overlay'],
  "net_b":k['net_base'],"net_o":k['net_overlay'],
  "status":status
}])
try: ds = pd.read_csv("corazon/daily_status.csv"); ds = pd.concat([ds,rowS], ignore_index=True)
except FileNotFoundError: ds = rowS
ds.to_csv("corazon/daily_status.csv", index=False)

rowX = pd.DataFrame([{"ts":dt,
  "pf_b":k['pf_base'],"pf_o":k['pf_overlay'],
  "mdd_b":k['mdd_base'],"mdd_o":k['mdd_overlay'],
  "vol_b":k['vol_base'],"vol_o":k['vol_overlay'],
  "net_b":k['net_base'],"net_o":k['net_overlay'],
  "xi_star":xi
}])
try: dx = pd.read_csv("corazon/daily_xi.csv"); dx = pd.concat([dx,rowX], ignore_index=True)
except FileNotFoundError: dx = rowX
dx.to_csv("corazon/daily_xi.csv", index=False)

print(f"[{status}] KPIs LIVE {dt} | xi*={xi:.4f}x")
PY
}
# --- LUNES: FREEZE semanal (con resumen humano) ---
runC_shadow_weekly_freeze() {
  set -euo pipefail
  if [[ -z "${FREEZE:-}" ]]; then
    echo "⛔ Define FREEZE primero. Ej: export FREEZE=\"$(date +%F) 00:00\""; return 1
  fi
  local fd="${FREEZE%% *}"
  local rules_snap="configs/heart_rules_${fd}.yaml"

  echo "[1] Snapshot de reglas -> ${rules_snap}"
  cp configs/heart_rules.yaml "$rules_snap" || true
  mkdir -p corazon
  (git rev-parse --short HEAD || echo unknown) > "corazon/RULES_${fd}.txt"

  echo "[2] Pesos FREEZE"
  python scripts/corazon_weights_generator.py \
    --rules "$rules_snap" \
    --ohlc  reports/ohlc_4h/BTC-USD.csv \
    --diamante signals/diamante.csv \
    --perla    signals/perla.csv \
    --out_weights "corazon/weights_${fd}.csv" \
    --out_lq     "corazon/lq_${fd}.csv"

  mkdir -p reports/heart
  cp "corazon/weights_${fd}.csv" reports/heart/w_diamante.csv

  echo "[3] Export baseline por-barra (FREEZE=$FREEZE)"
  export CORAZON_EXPORT_BARS=1
  EXCHANGE="${EXCHANGE:-binanceus}" python swing_4h_forward_diamond.py --skip_yf \
    --symbol BTC-USD --period 730d --horizons 30,60,90 \
    --freeze_end "$FREEZE" \
    --slip 0.0002 --cost 0.0004 \
    --out_csv "reports/diamante_btc_costes_freeze_${fd}.csv"
  local base="reports/diamante_btc_costes_freeze_${fd}_bars.csv"
  [[ -f "$base" ]] || { echo "⛔ No encontré $base"; return 1; }

  echo "[4] Overlay + KPIs (FREEZE)"
  runH_apply_overlay "$base"
  python scripts/report_heart_vs_baseline.py \
    --baseline_csv "$base" \
    --overlay_csv  "reports/heart/diamante_overlay_diamante_btc_costes_freeze_${fd}_bars.csv" \
    --out_md       "reports/heart/summary_diamante_btc_costes_freeze_${fd}_bars.md" \
    --out_csv      "reports/heart/kpis_diamante_btc_costes_freeze_${fd}_bars.csv" \
    --ts_col timestamp

  echo "[5] PASS/FAIL + ξ* + Resumen"
  python - "$fd" <<'PY'
import pandas as pd, sys, math
fd = sys.argv[1]
k = pd.read_csv(f"reports/heart/kpis_diamante_btc_costes_freeze_{fd}_bars.csv").iloc[0]
pf_ok  = (k['pf_overlay'] >= 0.90*k['pf_base']) and (k['pf_overlay'] <= 1.10*k['pf_base'])
mdd_ok = abs(k['mdd_overlay']) <= abs(k['mdd_base'])
vol_ok = k['vol_overlay'] <= k['vol_base']
status = "PASS" if (pf_ok and mdd_ok and vol_ok) else "FAIL"

mdd_ratio = abs(k["mdd_base"]) / max(1e-12, abs(k["mdd_overlay"]))
vol_ratio = k["vol_base"] / max(1e-12, k["vol_overlay"])
xi = min(mdd_ratio, vol_ratio) * 0.85

# Logs semanales
import time
row = pd.DataFrame([{
  "ts": fd, "freeze": fd+" 00:00",
  "pf_b":k['pf_base'],"pf_o":k['pf_overlay'],
  "mdd_b":k['mdd_base'],"mdd_o":k['mdd_overlay'],
  "vol_b":k['vol_base'],"vol_o":k['vol_overlay'],
  "net_b":k['net_base'],"net_o":k['net_overlay'],
  "status":status, "xi_star":xi
}])
for path in ["corazon/weekly_status.csv","corazon/weekly_xi.csv"]:
    try:
        df = pd.read_csv(path); df = pd.concat([df,row], ignore_index=True)
    except FileNotFoundError:
        df = row
    df.to_csv(path, index=False)

# Resumen humano
def pct(a,b):
    return (a/b-1.0)*100.0 if b not in (0,0.0) else float('nan')
wmean=q10=q50=q90=float('nan')
try:
    w = pd.read_csv("reports/heart/w_diamante.csv")
    if "w_diamante" in w.columns:
        s = w["w_diamante"].astype(float)
        wmean=float(s.mean()); q10,q50,q90 = [float(x) for x in s.quantile([.1,.5,.9]).values]
except Exception:
    pass

print(f"[{status}] FREEZE {fd} | xi*={xi:.4f}x")
print(f"[Resumen {fd}] PF {k.pf_overlay:.3f} ({pct(k.pf_overlay,k.pf_base):+.1f}%), "
      f"MDD {k.mdd_overlay:.4%} vs {k.mdd_base:.4%}, "
      f"σ {k.vol_overlay:.6f} vs {k.vol_base:.6f}, "
      f"NET {k.net_overlay:.5f}, w̄={wmean:.2f} (q10={q10:.2f}, q50={q50:.2f}, q90={q90:.2f})")
PY
}

# --- DIARIO: LIVE end-to-end (con resumen humano) ---
runC_shadow_daily() {
  set -euo pipefail
  local dt="${1:-$(date +%F)}"
  local rules_snap="configs/heart_rules_${FREEZE%% *}.yaml"
  [[ -f "$rules_snap" ]] || rules_snap="configs/heart_rules.yaml"

  echo "[1] Export LIVE por-barra"
  export CORAZON_EXPORT_BARS=1
  EXCHANGE="${EXCHANGE:-binanceus}" python swing_4h_forward_diamond.py --skip_yf \
    --symbol BTC-USD --period 730d --horizons 30,60,90 \
    --slip 0.0002 --cost 0.0004 \
    --out_csv "reports/diamante_btc_live_${dt}.csv"

  echo "[2] Pesos LIVE (YAML snapshot si existe)"
  python scripts/corazon_weights_generator.py \
    --rules "$rules_snap" \
    --ohlc  reports/ohlc_4h/BTC-USD.csv \
    --diamante signals/diamante.csv \
    --perla    signals/perla.csv \
    --out_weights "corazon/weights_live_${dt}.csv" \
    --out_lq     "corazon/lq_live_${dt}.csv"
  mkdir -p reports/heart
  cp "corazon/weights_live_${dt}.csv" reports/heart/w_diamante.csv

  echo "[3] Overlay + [4] KPIs LIVE"
  local base="reports/diamante_btc_live_${dt}_bars.csv"
  runH_apply_overlay "$base"
  python scripts/report_heart_vs_baseline.py \
    --baseline_csv "$base" \
    --overlay_csv  "reports/heart/diamante_overlay_diamante_btc_live_${dt}_bars.csv" \
    --out_md  "reports/heart/summary_diamante_btc_live_${dt}_bars.md" \
    --out_csv "reports/heart/kpis_diamante_btc_live_${dt}_bars.csv" \
    --ts_col timestamp

  echo "[5] PASS/FAIL + ξ* + Resumen"
  python - "$dt" <<'PY'
import pandas as pd, sys
dt = sys.argv[1]
k = pd.read_csv(f"reports/heart/kpis_diamante_btc_live_{dt}_bars.csv").iloc[0]
pf_ok  = (k['pf_overlay'] >= 0.90*k['pf_base']) and (k['pf_overlay'] <= 1.10*k['pf_base'])
mdd_ok = abs(k['mdd_overlay']) <= abs(k['mdd_base'])
vol_ok = k['vol_overlay'] <= k['vol_base']
status = "PASS" if (pf_ok and mdd_ok and vol_ok) else "FAIL"

mdd_ratio = abs(k["mdd_base"]) / max(1e-12, abs(k["mdd_overlay"]))
vol_ratio = k["vol_base"] / max(1e-12, k["vol_overlay"])
xi = min(mdd_ratio, vol_ratio) * 0.85

# Logs diarios
rowS = pd.DataFrame([{"ts":dt,
  "pf_b":k['pf_base'],"pf_o":k['pf_overlay'],
  "mdd_b":k['mdd_base'],"mdd_o":k['mdd_overlay'],
  "vol_b":k['vol_base'],"vol_o":k['vol_overlay'],
  "net_b":k['net_base'],"net_o":k['net_overlay'],
  "status":status
}])
try: ds = pd.read_csv("corazon/daily_status.csv"); ds = pd.concat([ds,rowS], ignore_index=True)
except FileNotFoundError: ds = rowS
ds.to_csv("corazon/daily_status.csv", index=False)

rowX = pd.DataFrame([{"ts":dt,
  "pf_b":k['pf_base'],"pf_o":k['pf_overlay'],
  "mdd_b":k['mdd_base'],"mdd_o":k['mdd_overlay'],
  "vol_b":k['vol_base'],"vol_o":k['vol_overlay'],
  "net_b":k['net_base'],"net_o":k['net_overlay'],
  "xi_star":xi
}])
try: dx = pd.read_csv("corazon/daily_xi.csv"); dx = pd.concat([dx,rowX], ignore_index=True)
except FileNotFoundError: dx = rowX
dx.to_csv("corazon/daily_xi.csv", index=False)

# Resumen humano
def pct(a,b):
    return (a/b-1.0)*100.0 if b not in (0,0.0) else float('nan')
wmean=q10=q50=q90=float('nan')
try:
    w = pd.read_csv("reports/heart/w_diamante.csv")
    if "w_diamante" in w.columns:
        s = w["w_diamante"].astype(float)
        wmean=float(s.mean()); q10,q50,q90 = [float(x) for x in s.quantile([.1,.5,.9]).values]
except Exception:
    pass

print(f"[{status}] KPIs LIVE {dt} | xi*={xi:.4f}x")
print(f"[Resumen {dt}] PF {k.pf_overlay:.3f} ({pct(k.pf_overlay,k.pf_base):+.1f}%), "
      f"MDD {k.mdd_overlay:.4%} vs {k.mdd_base:.4%}, "
      f"σ {k.vol_overlay:.6f} vs {k.vol_base:.6f}, "
      f"NET {k.net_overlay:.5f}, w̄={wmean:.2f} (q10={q10:.2f}, q50={q50:.2f}, q90={q90:.2f})")
PY
}# -- Corazón: FREEZE semanal con log de KPIs + ξ* --
runC_log_weekly_freeze() {
  local D="${1:-${FREEZE%% *}}"
  if [[ -z "$D" ]]; then echo "[ERR] FREEZE vacío. Uso: runC_log_weekly_freeze YYYY-MM-DD"; return 1; fi

  mkdir -p reports/heart corazon

  local BASE="reports/diamante_btc_costes_freeze_${D}_bars.csv"
  local OVER="reports/heart/diamante_overlay_diamante_btc_costes_freeze_${D}_bars.csv"
  local KPI="reports/heart/kpis_diamante_btc_costes_freeze_${D}_bars.csv"
  local MD="reports/heart/summary_diamante_btc_costes_freeze_${D}_bars.md"

  echo "[1] Export baseline FREEZE → $BASE"
  export CORAZON_EXPORT_BARS=1
  EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf \
    --symbol BTC-USD --period 730d --horizons 30,60,90 \
    --freeze_end "$D 00:00" \
    --slip 0.0002 --cost 0.0004 \
    --out_csv "reports/diamante_btc_costes_freeze_${D}.csv" || return $?

  [[ -f "$BASE" ]] || { echo "[alias] usando week1_bars como FREEZE"; cp reports/diamante_btc_costes_week1_bars.csv "$BASE"; }

  echo "[2] Pesos (YAML snapshot si existe)"
  local RULES="configs/heart_rules_${D}.yaml"
  [[ -f "$RULES" ]] || RULES="configs/heart_rules.yaml"

  python scripts/corazon_weights_generator.py \
    --rules  "$RULES" \
    --ohlc   reports/ohlc_4h/BTC-USD.csv \
    --diamante signals/diamante.csv \
    --perla    signals/perla.csv \
    --out_weights "corazon/weights_${D}.csv" \
    --out_lq     "corazon/lq_${D}.csv" || return $?

  cp "corazon/weights_${D}.csv" reports/heart/w_diamante.csv

  echo "[3] Overlay + [4] KPIs"
  python scripts/apply_heart_overlay.py "$BASE" \
    --weights_csv reports/heart/w_diamante.csv \
    --out_csv "$OVER" || return $?

  python scripts/report_heart_vs_baseline.py \
    --baseline_csv "$BASE" \
    --overlay_csv  "$OVER" \
    --out_md  "$MD" \
    --out_csv "$KPI" \
    --ts_col timestamp || return $?

  echo "[5] ξ* + PASS/FAIL → corazon/daily_xi.csv"
  D="$D" python - <<'PY'
import os, pandas as pd
D = os.environ["D"]
k = pd.read_csv(f"reports/heart/kpis_diamante_btc_costes_freeze_{D}_bars.csv").iloc[0]
mdd_ratio = abs(float(k["mdd_base"])) / max(1e-12, abs(float(k["mdd_overlay"])))
vol_ratio = float(k["vol_base"]) / max(1e-12, float(k["vol_overlay"]))
xi = min(mdd_ratio, vol_ratio) * 0.85
row = pd.DataFrame([{
 "ts": D, "mdd_ratio": mdd_ratio, "vol_ratio": vol_ratio, "xi_star": xi,
 "pf_base": k["pf_base"], "pf_overlay": k["pf_overlay"],
 "mdd_base": k["mdd_base"], "mdd_overlay": k["mdd_overlay"],
 "vol_base": k["vol_base"], "vol_overlay": k["vol_overlay"],
 "net_base": k["net_base"], "net_overlay": k["net_overlay"],
}])
try:
  dx = pd.read_csv("corazon/daily_xi.csv")
  dx = dx[dx["ts"] != D]
  dx = pd.concat([dx, row], ignore_index=True)
except FileNotFoundError:
  dx = row
dx.to_csv("corazon/daily_xi.csv", index=False)
print(f"[PASS] KPIs FREEZE {D} | ξ*={xi:.4f}x")
PY

  echo "[OK] Artefactos:"
  ls -lh "$OVER" "$KPI" "$MD" 2>/dev/null || true
}
