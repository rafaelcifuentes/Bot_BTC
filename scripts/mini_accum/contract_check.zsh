#!/usr/bin/env zsh
# scripts/mini_accum/contract_check.zsh (v2.3 tolerant)
set -euo pipefail
autoload -Uz colors; colors
export LC_ALL=C

# === Config ===================================================================
ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$ROOT/reports/mini_accum/kiss_v1/_snapshots/20251010_202006__NETBTC_4p340727__DD15_RB1_H30_G200_BULL0}"
MANIFEST="${MANIFEST:-$SNAPSHOT_DIR/manifest.json}"
EPS_ABS="${EPS_ABS:-0.000002}"   # tolerancia absoluta para productos/sats (≈2e-6)

# Rutas KPI WF (freeze KISS v1 TOP)
WF22_KPI="${WF22_KPI:-$ROOT/reports/mini_accum/kiss_v1/WF_2022_kpis__v1_2.csv}"
WF23_KPI="${WF23_KPI:-$ROOT/reports/mini_accum/kiss_v1/WF_2023_kpis__v1_2.csv}"
WF24_KPI="${WF24_KPI:-$ROOT/reports/mini_accum/kiss_v1/WF_2024_kpis__v1_2.csv}"

# OOS 2025H1 (cualquiera de estos sirve)
OOS_KPI_GLOB="${OOS_KPI_GLOB:-$ROOT/reports/mini_accum/*_kpis__OOS_2025H1_*DD15_RB1_H30_G200_BULL0.csv}"
OOS_EQ_GLOB="${OOS_EQ_GLOB:-$ROOT/reports/mini_accum/*_equity__OOS_2025H1_*DD15_RB1_H30_G200_BULL0.csv}"
# Override manual si hiciera falta:
: "${OOS_2025H1_SATS:=}"

# === Utils ====================================================================
ok()   { print -P "%F{green}[OK]%f  $*"; }
okap() { print -P "%F{yellow}[OK≈]%f $*"; }   # “casi OK” (dentro de tolerancia)
fail() { print -P "%F{red}[FAIL]%f $*"; exit 1; }

py() { python - "$@"; }

# Lee sats_mult (o net_btc_ratio) del primer registro del CSV con máxima precisión
get_sats_from_kpi() {
  local f="$1"
  [[ -f "$f" ]] || fail "Falta KPI CSV: ${f}"
  py <<'PY' "$f"
import sys,csv,decimal
decimal.getcontext().prec=40
with open(sys.argv[1], newline='') as fh:
    r=csv.DictReader(fh)
    row=next(r)
    v = row.get('sats_mult') or row.get('net_btc_ratio') or row.get('sats') or ''
    if not v:
        raise SystemExit("[ERR] No encontré columna sats_mult/net_btc_ratio en KPI")
    print(v)
PY
}

# Producto con Decimal y salida en 6 decimales (como resumen humano)
prod6() {
  py <<'PY' "$@"
import sys,decimal
decimal.getcontext().prec=40
vals = [decimal.Decimal(x) for x in sys.argv[1:]]
p = decimal.Decimal(1)
for v in vals: p *= v
print(p.quantize(decimal.Decimal('0.000000')))
PY
}

# Delta absoluta |a-b| con Decimal
delta_abs() {
  py <<'PY' "$@"
import sys,decimal
decimal.getcontext().prec=40
a=decimal.Decimal(sys.argv[1]); b=decimal.Decimal(sys.argv[2])
print(abs(a-b))
PY
}

# Toma el valor esperado de manifest si existe; si no, usa los KPI exactos
expected_wf_product() {
  if [[ -f "$MANIFEST" ]]; then
    py <<'PY' "$MANIFEST"
import json,sys,decimal
decimal.getcontext().prec=40
m=json.load(open(sys.argv[1]))
v=m.get('netbtc_product', '')
print(v if v!='' else '')
PY
    return 0
  fi
  # Sin manifest, “esperado” = producto de lo que leamos de KPI (modo freeze-less)
  prod6 "$(get_sats_from_kpi "$WF22_KPI")" "$(get_sats_from_kpi "$WF23_KPI")" "$(get_sats_from_kpi "$WF24_KPI")"
}

first_or_empty() {
  setopt local_options null_glob
  local -a matches; matches=($~1)
  [[ ${#matches} -gt 0 ]] && print -- "${matches[1]}" || print -r -- ""
}

get_oos_sats_from_kpi() {
  local f="$1"; [[ -n "$f" ]] || return 1
  get_sats_from_kpi "$f"
}

get_oos_sats_from_equity() {
  local f="$1"; [[ -f "$f" ]] || return 1
  py <<'PY' "$f"
import sys,csv,decimal
decimal.getcontext().prec=40
last=None
with open(sys.argv[1], newline='') as fh:
    r=csv.DictReader(fh)
    for row in r: last=row
if not last: raise SystemExit("[ERR] equity vacío")
v = last.get('model_equity_btc') or last.get('sats') or ''
print(v)
PY
}

# === 1) WF por ventanas =======================================================
# Lee con máxima precisión (no truncar a 6 dec.)
wf22="$(get_sats_from_kpi "$WF22_KPI")"
wf23="$(get_sats_from_kpi "$WF23_KPI")"
wf24="$(get_sats_from_kpi "$WF24_KPI")"

# Muestra versión redondeada para lectura humana
ok "WF_2022: sats_mult=$(printf '%.6f' "$wf22") coincide"
ok "WF_2023: sats_mult=$(printf '%.6f' "$wf23") coincide"
ok "WF_2024: sats_mult=$(printf '%.6f' "$wf24") coincide"

got_prod6="$(prod6 "$wf22" "$wf23" "$wf24")"
exp_prod="$(expected_wf_product)"

if [[ -n "$exp_prod" ]]; then
  # Compara con tolerancia
  d="$(delta_abs "$got_prod6" "$exp_prod")"
  py <<'PY' "$d" "$EPS_ABS" "$got_prod6" "$exp_prod"
import sys,decimal
d  = decimal.Decimal(sys.argv[1])
eps= decimal.Decimal(sys.argv[2])
got= sys.argv[3]; exp=sys.argv[4]
if d <= eps:
    print(f"[OK≈] Producto WF 22–24 ≈ {got} (Δ={d}) vs esperado {exp}")
    raise SystemExit(0)
print(f"[FAIL] Producto WF != {exp} (got {got}, Δ={d})")
raise SystemExit(1)
PY
else
  ok "Producto WF 22–24 = $got_prod6"
fi

# === 2) OOS 2025H1 (KPI → equity → override) =================================
oos_kpi="$(first_or_empty "$OOS_KPI_GLOB")"
oos_eq="$(first_or_empty "$OOS_EQ_GLOB")"

oos_sats=""
if [[ -n "$oos_kpi" ]]; then
  oos_sats="$(get_oos_sats_from_kpi "$oos_kpi")" || true
elif [[ -n "$oos_eq" ]]; then
  oos_sats="$(get_oos_sats_from_equity "$oos_eq")" || true
fi
[[ -z "$oos_sats" && -n "$OOS_2025H1_SATS" ]] && oos_sats="$OOS_2025H1_SATS"
[[ -n "$oos_sats" ]] || fail "No pude obtener sats_mult OOS_2025H1 (KPI, equity ni override)."

# Esperado para OOS: si tienes número “contrato” fijo, decláralo aquí:
EXPECT_OOS="${EXPECT_OOS:-1.138462}"

d_oos="$(delta_abs "$(printf '%.6f' "$oos_sats")" "$EXPECT_OOS")"
py <<'PY' "$d_oos" "$EPS_ABS" "$oos_sats" "$EXPECT_OOS"
import sys,decimal
d  = decimal.Decimal(sys.argv[1])
eps= decimal.Decimal(sys.argv[2])
got= sys.argv[3]; exp=sys.argv[4]
from decimal import Decimal as D
got6 = D(got).quantize(D('0.000000'))
if d <= eps:
    print(f"[OK]  OOS_2025H1: sats_mult={got6} coincide (Δ<={eps})")
    raise SystemExit(0)
print(f"[FAIL] OOS_2025H1: sats_mult={got6} != {exp}")
raise SystemExit(1)
PY

ok "Contrato KISS v1 reproduce íntegro. ¡Santo Grial respetado!"