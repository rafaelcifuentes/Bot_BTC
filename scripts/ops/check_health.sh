#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

# --- Config por defecto (overridable via env) ---
NETBTC_MIN=${NETBTC_MIN:-1.0}
E1_M_MAX=${E1_M_MAX:-0.12}
V1_M_MAX=${V1_M_MAX:-1.0}
FPY_CAP=${FPY_CAP:-12}

# --- Tag activo y freeze ---
TAG="$(tr -d '\r\n' < deploy/ACTIVE.tag 2>/dev/null || true)"
if [ -z "${TAG:-}" ]; then
  echo "[ERR] deploy/ACTIVE.tag vacío o ausente"
  exit 2
fi

FREEZE="reports/mini_accum/_freezes/${TAG}.freeze.yaml"
if [ ! -f "$FREEZE" ]; then
  echo "[ERR] No encontré freeze para TAG=$TAG en reports/mini_accum/_freezes"
  exit 2
fi

# --- Extraer KPIs (coerción numérica) ---
num_yq () { awk -F: -v key="$1" 'index($1,key){gsub(/[[:space:]]/,"",$2); print ($2+0)}' "$FREEZE"; }
SATS="$(num_yq sats_mult)"
MDD="$(num_yq mdd_vs_hodl)"
FLIPS="$(num_yq flips)"

# --- Tipo de preset: V1 o E1 ---
KIND=$([[ "$TAG" == *"E1"* ]] && echo "E1" || echo "V1")
M_MAX=$([ "$KIND" = "E1" ] && echo "$E1_M_MAX" || echo "$V1_M_MAX")

# --- Comparadores numéricos puros ---
lt(){ awk -v A="$1" -v B="$2" 'BEGIN{exit ((A+0)<(B+0))?0:1}'; }
gt(){ awk -v A="$1" -v B="$2" 'BEGIN{exit ((A+0)>(B+0))?0:1}'; }

# --- Checks ---
FAIL=0
if lt "$SATS" "$NETBTC_MIN"; then
  echo "[ALERT] NetBTC=$SATS < $NETBTC_MIN"
  FAIL=1
fi
if gt "$MDD" "$M_MAX"; then
  echo "[ALERT] MDD_vs_HODL=$MDD > $M_MAX ($KIND)"
  FAIL=1
fi
if gt "$FLIPS" "$FPY_CAP"; then
  echo "[ALERT] flips=$FLIPS > cap=$FPY_CAP"
  FAIL=1
fi

echo "[HEALTH] TAG=$TAG kind=$KIND  NetBTC=$SATS  MDD=$MDD  flips=$FLIPS  caps: NETBTC_MIN=$NETBTC_MIN M_MAX=$M_MAX FPY_CAP=$FPY_CAP"
exit "$FAIL"
