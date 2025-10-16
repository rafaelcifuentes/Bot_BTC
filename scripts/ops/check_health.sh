#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)"
cd "$REPO_ROOT"

TAG="$(tr -d '\r' < deploy/ACTIVE.tag | head -n1 | xargs || true)"
[ -n "${TAG:-}" ] || { echo "[ERR] ACTIVE.tag vacío o ilegible"; exit 2; }

: "${NETBTC_MIN:=1.0}"
: "${V1_M_MAX:=1.0}"
: "${E1_M_MAX:=0.12}"
: "${FPY_CAP:=12}"

resolve_freeze_and_kind() {
  case "$1" in
    PROD_E1_Y2_2022)     echo "reports/mini_accum/_freezes/E1_Y2_2022.freeze.txt E1";;
    PROD_KISSv1_2023)    echo "reports/mini_accum/_freezes/V1TOP_2023.freeze.txt V1";;
    PROD_KISSv1_2024)    echo "reports/mini_accum/_freezes/V1TOP_2024.freeze.txt V1";;
    PROD_KISSv1_2025H1)  echo "reports/mini_accum/_freezes/V1TOP_2025H1.freeze.txt V1";;
    *)                   return 1;;
  esac
}

read -r FRZ KIND <<<"$(resolve_freeze_and_kind "$TAG")" || { 
  echo "[ERR] TAG=$TAG sin mapping conocido"; exit 2; }

FRZ_ABS="$REPO_ROOT/$FRZ"
[ -s "$FRZ_ABS" ] || { echo "[ERR] No encontré freeze para TAG=$TAG en $FRZ_ABS"; exit 2; }

# --- Extracción robusta: primero YAML, luego fallback clave=valor ---
yaml_get_num() { grep -E "^[[:space:]]*$1:[[:space:]]*[0-9.]+$" "$FRZ_ABS" | sed -E "s/.*$1:[[:space:]]*([0-9.]+)/\1/" | head -n1; }
kv_get_num()   { awk -F= -v k="$1" '$1~k{gsub(/[[:space:]]/,"",$2);print $2;exit}' "$FRZ_ABS"; }

SATS="$(yaml_get_num 'sats_mult' || true)";     [ -n "$SATS" ] || SATS="$(kv_get_num 'NetBTC' || true)"
MDD="$(yaml_get_num 'mdd_vs_hodl' || true)";    [ -n "$MDD"  ] || MDD="$(kv_get_num 'MDD_vs_HODL' || true)"
FLIPS="$(yaml_get_num 'flips' || true)";        [ -n "$FLIPS" ] || FLIPS="$(kv_get_num 'flips' || true)"

SATS="${SATS:-0}"; MDD="${MDD:-0}"; FLIPS="${FLIPS:-0}"

M_MAX="$V1_M_MAX"; [ "$KIND" = "E1" ] && M_MAX="$E1_M_MAX"

FAIL=0
cmp_float(){ awk "BEGIN{exit !($1)}"; }

cmp_float "$SATS >= $NETBTC_MIN" || { echo "[ALERT] NetBTC=$SATS < $NETBTC_MIN"; FAIL=1; }
cmp_float "$MDD <= $M_MAX"       || { echo "[ALERT] MDD_vs_HODL=$MDD > $M_MAX ($KIND)"; FAIL=1; }
cmp_float "$FLIPS <= $FPY_CAP"   || { echo "[ALERT] flips=$FLIPS > cap=$FPY_CAP"; FAIL=1; }

printf "[HEALTH] TAG=%s kind=%s  NetBTC=%s  MDD=%s  flips=%s  caps: NETBTC_MIN=%s M_MAX=%s FPY_CAP=%s\n" \
  "$TAG" "$KIND" "$SATS" "$MDD" "$FLIPS" "$NETBTC_MIN" "$M_MAX" "$FPY_CAP"

exit "$FAIL"
