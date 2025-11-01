#!/usr/bin/env zsh
set -euo pipefail
ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
MANIFEST_E1Y2="${MANIFEST_E1Y2:-$ROOT/reports/mini_accum/kiss_v1/_snapshots/E1Y2_2022/manifest.json}"
E1Y2_SATS="${E1Y2_SATS:-2.9624647328602833}"
EPS_ABS="${EPS_ABS:-0.000002}"

if ! command -v jq >/dev/null 2>&1; then
  echo "[FAIL] 'jq' no encontrado (brew install jq)"; exit 1
fi
test -s "$MANIFEST_E1Y2" || { echo "[FAIL] no existe manifest E1_Y2: $MANIFEST_E1Y2"; exit 1; }

# 1) Intenta dentro de .windows (WF_2022 u OOS_2022).
sat="$(jq -r '((.windows? // []) | map(select(.window=="WF_2022" or .window=="OOS_2022") | .sats_mult) | first) // empty' "$MANIFEST_E1Y2")"

# 2) Si no, busca un campo plano.
if [[ -z "${sat}" || "${sat}" == "null" ]]; then
  sat="$(jq -r '.sats_mult? // .metrics?.sats_mult? // empty' "$MANIFEST_E1Y2")"
fi

# 3) Si aún no, parsea desde el nombre del snapshot (...__sats_2p962464...).
if [[ -z "${sat}" || "${sat}" == "null" ]]; then
  SNAP_DIR="$(python - "$MANIFEST_E1Y2" <<'PY'
import os,sys
p=sys.argv[1]
print(os.path.realpath(os.path.dirname(p)))
PY
)"
  base="$(basename "$SNAP_DIR")"
  # Convierte 2p962464... -> 2.962464...
  parsed="$(printf "%s\n" "$base" | sed -nE 's/.*sats_([0-9]+)p([0-9]+).*/\1.\2/p')"
  if [[ -n "$parsed" ]]; then
    sat="$parsed"
  fi
fi

if [[ -z "${sat}" || "${sat}" == "null" ]]; then
  echo "[FAIL] No pude extraer sats_mult de $MANIFEST_E1Y2 (ni de nombre de snapshot)"; exit 2
fi

python - "$E1Y2_SATS" "$sat" "$EPS_ABS" <<'PY'
import sys
exp=float(sys.argv[1]); got=float(sys.argv[2]); eps=float(sys.argv[3])
d=abs(exp-got)
if d<=eps:
  print(f"[OK]  E1_Y2 WF/OOS 2022: sats_mult={got} coincide (Δ={d:.9g} ≤ {eps})")
  sys.exit(0)
else:
  print(f"[NO OK] E1_Y2 WF/OOS 2022: esperado={exp} got={got} (Δ={d:.9g} > {eps})")
  sys.exit(2)
PY

echo "[OK]  Contrato E1_Y2 reproducible."
