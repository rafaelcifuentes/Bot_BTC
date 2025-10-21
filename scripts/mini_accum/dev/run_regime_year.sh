#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-}"
if [[ -z "${TAG}" ]]; then
  echo "Uso: $0 {2022|2023|2024|2025H1}" >&2
  exit 1
fi

# Ventana
case "$TAG" in
  2022)   Y=2022; START=2022-01-01; END=2022-12-31 ;;
  2023)   Y=2023; START=2023-01-01; END=2023-12-31 ;;
  2024)   Y=2024; START=2024-01-01; END=2024-12-31 ;;
  2025H1) Y=2025; START=2025-01-01; END=2025-06-30 ;;
  *) echo "TAG inválido: $TAG" >&2; exit 1;;
esac

# Halvings fijos
HALVINGS=(2012 2016 2020 2024)
last=${HALVINGS[0]}
for h in "${HALVINGS[@]}"; do (( h<=Y )) && last=$h; done
years_since=$((Y-last))

# Presets base
KISS_V1_CFG="configs/mini_accum/presets/CORE_2025.yaml"
E1_CFG="configs/mini_accum/presets/E1_Y2.yaml"

BASE_CFG="$KISS_V1_CFG"
(( years_since == 2 )) && BASE_CFG="$E1_CFG"
BASE_CFG="${MA_FORCE_CFG:-$BASE_CFG}"

# Rutas ABSOLUTAS de datos (NO usamos symlinks)
ROOT="$(pwd)"
D1_SRC="$ROOT/data/wf_yearly/1d/BTC-USD_1d_WF_${Y}.csv"
H4_SRC="$ROOT/data/wf_canonical/BTC-USD_4h.csv"

[[ -s "$D1_SRC" ]] || { echo "[ERR] Falta 1D: $D1_SRC" >&2; exit 2; }
[[ -s "$H4_SRC" ]] || { echo "[ERR] Falta 4H canónico: $H4_SRC" >&2; exit 2; }

# Genera YAML temporal con overrides de data.*
TMP_CFG="$(mktemp -t regime_${TAG}_XXXX.yaml)"
if command -v yq >/dev/null 2>&1; then
  yq e ".data.ohlc_d1_csv = \"$D1_SRC\" | .data.ohlc_4h_csv = \"$H4_SRC\"" "$BASE_CFG" > "$TMP_CFG"
else
  # fallback en Python si no hay yq
  python - "$BASE_CFG" "$TMP_CFG" "$D1_SRC" "$H4_SRC" <<'PY'
import sys, yaml
src, dst, d1, h4 = sys.argv[1:]
cfg = yaml.safe_load(open(src))
cfg.setdefault("data", {})
cfg["data"]["ohlc_d1_csv"] = d1
cfg["data"]["ohlc_4h_csv"] = h4
yaml.safe_dump(cfg, open(dst,"w"), sort_keys=False)
print("[PY] wrote", dst)
PY
fi

echo "[REGIME] TAG=$TAG Y=$Y (+$years_since post-halving)"
echo "        1D -> $D1_SRC"
echo "        4H -> $H4_SRC"
echo "        CFG= $BASE_CFG -> $TMP_CFG"

# Sanity de cabeceras
head -1 "$D1_SRC" | grep -q 'ts,open,high,low,close,volume' || { echo "[ERR] Cabecera inválida en 1D"; exit 3; }
head -1 "$H4_SRC" | grep -q 'ts,open,high,low,close,volume' || { echo "[ERR] Cabecera inválida en 4H"; exit 3; }

python -m mini_accum.cli \
  --config "$TMP_CFG" \
  --start  "$START" \
  --end    "$END" \
  --suffix "OOS_${TAG}_REGIME"

echo "[DONE] $TAG"