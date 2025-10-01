#!/usr/bin/env bash
# Barrida PT sobre ventanas OOS (walk-forward)
# Params: DD ∈ {14,15,16}, RB ∈ {1,2}, H ∈ {30,31,32}
# Gate: SMA200 (modo sell), BULL0

# --- Guard para evitar "source" accidental que mate tu sesión con set -e ---
if [[ "${BASH_SOURCE[0]-$0}" != "$0" ]]; then
  echo "❌ No ejecutes este script con 'source'. Usa: bash ${BASH_SOURCE[0]-wf_sweep.sh}"
  return 1 2>/dev/null || exit 1
fi

set -u -o pipefail

CONFIG=${CONFIG:-configs/mini_accum/kiss_v1.yaml}
CSV_PATH=${1:-configs/mini_accum/windows_walkforward.csv}

# Param grid
PARAMS=()
for dd in 14 15 16; do
  for rb in 1 2; do
    for h in 30 31 32; do
      PARAMS+=("DD${dd}_RB${rb}_H${h}")
    done
  done
done

# Check insumos
[[ -f "$CONFIG" ]]  || { echo "❌ Falta config: $CONFIG"; exit 2; }
[[ -f "$CSV_PATH" ]]|| { echo "❌ Falta CSV:   $CSV_PATH"; exit 2; }

# Barrida por ventanas
while IFS=, read -r name tr_start tr_end te_start te_end; do
  [[ -z "${name:-}" || "$name" == name* ]] && continue

  echo "────────────────────────────────────────────────────────"
  echo "Ventana: $name  (TEST: $te_start → $te_end)"
  for p in "${PARAMS[@]}"; do
    dd="${p%%_*}"; dd="${dd#DD}"
    rest="${p#*_}"; rb="${rest%%_*}"; rb="${rb#RB}"
    h="${p##*_}";   h="${h#H}"

    suffix="WF_${name}_PT_G200_DD${dd}_RB${rb}_H${h}_BULL0"
    echo "→ Run: DD=$dd  RB=$rb  H=$h  | suffix=$suffix"

    # IMPORTANTE: usar flags que sí existen en kiss_v1.py
    if ! python scripts/mini_accum/kiss_v1.py \
      --config "$CONFIG" \
      --mode pt --gate_sma 200 --gate_mode sell \
      --dd_pct "$dd" --rb_pct "$rb" --dd_hard_pct "$h" \
      --start "$te_start" --end "$te_end" \
      --suffix "$suffix"
    then
      echo "⚠️  Falló: $suffix  (exit=$?) — sigo con el siguiente"
      continue
    fi
  done
done < "$CSV_PATH"

echo "✅ Barrida terminada."