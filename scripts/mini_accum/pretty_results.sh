#!/usr/bin/env bash
set -Eeuo pipefail

# Uso:
#   pretty_results.sh [REGEX_SUFFIX] [WINDOW]
# Ejemplos:
#   pretty_results.sh 'v3p3N2g-F2-(H1_FZ|Q4_E3)-' 2024H1
#   pretty_results.sh 'v3p3N2g-F2-(H1_FZ|Q4_E3)-' 2023Q4
#
# Notas:
# - Calcula Δbps vs BASE usando netBTC (columna robusta).
# - Detecta dinámicamente las columnas finales: ... flips  fpy  PASS  [FAIL_REASON]
# - Si mini-accum-dictamen falla, usa el último TSV de reports/mini_accum/.
#
# Variables:
#   BASE     (default 1.044263)   → baseline netBTC para calcular bps
#   OUT_TSV  (default /tmp/mini_accum_dictamen.tsv)

PATTERN="${1:-v3p3N2g-}"       # regex de sufijos a incluir
WINDOW="${2:-2024H1}"          # 2024H1 | 2023Q4 | (si no coincide, no filtra por ventana)
BASE="${BASE:-1.044263}"       # netBTC baseline para Δ bps
OUT_TSV="${OUT_TSV:-/tmp/mini_accum_dictamen.tsv}"

# 1) Intentar generar un TSV fresco
if command -v mini-accum-dictamen >/dev/null 2>&1; then
  if ! mini-accum-dictamen --reports-dir reports/mini_accum --out "$OUT_TSV" --format tsv --quiet; then
    echo "⚠️ mini-accum-dictamen falló, intentaré usar el último TSV existente..." >&2
  fi
else
  echo "⚠️ mini-accum-dictamen no está en PATH; usaré el último TSV existente..." >&2
fi

# 2) Elegir fuente: archivo fresco o el último en reports/mini_accum
SRC="$OUT_TSV"
if [[ ! -s "$SRC" ]]; then
  SRC="$(ls -t reports/mini_accum/dictamen_*.tsv 2>/dev/null | head -1 || true)"
fi
if [[ -z "$SRC" || ! -s "$SRC" ]]; then
  echo "❌ No encontré dictamen TSV legible. (intenté $OUT_TSV y reports/mini_accum/dictamen_*.tsv)" >&2
  exit 1
fi

# 3) Render bonito y robusto a cambios de columnas internas
(
  printf "suffix\twindow\tnetBTC\tdbps\tmdd_vs_HODL\tflips\tfpy\tPASS\tFAIL_REASON\n"
  awk -F'\t' -v pat="$PATTERN" -v win="$WINDOW" -v base="$BASE" '
    function last_pass_idx(nf,i){ for(i=nf;i>=1;i--){ if($i=="True"||$i=="False") return i } return 0 }
    BEGIN{ OFS="\t" }
    {
      # Filtra por patrón de sufijo
      if ($1 !~ pat) next
      # Filtra por ventana si WINDOW tiene formato típico (2024H1 / 2023Q4)
      if (win ~ /^[0-9]{4}(H1|H2|Q[1-4])$/ && $2 != win) next

      # Detecta PASS y deduce campos relativos al final
      pi = last_pass_idx(NF); if(pi==0) next
      pass=$pi
      fail=(pi+1<=NF)? $(pi+1) : ""
      fpy=$(pi-1)
      flips=$(pi-2)
      mdd=$(pi-3)

      # netBTC suele estar en la 3ª columna (estable)
      net=$3
      dbps=(net-base)*10000

      printf "%s\t%s\t%.6f\t%+.1f\t%.3f\t%.1f\t%.2f\t%s\t%s\n", $1,$2,net,dbps,mdd,flips,fpy,pass,fail
    }' "$SRC"
) | sort -t $'\t' -k3,3nr | column -t -s $'\t'
