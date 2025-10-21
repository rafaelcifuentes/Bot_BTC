#!/usr/bin/env bash
set -euo pipefail
f="reports/mini_accum/STRESS_COSTS.txt"
[ -s "$f" ] || { echo "[ERR] No hay datos en $f"; exit 1; }

# Extrae solo el último bloque
last_block() {
  awk 'BEGIN{start=0} /^== STRESS START/{start=NR} {L[NR]=$0} END{for(i=start;i<=NR;i++) print L[i]}' "$f"
}

# Totales PASS/FAIL del último bloque
last_block | awk -F'|' '
  BEGIN{pass=0; fail=0}
  /^TAG|^----|^==/ {next}
  NF<8 {next}
  { for(i=1;i<=NF;i++){gsub(/^[ \t]+|[ \t]+$/,"",$i)} }
  ($8=="PASS"){pass++}
  ($8=="FAIL"){fail++}
  END{printf "STRESS PASS=%d  FAIL=%d\n", pass, fail}
'

# Min/Max por política en el último bloque (usa S_ADJ = col 7)
last_block | awk -F'|' '
  /^TAG|^----|^==/ {next} NF<8 {next}
  {
    for(i=1;i<=NF;i++){gsub(/^[ \t]+|[ \t]+$/,"",$i)}
    pol=$2; s_adj=$7+0
    if(!(pol in min) || s_adj<min[pol]){min[pol]=s_adj; minl[pol]=$0}
    if(!(pol in max) || s_adj>max[pol]){max[pol]=s_adj; maxl[pol]=$0}
  }
  END{
    for(p in min){
      print "--", p, "min:", minl[p]
      print "--", p, "max:", maxl[p]
    }
  }
'
