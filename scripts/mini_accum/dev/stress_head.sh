#!/usr/bin/env bash
set -euo pipefail
f="reports/mini_accum/STRESS_COSTS.txt"
[ -s "$f" ] || { echo "[ERR] No hay datos en $f. Ejecuta: make stress-costs"; exit 1; }

awk -F'|' '
BEGIN{
  e1_s=(ENVIRON["E1_S_MIN"]?ENVIRON["E1_S_MIN"]:2.9)+0
  e1_m=(ENVIRON["E1_M_MAX"]?ENVIRON["E1_M_MAX"]:0.12)+0
  v1_s=(ENVIRON["V1_S_MIN"]?ENVIRON["V1_S_MIN"]:1.0)+0
  v1_m=(ENVIRON["V1_M_MAX"]?ENVIRON["V1_M_MAX"]:1.0)+0
  hdr=""; out=""
}
# capturamos solo el ÚLTIMO bloque
/^== STRESS START/ {hdr=$0; out=""; next}
(/^TAG|^----|^==/ || NF<8) {next}

{
  for(i=1;i<=NF;i++){gsub(/^[ \t]+|[ \t]+$/,"",$i)}
  pol=$2
  split($4,fs,"/"); fee=fs[1]+0; slip=fs[2]+0
  s0=$5+0; mdd=$7+0

  s_min=(pol=="E1")?e1_s:((pol=="V1")?v1_s:1.0)
  m_max=(pol=="E1")?e1_m:((pol=="V1")?v1_m:1.0)

  # S_req escala con (fee+slip)/3
  s_req=s_min*(fee+slip)/3
  s_head=s0 - s_req
  m_head=m_max - mdd

  miss_s=(s_head < -1e-9)
  miss_m=(m_head < -1e-9)

  status=(miss_s||miss_m)?"MISS":"OK"
  if(miss_s && miss_m) status="MISS[S,M]"
  else if(miss_s)      status="MISS[S]"
  else if(miss_m)      status="MISS[M]"

  line=sprintf("%-16s | %2s | %5s | %4s | %10.6f | %10.6f | %9.6f | %8.6f | %+.6f | %+.6f | %s",
               $1, pol, $3, $4, s0, s_req, mdd, m_max, s_head, m_head, status)
  out=out line "\n"
}

END{
  if(hdr!=""){
    printf "== THR E1_S_MIN=%.6f E1_M_MAX=%.6f V1_S_MIN=%.6f V1_M_MAX=%.6f ==\n", e1_s, e1_m, v1_s, v1_m
    print hdr
    print "TAG              | PL | FLIPS | FEE/SLIP | S0        | S_req     | MDD       | M_max    | headS     | headM     | RES"
    printf "%s", out
  } else {
    print "[ERR] No se encontró bloque == STRESS START"
  }
}' "$f" | column -t
