#!/usr/bin/env bash
set -Eeuo pipefail

is_mode() {
  case "${1:-}" in
    all|counts|q4pass|q4fail|h1pass|h1fail|which|counts-by-file) return 0;;
    *) return 1;;
  esac
}

usage() {
  echo "Uso:"
  echo "  pretty_dictamen.sh [archivo.tsv] {all|counts|q4pass|q4fail|h1pass|h1fail|which|counts-by-file}"
  echo
  echo "Si no pasas archivo, autodetecta el último reports/mini_accum/dictamen_*.tsv con -oos."
}

mode=""; file=""

# Soporta: [file] mode | mode [file] | solo mode | solo file
if [[ $# -gt 0 ]]; then
  if [[ -r "${1:-}" ]]; then file="$1"; shift
  elif is_mode "${1:-}"; then mode="$1"; shift
  fi
fi
# ← CORRECCIÓN: no llamar funciones dentro de [[ ... ]]
if [[ -z "${mode:-}" && $# -gt 0 ]]; then
  if is_mode "${1:-}"; then mode="$1"; shift; fi
fi
: "${mode:=counts}"

pick_file() {
  local cand
  for cand in $(ls -t reports/mini_accum/dictamen_*.tsv 2>/dev/null); do
    [[ -r "$cand" ]] || continue
    if LC_ALL=C grep -qE -- '-oos(23Q4|24H1)\b' "$cand"; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  ls -t reports/mini_accum/dictamen_*.tsv 2>/dev/null | head -1 || true
}

[[ -z "${file:-}" ]] && file="$(pick_file || true)"
if [[ -z "${file:-}" || ! -r "$file" ]]; then
  echo "No encontré dictamen TSV legible. Esperaba: reports/mini_accum/dictamen_*.tsv" >&2
  exit 1
fi

header(){ printf 'suffix\twindow\tnetBTC\tdbps\tmdd_vs_HODL\tflips\tPASS\tFAIL\n'; }

case "$mode" in
  which)
    echo "Usando: <$file>"
    ;;

  counts)
    echo "Usando: <$file>" >&2
    awk -F'\t' '{
      sub(/\r$/,"",$1)
      if ($1 ~ /-oos23Q4$/) { if ($7=="True") q4p++; else q4f++ }
      if ($1 ~ /-oos24H1$/) { if ($7=="True") h1p++; else h1f++ }
    } END {
      printf "Q4 oos23Q4: PASS=%d FAIL=%d\n", q4p+0, q4f+0
      printf "H1 oos24H1: PASS=%d FAIL=%d\n", h1p+0, h1f+0
    }' "$file"
    ;;

  counts-by-file)
    for f in $(ls -t reports/mini_accum/dictamen_*.tsv 2>/dev/null | head -5); do
      awk -v F="$f" -F'\t' '{
        sub(/\r$/,"",$1)
        if ($1 ~ /-oos23Q4$/) { if ($7=="True") q4p++; else q4f++ }
        if ($1 ~ /-oos24H1$/) { if ($7=="True") h1p++; else h1f++ }
      } END {
        printf "%s => Q4: P=%d F=%d | H1: P=%d F=%d\n", F, q4p+0, q4f+0, h1p+0, h1f+0
      }' "$f"
    done
    ;;

  all)
    echo "Usando: <$file>" >&2
    { header
      awk -F'\t' 'BEGIN{OFS="\t"} {sub(/\r$/,"",$1)}
        $1 ~ /^v3p3N2g-F2-(H1_FZ|Q4_E3)(-|$)/ {print $1,$2,$3,$4,$5,$7,$8,$9}
      ' "$file"
    } | column -t -s $'\t'
    ;;

  q4fail)
    echo "Usando: <$file>" >&2
    { header
      awk -F'\t' 'BEGIN{OFS="\t"} {sub(/\r$/,"",$1)}
        $1 ~ /^v3p3N2g-F2-(H1_FZ|Q4_E3)-.*-oos23Q4$/ && $7!="True" {print $1,$2,$3,$4,$5,$7,$8,$9}
      ' "$file"
    } | column -t -s $'\t'
    ;;

  h1fail)
    echo "Usando: <$file>" >&2
    { header
      awk -F'\t' 'BEGIN{OFS="\t"} {sub(/\r$/,"",$1)}
        $1 ~ /^v3p3N2g-F2-(H1_FZ|Q4_E3)-.*-oos24H1$/ && $7!="True" {print $1,$2,$3,$4,$5,$7,$8,$9}
      ' "$file"
    } | column -t -s $'\t'
    ;;

  q4pass)
    echo "Usando: <$file>" >&2
    { header
      awk -F'\t' 'BEGIN{OFS="\t"} {sub(/\r$/,"",$1)}
        $1 ~ /^v3p3N2g-F2-(H1_FZ|Q4_E3)-.*-oos23Q4$/ && $7=="True" {print $1,$2,$3,$4,$5,$7,$8,$9}
      ' "$file"
    } | column -t -s $'\t'
    ;;

  h1pass)
    echo "Usando: <$file>" >&2
    { header
      awk -F'\t' 'BEGIN{OFS="\t"} {sub(/\r$/,"",$1)}
        $1 ~ /^v3p3N2g-F2-(H1_FZ|Q4_E3)-.*-oos24H1$/ && $7=="True" {print $1,$2,$3,$4,$5,$7,$8,$9}
      ' "$file"
    } | column -t -s $'\t'
    ;;

  *) usage; exit 2;;
esac
