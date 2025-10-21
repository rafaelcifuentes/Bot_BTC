#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/deploy/live_fee_slip"
ACTIVE_TAG="$ROOT/deploy/ACTIVE.tag"

# Defaults prudentes si no hay recibo
fee=3
slip=3

parse_line_kv(){ awk -F= -v k="$1" '$1==k{gsub(/[[:space:]]/,"",$2); print $2; exit}'; }
parse_csv_col(){ awk -F, -v k="$1" 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next} NR==2{print $h[k]; exit}'; }

if [[ -s "$RUNTIME" ]]; then
  # Intenta KV
  f_kv=$(parse_line_kv fee_bps_per_side < "$RUNTIME" || true)
  s_kv=$(parse_line_kv slip_bps_per_side < "$RUNTIME" || true)
  # Si no hay KV, intenta CSV
  if [[ -z "${f_kv:-}" || -z "${s_kv:-}" ]]; then
    f_csv=$(parse_csv_col fee_bps_per_side < "$RUNTIME" || true)
    s_csv=$(parse_csv_col slip_bps_per_side < "$RUNTIME" || true)
  fi
  fee="${f_kv:-${f_csv:-$fee}}"
  slip="${s_kv:-${s_csv:-$slip}}"
fi

# Normaliza a números
fee=$(printf "%.6g" "${fee}")
slip=$(printf "%.6g" "${slip}")

# Regla de decisión
new_tag="DRIVE_2023"
awk "BEGIN{exit !(${fee} <= 2 && ${slip} <= 2)}" && new_tag="SPORT_2024"

# Lee tag actual (si existe)
old_tag="UNKNOWN"
[[ -s "$ACTIVE_TAG" ]] && old_tag="$(cat "$ACTIVE_TAG" | tr -d '[:space:]')"

# Idempotente: sólo escribe si cambia
if [[ "$new_tag" != "$old_tag" ]]; then
  echo "$new_tag" > "$ACTIVE_TAG"
  TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mkdir -p "$ROOT/logs"
  echo "$TS select_tag: fee=${fee}bps side, slip=${slip}bps side → $new_tag (was: $old_tag)" >> "$ROOT/logs/deploy.log"
  echo "[APPLY] $old_tag → $new_tag  (fee=${fee}, slip=${slip})"
else
  echo "[KEEP]  $old_tag (fee=${fee}, slip=${slip})"
fi
