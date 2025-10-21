#!/usr/bin/env bash
set -euo pipefail

MD="reports/mini_accum/COMPARISON/v2_vs_v1_summary.md"

lower(){ printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

kpi_from_freeze(){ # $1=freeze_file
  local f="$1" s="" m=""
  [[ -s "$f" ]] || return 0
  s=$(yq -r '.kpis.sats_mult // empty' "$f" 2>/dev/null || true)
  m=$(yq -r '.kpis.mdd_vs_hodl // empty' "$f" 2>/dev/null || true)
  s="$(printf '%s' "$s" | tr -d '[:space:]')"
  m="$(printf '%s' "$m" | tr -d '[:space:]')"
  [[ "$(lower "$s")" = "nan" ]] && s=""
  [[ "$(lower "$m")" = "nan" ]] && m=""
  [[ -n "$s" && -n "$m" ]] && printf '%s %s\n' "$s" "$m"
}

kpi_from_md(){ # $1=year_label (2022|2023|2024|2025H1)
  local y="$1"
  awk -v Y="$y" -F'|' '
    $0 ~ "\\|[[:space:]]*"Y"[[:space:]]*\\|" {
      s=$3; m=$4
      gsub(/[[:space:]]/,"",s); gsub(/[[:space:]]/,"",m)
      if (s!="" && tolower(s)!="nan" && m!="" && tolower(m)!="nan") { print s,m; exit }
    }' "$MD" 2>/dev/null || true
}

retag_one(){ # $1=YR_LABEL $2=TAG $3=FREEZE_PATH
  local Y="$1" TAG="$2" F="$3" D1H="" H4H="" SATS="" MDD=""
  [[ -s "$F" ]] && {
    D1H="$(awk '/^data_1d_sha256:/ {print $2}' "$F" 2>/dev/null)"
    H4H="$(awk '/^data_4h_sha256:/ {print $2}' "$F" 2>/dev/null)"
  }
  read -r SATS MDD <<<"$(kpi_from_freeze "$F" || true)" || true
  [[ -z "${SATS:-}" || -z "${MDD:-}" ]] && read -r SATS MDD <<<"$(kpi_from_md "$Y" || true)" || true

  case "$(lower "${SATS:-}")" in ""|"null"|"nan") echo "[SKIP] $TAG ⇒ KPIs vacíos"; return 0;; esac
  case "$(lower "${MDD:-}")"  in ""|"null"|"nan") echo "[SKIP] $TAG ⇒ KPIs vacíos"; return 0;; esac

  git tag -a "$TAG" -f -m "NetBTC=${SATS} MDD_vs_HODL=${MDD} | 1D=${D1H} 4H=${H4H} | costs fee=2bps slip=1bps"
  echo "[OK] $TAG ⇒ NetBTC=$SATS  MDD=$MDD"
}

main(){
  retag_one 2022   PROD_E1_Y2_2022   "reports/mini_accum/_freezes/E1_Y2_2022.freeze.txt"
  retag_one 2023   PROD_KISSv1_2023  "reports/mini_accum/_freezes/V1TOP_2023.freeze.txt"
  retag_one 2024   PROD_KISSv1_2024  "reports/mini_accum/_freezes/V1TOP_2024.freeze.txt"
  retag_one 2025H1 PROD_KISSv1_2025H1 "reports/mini_accum/_freezes/V1TOP_2025H1.freeze.txt"
  git tag -n | grep -E 'PROD_(E1_Y2|KISSv1)_(2022|2023|2024|2025H1)' || true
}
main "$@"
