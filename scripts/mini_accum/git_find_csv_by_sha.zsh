#!/usr/bin/env zsh
set -euo pipefail
ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
cd "$ROOT"

WANT="${1:-}"
[[ -n "$WANT" ]] || { echo "Uso: $0 <sha256>"; exit 1; }

tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
found=0
for c in $(git rev-list --all); do
  # lista solo CSV pesados para reducir ruido
  git ls-tree -r --name-only "$c" | grep -Ei '\.csv$' | while read -r p; do
    git show "$c:$p" > "$tmp" 2>/dev/null || continue
    h="$(shasum -a 256 "$tmp" | awk '{print $1}')"
    if [[ "$h" == "$WANT" ]]; then
      echo "[FOUND] commit=$c path=$p"
      found=1
      # guardamos copia en cache canon
      out="$ROOT/data/_canon/$(basename "$(dirname "$p")")/$(basename "${p%.csv}")__${WANT}.csv"
      mkdir -p "$(dirname "$out")"
      mv "$tmp" "$out"
      echo "[CACHED] $out"
      # rearmar tmp para seguir buscando por si hay más apariciones
      tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
    fi
  done
done

[[ "$found" == "1" ]] || { echo "[MISS] No hallé ese sha en el repo"; exit 2; }
