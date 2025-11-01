#!/usr/bin/env zsh
set -euo pipefail

ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
cd "$ROOT"

# Hashes canónicos (contrato CORE 2025H1)
EXPECT_HASH_4H="2bf8c646589db1cd52fdd7b4bfc822860d9b8283fe3f2e129961732a3ff0d947"
EXPECT_HASH_1D="4fc2fdcac21ac1f9acf0ceb624d1622c990b3e4e76302cf3d624c448ebc2441b"

restore_one() {
  local rel="$1"     # e.g., data/ohlc/4h/BTC-USD.csv
  local want="$2"    # sha256 esperado

  local abs="$ROOT/$rel"
  local tf="$(echo "$rel" | awk -F/ '{print $(NF-1)}')"     # 4h o 1d
  local base="$(basename "$rel")"                           # BTC-USD.csv
  local stem="${base%.csv}"
  local canon_dir="$ROOT/data/_canon/$tf"
  local canon_file="$canon_dir/${stem}__${want}.csv"

  mkdir -p "$canon_dir" "$ROOT/data/_bk"

  # 0) backup del actual si existe
  if [[ -s "$abs" ]]; then
    cursha="$(shasum -a 256 "$abs" | awk '{print $1}')"
    ts="$(date -u +%Y%m%dT%H%M%SZ)"
    cp "$abs" "$ROOT/data/_bk/${stem}_${tf}_${ts}__${cursha}.csv"
    echo "[BKUP] $rel -> data/_bk/${stem}_${tf}_${ts}__${cursha}.csv"
  fi

  # 1) si ya está en caché, linkeamos
  if [[ -s "$canon_file" ]]; then
    ln -sfn "../../_canon/$tf/${stem}__${want}.csv" "$abs"
    echo "[LINK] $rel -> ${canon_file#$ROOT/}"
    return 0
  fi

  echo "[SEARCH] Buscando en historial git el contenido de $rel con sha=$want ..."
  tmp="$(mktemp)"
  found=0
  # Limita el recorrido a commits que tocaron ese path
  for c in $(git rev-list --all -- "$rel"); do
    # Algunas entradas pueden no existir en ciertos commits → silencia
    if git show "$c:$rel" > "$tmp" 2>/dev/null; then
      h="$(shasum -a 256 "$tmp" | awk '{print $1}')"
      if [[ "$h" == "$want" ]]; then
        mv "$tmp" "$canon_file"
        ln -sfn "../../_canon/$tf/${stem}__${want}.csv" "$abs"
        echo "[RESTORED] $rel desde commit $c → ${canon_file#$ROOT/}"
        found=1
        break
      fi
    fi
  done
  rm -f "$tmp" || true

  if [[ "$found" == "0" ]]; then
    echo "[ERR] No encontré en git una versión de $rel con sha=$want"
    echo "      → Necesitas traer el CSV canónico por otro medio (backup externo) y guardarlo en:"
    echo "        $canon_file"
    return 1
  fi
}

# ---- Ejecuta restauraciones ----
restore_one "data/ohlc/4h/BTC-USD.csv" "$EXPECT_HASH_4H"
restore_one "data/ohlc/1d/BTC-USD.csv" "$EXPECT_HASH_1D"

# Verificación final de hashes activos
echo "[CHECK] Activos:"
echo "4H = \$(shasum -a 256 data/ohlc/4h/BTC-USD.csv | awk '{print \$1}')"
echo "1D = \$(shasum -a 256 data/ohlc/1d/BTC-USD.csv | awk '{print \$1}')"

# Opcional: re-correr el reconstructor
if [[ -x "$ROOT/scripts/mini_accum/reconstruct_core_2025H1_exact.zsh" ]]; then
  echo "[RUN] reconstruct_core_2025H1_exact.zsh"
  "$ROOT/scripts/mini_accum/reconstruct_core_2025H1_exact.zsh"
fi
