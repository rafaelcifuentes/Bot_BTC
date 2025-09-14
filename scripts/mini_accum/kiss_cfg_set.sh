#!/usr/bin/env bash
set -Eeuo pipefail
CFG="${KISS_CFG:-configs/mini_accum/config.yaml}"
EXPR="${1:?falta expresión yq}"
chflags nouchg "$CFG" 2>/dev/null || true
yq -i "$EXPR" "$CFG"
chflags uchg "$CFG" 2>/dev/null || true
echo "[OK] aplicado: $EXPR"
