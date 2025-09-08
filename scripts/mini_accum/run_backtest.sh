#!/usr/bin/env bash
set -euo pipefail
# Ejecuta el CLI empaquetado leyendo configs/mini_accum.yaml
mini-accum-backtest --config "configs/mini_accum.yaml" "$@"
