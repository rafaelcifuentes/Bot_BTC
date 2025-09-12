# Sweep Pack v2b — `scripts/mini_accum/*`

Este pack reubica los Bash a `scripts/mini_accum/` (opcional subcarpeta `FX/`) para alinearse con tu repo.
Funciona con el CLI `mini-accum-backtest` y `mini-accum-dictamen` en tu `.venv`.

## Estructura
```
scripts/mini_accum/
  render.sh
  run_F1.sh       # xb × dw
  run_F1b_mb.sh   # micro-sweep de macro_buffer
  run_F2.sh       # ATR gating
  run_F3.sh       # exit delta 5/10 bps (+ confirm 0/1)
  run_F4.sh       # EMA fast 12/14
  analyze.sh      # agrega dictamen + stats de flips
  flip_stats.py   # min4h / weeks>1 por flips CSV
configs/templates/
  mini_accum__template.yaml
  mini_accum__template_exitdelta.yaml
```

> Ruta base inferida: cada script calcula `REPO_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"`.

## Requisitos
- macOS/Linux con bash/zsh
- Python 3.11 + `.venv` con tus bins (`.venv/bin/mini-accum-backtest`, etc.)
- CSVs de OHLC en `data/ohlc/_rescued/...` (mismo layout que ya usas)

## Uso rápido
```bash
bash scripts/mini_accum/run_F1.sh
XB=36 DW=72 bash scripts/mini_accum/run_F1b_mb.sh
XB=36 DW=72 bash scripts/mini_accum/run_F2.sh
XB=36 DW=72 bash scripts/mini_accum/run_F3.sh
XB=36 DW=72 bash scripts/mini_accum/run_F4.sh
bash scripts/mini_accum/analyze.sh
```
