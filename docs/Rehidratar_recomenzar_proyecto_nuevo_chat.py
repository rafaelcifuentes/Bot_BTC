# Proyecto: Bot_BTC — Fase A (swing_4h_forward_diamond)

## 1) Objetivo (sigue vigente)
- Validar **estabilidad** (PF, WR, MDD, Sortino) y **competitividad** con costes realistas.
- Medir desempeño semanal vs **B&H**.
- Decidir en semanas 4–6 si pasamos a **paper trading** (conservador).

## 2) Principios (inmutables)
- **Reproducibilidad:** usar `--freeze_end` y/o `--max_bars`.
- **Conexión:** `EXCHANGE=binanceus` (evita 451), `--skip_yf` cuando aplique.
- **Datos:** `_fetch_ccxt` con paginación `limit=1500`.

## 3) Estado / Hito actual
- Estamos en **Fase A — Diamante**, Día 3: **Micro-grid** (threshold / tp / sl / partial) y selección por validación.
- Producción real: **solo BTC**. ETH (y SOL si conviene) solo para robustez y refinamiento.
- Último dry-run (ejemplo BTC): ENTRY≈113198.165, STOP≈111500.193, TP1≈114896.137, TP2≈116594.110, RISK=0.0075, SIZE≈0.03091923. (Modo `dry`; no coloca órdenes reales).

## 4) Datos y rutas
- OHLC 1m:
  - `data/ohlc/1m/BTC-USD.csv`  (y `ETH-USD.csv` si hace falta)
- Señales por ventana (proxy o reales):
  - `reports/windows/2022H2/BTC-USD.csv`
  - `reports/windows/2023Q4/BTC-USD.csv`
  - `reports/windows/2024H1/BTC-USD.csv`
- Resultados micro-grid:
  - KPIs: `reports/kpis_grid.csv`
  - Top: `reports/top_configs.csv`
  - Selección: `reports/microgrid_selection.json` (**status: full_coverage**) y `reports/microgrid_selection.md`
  - Comparativa vs B&H: `reports/selected_vs_bh.md`
- Logs/colocación:
  - `logs/${RUN_ID}_dry.log`
  - Preview órdenes: `reports/orders_preview_${RUN_ID}.csv` (ej. `orders_preview_wk1_wf3_th060_gate.csv`)

## 5) Parámetros / flags de trabajo (usar como base)
- Ventanas: `2022H2:2022-06-01:2023-01-31` | `2023Q4:2023-10-01:2023-12-31` | `2024H1:2024-01-01:2024-06-30`
- Activos: `BTC-USD` (producción), `ETH-USD` opcional para robustez
- Horizontes: `15 30 60 120`
- Thresholds: `0.50 0.55 0.60 0.65`
- Costes: `--fee_bps 6 --slip_bps 6`
- Gestión: `--partial 50_50 --breakeven_after_tp1`
- Riesgo: `--risk_total_pct 0.75`
- Pesos: `--weights BTC-USD=1.0` (si incluyes ETH: `BTC-USD=0.7 ETH-USD=0.3`)
- Gates (ajustar según validación): `--gate_pf 1.6 --gate_wr 0.60 --gate_trades 30`

## 6) Comandos rápidos (copiar/pegar)

# (A) Generar/actualizar señales proxy por ventana (si no usas las reales)
python scripts/gen_proxy_signals.py \
  --ohlc_csv data/ohlc/1m/BTC-USD.csv \
  --asset BTC-USD \
  --windows "2022H2:2022-06-01:2023-01-31" "2023Q4:2023-10-01:2023-12-31" "2024H1:2024-01-01:2024-06-30" \
  --out_root reports/windows \
  --resample 4h --lookback 10

# (B) Grid de backtests (BTC-only)
python backtest_grid.py \
  --windows "2022H2:2022-06-01:2023-01-31" "2023Q4:2023-10-01:2023-12-31" "2024H1:2024-01-01:2024-06-30" \
  --assets BTC-USD \
  --horizons 15 30 60 120 \
  --thresholds 0.50 0.55 0.60 0.65 \
  --signals_root reports/windows \
  --ohlc_root data/ohlc/1m \
  --fee_bps 6 --slip_bps 6 \
  --partial 50_50 --breakeven_after_tp1 \
  --risk_total_pct 0.75 \
  --weights BTC-USD=1.0 \
  --gate_pf 1.6 --gate_wr 0.60 --gate_trades 30 \
  --out_csv reports/kpis_grid.csv \
  --out_top reports/top_configs.csv

# (C) Selección micro-grid y reporte
python scripts/select_microgrid_config.py \
  --kpis_csv reports/kpis_grid.csv \
  --windows 2022H2 2023Q4 2024H1 \
  --asset BTC-USD \
  --gate_pf 1.6 --gate_wr 0.60 --gate_trades 30 \
  --out_json reports/microgrid_selection.json \
  --out_md   reports/microgrid_selection.md

python scripts/report_selected_vs_bh.py \
  --kpis_csv reports/kpis_grid.csv \
  --selection_json reports/microgrid_selection.json \
  --out_md reports/selected_vs_bh.md

# (D) Dry-run de colocación (ajusta RUN_ID y pesos)
export CAPITAL_USD=10000
export RUN_ID=wk1_wf3_th060_gate
export STOP_BTC_PCT=0.015
bash ./run_diamante_place.sh dry && tail -n 120 logs/${RUN_ID}_dry.log

## 7) Notas útiles
- Si ves errores de **tz-naive vs tz-aware**, ya normalizamos a UTC en `gen_proxy_signals.py`.
- Producción: mantener **BTC-only**; ETH solo para validar robustez out-of-sample.