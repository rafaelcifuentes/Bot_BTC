# Mini Accum (KISS) — Baseline

**Fecha del PASS:** 2025-09-13  
**Ventana:** 2025-05-10 → 2025-09-07 (BTC-USD 4h, macro D1)

**Config única:** `configs/mini_accum/config.yaml`  
**Run diag:** `bash scripts/mini_accum/diag_gate_mix.sh`

## Parámetros (KISSv4)
- Entrada: `ema21 > ema55 and macro_green`
- Salida activa: `close < ema21 and macro_red` (confirm=2) + `exit_margin=0.003`
- Filtros: `ADX14 ≥ 22`, `macro_filter.use=true`, `ATR regime p65 (on)`
- Anti-whipsaw: `dwell_min_between_flips=96`
- Costes: fee/slip = 6/6 bps por lado
- Reports: `reports/mini_accum/`

## KPIs del PASS
- `netBTC=1.0130`, `mdd_vs_HODL=0.716`, `fpy=18.42` (6 flips)
- Ver detalle en `docs/runs/*_PASS.md` (autogenerable abajo).

## Próximos pasos (en foco)
1) Afinar `exit_margin` 30→35 bps si FPY ≤ 26.  
2) ADX 20 vs 22 con ATR p65 y medir netBTC/MDD/FPY.  
3) (Opcional) `dwell` 96↔72 si vemos re-entradas tardías.
