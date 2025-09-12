# Baseline H1_FZ (FREEZE)

- Ventana: 2024-01-01 → 2024-06-30 (H1'24 holdout)
- Señal: EMA(13/55), **exit_active desactivada**
- Anti-whipsaw: dwell_min=76, sin pausa, sin ATR regime
- Presupuesto: hard_per_year=26

## Motivación
Replicar/estabilizar el desempeño del TOP en H1 con menor whipsaw. Resultados: PASS y mejora vs TOP.

## Archivos
- `H1_FZ_equity.csv`, `H1_FZ_kpis.csv`, `H1_FZ_summary.md`, `H1_FZ_flips.csv`, `F2_H1_FZ.yaml`
- Ver `MANIFEST.json` para hash y métricas clave.
