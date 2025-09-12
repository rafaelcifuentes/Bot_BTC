## Baseline congelado: H1_FZ (FREEZE)
**Config:** `configs/baselines/F2_H1_FZ.yaml`
Ventana: 2024H1 (holdout)
Resultados vs TOP:
- H1_FZ: netBTC 1.0889, dbps +382.7, mdd_vs_HODL 0.789, flips 8 (fpy 16.2) — PASS
- TOP:   netBTC 1.0506, dbps -0.5,   mdd_vs_HODL 0.730, flips 12 (fpy 24.4)

Cambios vs TOP:
- exit_active.enabled=false
- dwell_bars_min_between_flips=76
- pause_after_flip_bars=0
- atr_regime.enabled=false
