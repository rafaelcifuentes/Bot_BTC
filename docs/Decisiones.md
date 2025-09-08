# Decisiones operacionales — Bot BTC
Fecha: 2025-09-08 22:11 UTC

## Estado (shadows activos)
- **cb0965_shadow** → PF=1.03 (ΔPF=−0.03), |MDD_overlay|/|MDD_base|=0.63
- **wfloor045_shadow** → PF=1.03 (ΔPF=−0.03), |MDD_overlay|/|MDD_base|=0.63

| rid                                   |   pf |   dpf |   mdd_ratio | act  |
|:--------------------------------------|-----:|------:|------------:|:-----|
| slim_ema200_atrpct_20250908_cb0965_shadow  | 1.03 | -0.03 |        0.63 | —    |
| slim_ema200_atrpct_20250908_wfloor045_shadow| 1.03 | -0.03 |        0.63 | —    |

## Decisión operacional
- **Esta semana:** mantener ambos *shadows* activos y revisar sus `*_vs_base.md` el fin de semana.
- **Próximo lunes:** ejecutar **ATR_MAX=0.08** con `heart_monday`.
- **Perla:** si es necesario, usar **EMA soft (band=0.990, ttl=1)** en *shadow* (o *prod* vía symlink) y medir en el dashboard semanal.

## Próximo lunes (ejemplo)
```bash
heart_monday "2025-09-15 00:00" 0.08 cg20_t035
```
