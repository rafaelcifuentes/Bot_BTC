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



## Corazón — FREEZE semanal 2025-09-08 (sombra)

**KPIs (overlay vs base)**  
- PF: **2.4425** vs **2.1744** (**+12.4%**)  
- MDD: **−0.8423%** vs **−1.7351%** (**−51.4%**)  
- σ: **0.000958** vs **0.001746** (**−45.1%**)  
- NET: **0.03055** vs **0.05318** (**−42.6%**) — baja esperada por gating; recuperable con ξ*.

**ξ***: **1.5492×** (min(MDD_ratio **2.0599**, σ_ratio **1.8226**) × 0.85).  
**Veredicto:** **PASS**. Mantener evaluación en sombra; si se sostiene 1–2 semanas → apto para **guardrail opt‑in (W5)**.

**Acciones**
- Producción esta semana: mantener **ATR_MAX=0.07**.  
- Shadow próximo lunes: `heart_monday "2025-09-15 00:00" 0.08 cg20_t035`.  
- Registrar KPIs y ξ* en `corazon/daily_xi.csv` (ya guardado).  
- No tocar Cerebro todavía; sizing 1.55× preparado en el hilo del Allocator.

**Artefactos**
- `reports/heart/diamante_overlay_diamante_btc_costes_freeze_2025-09-08_bars.csv`  
- `reports/heart/kpis_diamante_btc_costes_freeze_2025-09-08_bars.csv`  
- `reports/heart/summary_diamante_btc_costes_freeze_2025-09-08_bars.md`
