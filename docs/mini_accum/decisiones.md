
## 2025-10-11 — Decisión: mantener **KISS v1 BASE**; overlays OFF
- **Base OOS 2025H1 (DD15/RB1/H30/G200/BULL0)**:
  - sats_mult = 1.138462  (≈ +13.85% en 6m)
  - mdd_vs_hodl = 0.741494
  - flips_total = 2
- **Overlays estáticos (10×30 y 12×40) OOS 2025H1**:
  - sats_mult ≈ 0.903167  → lift ≈ -20.67% vs base  ❌
  - mdd_vs_hodl ≈ 0.394  (mejor riesgo), flips ≈ 5
  - Gate: **FAIL** (lift < +5% ⇒ no promueve, aunque el MDD mejore)
- **v1.1 (H29/H31 y/o RB2)**: lift ≤ 0% o negativo → **FAIL**.
- **Acción**: mantener **DD15 / RB1 / H30 / G200 / BULL0** en PROD; overlays quedan **OFF**.
- **Regla de oro ACCUM**: promover solo si NetBTC > HODL **al mismo o menor MDD**.

## 2025-10-11 — v1.2 (SL/TP estático) A/B OOS 2025H1 → FAIL gate
- Base: RB1/H30/G200/DD15/BULL0 → sats=1.138462, mdd_vs_hodl=0.741494, flips=2.
- Cands: SLTP 10×30 y 12×40 → sats=0.903167, mdd_vs_hodl=0.394118, flips=5.
- Lift vs base: -20.67% (req. ≥ +5%). MDD mejora, pero falla criterio de éxito (NetBTC/SATS).
- Acción: mantener KISS v1 BASE en PROD; overlays SLTP **OFF** (experimento).

## 2025-10-11 — Cierre v1.2 y Release PROD
- v1.2 (SLTP 10×30, 12×40) → FAIL (lift ~ –20.67%).
- Se congela KISS v1 BASE (RB1/H30/G200/BULL0) en PROD.
- Tag release: 
- Gate queda como guardián por defecto (≥ +5%, MDD ≤ base; robustez si aplica).
- Añadido assert_kpi_has_sats para evitar falsos positivos.

## 2025-10-11 — v1.2 (WF 2022/2023/2025) ejecutado
- Tabla NetBTC por ventana (objetivo ≥1.0). Gate OOS 2025H1 vs BASE si disponible.

## 2025-10-11 — v1.2 WF_2025 → FAIL gate
- BASE 2025H1: sats=1.138462, mdd_vs_hodl=0.741494, flips=2.
- v1.2 WF 2025: sats=0.917246, mdd_vs_hodl=0.394118, flips=9.
- Lift vs BASE: –19.43% (req. ≥ +5%). MDD mejora, pero no cedemos sats → **NO PROMOVER**.
- Acción: v1.2 **OFF**; mantener **KISS v1 BASE** en PROD.

## 2025-10-11 — v1.2 (WF_2025) FAIL gate
- BASE OOS 2025H1: sats=1.138462, mdd_vs_hodl=0.741494, flips=2
- v1.2 WF_2025:     sats=0.917246, mdd_vs_hodl=0.394118, flips=9
- Lift vs BASE: –19.43% (req. ≥ +5%). Acción: mantener BASE; v1.2 OFF.

## 2025-10-11 — Gate v1.2 vs BASE (WF_2025)
- BASE OOS 2025H1: sats=1.138462; CAND v1.2 WF_2025: sats=0.917246
- Lift: –19.43%  → FAIL (umbral +5%); acción: mantener BASE; v1.2 OFF

## 2025-10-12 — Gate FAIL v2.0 (WF_2025)
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/base_v0_1_20251012_0050_kpis__WF_2025_v2_0.csv
- Resultado: **FAIL** (lift −19.43% < +5% req.; MDD mejora pero no supera criterio de lift)
- Métricas: sats_BASE=1.138462, sats_CAND=0.917246, ΔMDD=−0.347376, flips: base=2 cand=9
- Decisión: mantener KISS v1 BASE en PROD; **v2.0 OFF (opt-in)**; seguir iterando en rama.
## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3b_offhib
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1620_kpis__WF_2025_v2_0_E3b_offhib.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.326297)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.415198, flips: base=2 cand=1
- Decisión: mantener BASE; candidato OFF (opt-in)
## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3b_offhib
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1620_kpis__WF_2025_v2_0_E3b_offhib.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.326297)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.415198, flips: base=2 cand=1
- Decisión: mantener BASE; candidato OFF (opt-in)
## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3c
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1623_kpis__WF_2025_v2_0_E3c.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.326297)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.415198, flips: base=2 cand=1
- Decisión: mantener BASE; candidato OFF (opt-in)
## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3c_wide
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1624_kpis__WF_2025_v2_0_E3c_wide.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.326297)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.415198, flips: base=2 cand=1
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3b_offhib
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1747_kpis__WF_2025_v2_0_E3b_offhib.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.326297)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.415198, flips: base=2 cand=1; fpy: base=4.06 cand=1.42
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3b_offhib
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1747_kpis__WF_2025_v2_0_E3b_offhib.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.326297)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.415198, flips: base=2 cand=1; fpy: base=4.06 cand=1.42
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3b_offhib
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1747_kpis__WF_2025_v2_0_E3b_offhib.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.326297)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.415198, flips: base=2 cand=1; fpy: base=4.06 cand=1.42
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3b_offhib
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1747_kpis__WF_2025_v2_0_E3b_offhib.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.326297)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.415198, flips: base=2 cand=1; fpy: base=4.06 cand=1.42
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3b_offhib_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1756_kpis__WF_2025_v2_0_E3b_offhib_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3b_offhib_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1756_kpis__WF_2025_v2_0_E3b_offhib_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3b_offhib_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1809_kpis__WF_2025_v2_0_E3b_offhib_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3b_offhib_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1809_kpis__WF_2025_v2_0_E3b_offhib_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL v2.0 (preset: E3b_offhib, ventana: 2025-01-01..2025-06-30)
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1809_kpis__WF_2025_v2_0_E3b_offhib_H1.csv
- Resultado: **FAIL** — lift **−14.84%**, ΔMDD **−0.3651**, FPY base **4.06**, FPY cand **2.03**.
- Motivo: No cumple lift ≥ +5% pese a menor MDD; v1 permanece en PROD; v2 **OFF (opt-in)**.
- Acción: iterar variantes minimalistas apples-to-apples y re-evaluar.

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E1_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1903_kpis__WF_2025_v2_0_E1_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E2_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1903_kpis__WF_2025_v2_0_E2_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1903_kpis__WF_2025_v2_0_E3_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3a_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1903_kpis__WF_2025_v2_0_E3a_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3b_offhib_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1903_kpis__WF_2025_v2_0_E3b_offhib_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3c_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1903_kpis__WF_2025_v2_0_E3c_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3c_wide_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1903_kpis__WF_2025_v2_0_E3c_wide_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E1_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1910_kpis__WF_2025_v2_0_E1_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E2_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1910_kpis__WF_2025_v2_0_E2_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1910_kpis__WF_2025_v2_0_E3_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3a_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1910_kpis__WF_2025_v2_0_E3a_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3b_offhib_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1910_kpis__WF_2025_v2_0_E3b_offhib_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3c_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1910_kpis__WF_2025_v2_0_E3c_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3c_wide_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1910_kpis__WF_2025_v2_0_E3c_wide_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E1_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1914_kpis__WF_2025_v2_0_E1_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E2_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1914_kpis__WF_2025_v2_0_E2_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1914_kpis__WF_2025_v2_0_E3_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3a_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1914_kpis__WF_2025_v2_0_E3a_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3b_offhib_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1914_kpis__WF_2025_v2_0_E3b_offhib_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3c_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1914_kpis__WF_2025_v2_0_E3c_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3c_wide_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1914_kpis__WF_2025_v2_0_E3c_wide_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E1_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1919_kpis__WF_2025_v2_0_E1_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E2_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1919_kpis__WF_2025_v2_0_E2_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1919_kpis__WF_2025_v2_0_E3_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3a_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1919_kpis__WF_2025_v2_0_E3a_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3b_offhib_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1919_kpis__WF_2025_v2_0_E3b_offhib_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3b_onhib_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1919_kpis__WF_2025_v2_0_E3b_onhib_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3c_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1919_kpis__WF_2025_v2_0_E3c_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3c_wide_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1919_kpis__WF_2025_v2_0_E3c_wide_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2025_v2_0_E3d_adx22_H1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251012_1919_kpis__WF_2025_v2_0_E3d_adx22_H1.csv
- Resultado: **FAIL** (lift -14.84%; ΔMDD=-0.365123)
- Métricas: sats_BASE=1.288888, sats_CAND=1.097678, mdd_BASE=0.741494, mdd_CAND=0.376372, flips: base=2 cand=1; fpy: base=4.06 cand=2.03
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2023_v2_0_E1
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251013_0127_kpis__WF_2023_v2_0_E1.csv
- Resultado: **FAIL** (lift -22.96%; ΔMDD=+0.258506)
- Métricas: sats_BASE=1.288888, sats_CAND=0.992946, mdd_BASE=0.741494, mdd_CAND=1.000000, flips: base=2 cand=7; fpy: base=4.06 cand=7.02
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2023_v2_0_E2
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251013_0127_kpis__WF_2023_v2_0_E2.csv
- Resultado: **FAIL** (lift -22.96%; ΔMDD=+0.258506)
- Métricas: sats_BASE=1.288888, sats_CAND=0.992946, mdd_BASE=0.741494, mdd_CAND=1.000000, flips: base=2 cand=7; fpy: base=4.06 cand=7.02
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2023_v2_0_E3
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251013_0127_kpis__WF_2023_v2_0_E3.csv
- Resultado: **FAIL** (lift -22.96%; ΔMDD=+0.258506)
- Métricas: sats_BASE=1.288888, sats_CAND=0.992946, mdd_BASE=0.741494, mdd_CAND=1.000000, flips: base=2 cand=7; fpy: base=4.06 cand=7.02
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2023_v2_0_E3a
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251013_0127_kpis__WF_2023_v2_0_E3a.csv
- Resultado: **FAIL** (lift -22.96%; ΔMDD=+0.258506)
- Métricas: sats_BASE=1.288888, sats_CAND=0.992946, mdd_BASE=0.741494, mdd_CAND=1.000000, flips: base=2 cand=7; fpy: base=4.06 cand=7.02
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2023_v2_0_E3b_offhib
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251013_0127_kpis__WF_2023_v2_0_E3b_offhib.csv
- Resultado: **FAIL** (lift -22.96%; ΔMDD=+0.258506)
- Métricas: sats_BASE=1.288888, sats_CAND=0.992946, mdd_BASE=0.741494, mdd_CAND=1.000000, flips: base=2 cand=7; fpy: base=4.06 cand=7.02
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2023_v2_0_E3b_onhib
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251013_0128_kpis__WF_2023_v2_0_E3b_onhib.csv
- Resultado: **FAIL** (lift -22.96%; ΔMDD=+0.258506)
- Métricas: sats_BASE=1.288888, sats_CAND=0.992946, mdd_BASE=0.741494, mdd_CAND=1.000000, flips: base=2 cand=7; fpy: base=4.06 cand=7.02
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2023_v2_0_E3c
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251013_0128_kpis__WF_2023_v2_0_E3c.csv
- Resultado: **FAIL** (lift -22.96%; ΔMDD=+0.258506)
- Métricas: sats_BASE=1.288888, sats_CAND=0.992946, mdd_BASE=0.741494, mdd_CAND=1.000000, flips: base=2 cand=7; fpy: base=4.06 cand=7.02
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2023_v2_0_E3c_wide
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251013_0128_kpis__WF_2023_v2_0_E3c_wide.csv
- Resultado: **FAIL** (lift -22.96%; ΔMDD=+0.258506)
- Métricas: sats_BASE=1.288888, sats_CAND=0.992946, mdd_BASE=0.741494, mdd_CAND=1.000000, flips: base=2 cand=7; fpy: base=4.06 cand=7.02
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL WF_2023_v2_0_E3d_adx22
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/WF_2025/v2_0/base_v0_1_20251013_0128_kpis__WF_2023_v2_0_E3d_adx22.csv
- Resultado: **FAIL** (lift -22.96%; ΔMDD=+0.258506)
- Métricas: sats_BASE=1.288888, sats_CAND=0.992946, mdd_BASE=0.741494, mdd_CAND=1.000000, flips: base=2 cand=7; fpy: base=4.06 cand=7.02
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate FAIL OOS_2023_G200_DD15_RB1_H30_BULL0
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/OOS_BASE_v1/base_v0_1_20251013_0200_kpis__OOS_2023_G200_DD15_RB1_H30_BULL0.csv
- Resultado: **FAIL** (lift -50.62%; ΔMDD=+0.486739)
- Métricas: sats_BASE=1.288888, sats_CAND=0.636413, mdd_BASE=0.741494, mdd_CAND=1.228233, flips: base=2 cand=154; fpy: base=4.06 cand=154.53
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate PASS OOS_2022_G200_DD15_RB1_H30_BULL0
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/OOS_BASE_v1/base_v0_1_20251013_0322_kpis__OOS_2022_G200_DD15_RB1_H30_BULL0.csv
- Resultado: **PASS** (lift 119.66%; ΔMDD=-0.596564)
- Métricas: sats_BASE=1.288888, sats_CAND=2.831177, mdd_BASE=0.741494, mdd_CAND=0.144930, flips: base=2 cand=10; fpy: base=4.06 cand=10.03
- Decisión: promover

## 2025-10-12 — Gate FAIL OOS_2023_G200_DD15_RB1_H30_BULL0
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/OOS_BASE_v1/base_v0_1_20251013_0322_kpis__OOS_2023_G200_DD15_RB1_H30_BULL0.csv
- Resultado: **FAIL** (lift -50.62%; ΔMDD=+0.486739)
- Métricas: sats_BASE=1.288888, sats_CAND=0.636413, mdd_BASE=0.741494, mdd_CAND=1.228233, flips: base=2 cand=154; fpy: base=4.06 cand=154.53
- Decisión: mantener BASE; candidato OFF (opt-in)

## 2025-10-12 — Gate PASS OOS_2022_G200_DD15_RB1_H30_BULL0
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/OOS_BASE_v1/base_v0_1_20251013_0323_kpis__OOS_2022_G200_DD15_RB1_H30_BULL0.csv
- Resultado: **PASS** (lift 119.66%; ΔMDD=-0.596564)
- Métricas: sats_BASE=1.288888, sats_CAND=2.831177, mdd_BASE=0.741494, mdd_CAND=0.144930, flips: base=2 cand=10; fpy: base=4.06 cand=10.03
- Decisión: promover

## 2025-10-12 — Gate FAIL OOS_2023_G200_DD15_RB1_H30_BULL0
- BASE: reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv
- CAND: reports/mini_accum/OOS_BASE_v1/base_v0_1_20251013_0323_kpis__OOS_2023_G200_DD15_RB1_H30_BULL0.csv
- Resultado: **FAIL** (lift -50.62%; ΔMDD=+0.486739)
- Métricas: sats_BASE=1.288888, sats_CAND=0.636413, mdd_BASE=0.741494, mdd_CAND=1.228233, flips: base=2 cand=154; fpy: base=4.06 cand=154.53
- Decisión: mantener BASE; candidato OFF (opt-in)

# 2025-10-153— ✅ Promoción: E1_Y2 (Año 2 post-halving, bar=1D)
- Preset: E1_Y2 (12/26 + RSI14 buy/sell 35/65 · ADX len14 min22 · macro_sma200 ON · dwell3)
- OOS 2022: NetBTC ~ 2.916 · MDD_vs_HODL ~ 0.1055 · flips = 8
- Freeze: reports/mini_accum/_freezes/E1_Y2_2022.freeze.txt
- Uso operativo: aplicar sólo en Year+2 post-halving; resto de años = KISS v1 TOP.
## 2025-10-17T23:27:06Z — Gate v2.0 (mini-accum)
```
[DBG] keys BASE: {'sats': None, 'mdd': None, 'fpy': 'flips_per_year', 'flips': 'flips_total'}
[DBG] keys CAND: {'sats': None, 'mdd': None, 'fpy': 'flips_per_year', 'flips': 'flips_total'}
[GATE] FAIL: KPI sin métrica trazable de equity/net (e.g. sats_mult).
```

## 2025-10-17T23:37:30Z — Gate v2.0 (mini-accum, oos24H1)
```
[DBG] BASE_KPI=docs/mini_accum/checkpoints/20251017_1908UTC/base_v0_1_20250912_0943_kpis__v3p3N2g-F2-Q4_E3-oos24H1.csv
[DBG] CAND_KPI=docs/mini_accum/checkpoints/20251017_1908UTC/base_v0_1_20250912_1002_kpis__v3p3N2g-F2-Q4_E3-oos24H1.csv
[DBG] BASE_EQ=/tmp/base_eq_oos24.csv
[DBG] CAND_EQ=/tmp/cand_eq_oos24.csv
[DBG] LIFT_MIN=5
[GATE] FAIL: KPI sin métrica trazable de equity/net y no se pasó --base-eq/--cand-eq.
```

## 2025-10-18T00:06:48Z — Gate mini-accum v2.0 (oos24H1 real)
```
[DBG] BASE_KPI=docs/mini_accum/checkpoints/20251017_1908UTC/WF_2023_kpis__v1_2.csv
[DBG] CAND_KPI=docs/mini_accum/checkpoints/20251017_1908UTC/WF_2024_kpis__v1_2.csv
[DBG] BASE_EQ=
[DBG] CAND_EQ=
[DBG] LIFT_MIN=5
[DBG] LABELS: BASE=WF_2023 CAND=WF_2024
```

