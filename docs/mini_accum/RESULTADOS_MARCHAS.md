# Resultados por Marcha (2022–2025)

> Base 1 BTC. M1/M2/M3 según semáforo. Artefactos y fuentes al final.

## Tabla consolidada

| Período | Semáforo        | Marcha | Preset/Modo                 | sats_mult (gross) | sats_mult (net)* | mdd_vs_hodl | flips | Fuente KPI |
|---|---|---|---|---:|---:|---:|---:|---|
| 2022    | Bear/Shock      | M1     | `E1_Y2` (1D)                | 2.921250 | —     | 0.104540 | 8 | `reports/mini_accum/base_v0_1_20251013_0231_kpis__OOS_2022_E1.csv` |
| 2023    | Rango/Neutral   | M2     | `CORE_2025`                 | 2.641397 | —     | 0.936073 | 7 | `reports/mini_accum/base_v0_1_20251014_1509_kpis__OOS_2023_REGIME.csv` |
| 2024    | Rango/Neutral   | M2     | `CORE_2025`                 | 1.613240 | —     | 0.768424 | 6 | `reports/mini_accum/base_v0_1_20251014_1509_kpis__OOS_2024_REGIME.csv` |
| 2025 H1 | Bull claro      | M3     | `bull_hold_ext` (freeze)    | **1.355449** | **~1.293214**† | 0.360052 | 0 | `reports/mini_accum/base_20251101_060922_kpis__OOS_2025H1_bullhold_ext.csv` |
| 2025 H2 | Rango (plano)   | M2     | `CORE_2025`                 | 1.027582 | —     | 0.580359 | 6 | `reports/mini_accum/base_v0_1_20251102_0227_kpis__OOS_2025H2_core.csv` |

\* **net** aplica solo cuando se descuenta carry explícito (p.ej., borrow APR).  
† net ≈ 1.293214 para ~180 días con borrow_apr=0.10 y funding=0.00.

## Notas
- El **OOS 2025H1 baseline (contrato)** es 1.138462 (TOP CORE), mantenido para reproducibilidad del Santo Grial.
- M3 freeze H1-2025 es **inmutable** y se usa solo cuando el semáforo está en bull (D1>EMA200 & ADX≥20).
- H2-2025: overlay M3 **FAIL** net (<1.05), por eso **M2/M1** activos.

## Artefactos relevantes
- M3 freeze (H1-2025):  
  `reports/mini_accum/base_20251101_060922_{equity,kpis}__OOS_2025H1_bullhold_ext.csv`  
  `reports/mini_accum/_freezes/M3_2025H1_bullhold_ext_20251101_133227/`
- A/B H2-2025 (M3 candidato):  
  `reports/mini_accum/base_v0_1_20251102_0148_{equity,kpis}__OOS_2025H2_m3_try.csv`
- CORE H2-2025 (M2):  
  `reports/mini_accum/base_v0_1_20251102_0227_kpis__OOS_2025H2_core.csv`

## Referencias
- **README_MARCHAS.md** — reglas del semáforo y uso de marchas.  
- **PRESETS.md** — runners, overlays y cómo correr A/B.
### Estado M3 en H2-2025
- **Overlay “puro” (sin leverage, 0 flips del CORE)**: `net=1.02835`, `bull_pct=1.00` → **FAIL** (gate net ≥ 1.05 no alcanzado).
- **Acción operativa**: **M2 (CORE_2025)** como marcha por defecto en H2-2025; re-test semanal automatizado (`m3_weekly_gate`) y promoción de M3 solo si pasa **net ≥ 1.05** con **bull_pct ≥ 0.90**.
- **Freeze M3 H1-2025**: se mantiene como referencia canónica (no se re-escribe).

## Cierre H2-2025 (M3 vs M2) — 2025-11-03T17:47:46Z
- **M3 (bull-hold puro)**: net=**1.0283506**, bull_pct=**1.00** → **FAIL** (gate net ≥ 1.05 no alcanzado).
  - Equity: `reports/mini_accum/base_v0_1_20251103_1744_equity__OOS_2025H2_m3_puro.csv`
- **M2 (CORE_2025)**: sats_mult=**1.0275823**, mdd_vs_hodl=**0.580359**, flips=**6** (ref KPI: `reports/mini_accum/base_v0_1_20251102_0227_kpis__OOS_2025H2_core.csv`)
- **Decisión**: mantener **M2** como marcha activa en H2-2025. Programar re-chequeo semanal de M3.
