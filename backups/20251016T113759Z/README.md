# Bot_BTC Monorepo

Mini‑BOT BTC (mini_accum) en paquete aislado.
<!-- KISS-AUTO:BEGIN -->
# Bot_BTC (KISS)

## Regla estacional y costes
- Regla: año +2 post-halving ⇒ **E1_Y2**; resto de años ⇒ **V1 TOP**.  
- Costes baseline: fee = 2 bps/side, slip = 1 bps/side (2/1).  
- Presets fuente de verdad: `E1_Y2.yaml` (E1_Y2) y `CORE_2025.yaml` (V1 TOP).

## Gate y selector (DRIVE/SPORT)
- DRIVE (núcleo): `PROD_KISSv1_2023`  
- SPORT (turbo condicionado): `PROD_KISSv1_2024`  
- OFF: `PROD_KISSv1_2025H1`, `PROD_E1_Y2_2022` (aparcados hasta mejorar)
- Fichero de fricción viva: `deploy/live_fee_slip` con valores tipo 2/1, 2/2, 2/3…
- Regla de activación: si `live_fee_slip ∈ {2/1, 2/2}` ⇒ SPORT (2024); en caso contrario ⇒ DRIVE (2023)
- Operativa:
  - Plan: `make -f mk/deploy.mk -s deploy-plan`
  - Selector: `make -f mk/deploy.mk -s deploy-select`
  - Estado: `make -f mk/deploy.mk -s deploy-status`

## KPIs por periodo (freezes de producción)
| Periodo | Tag                | NetBTC  | MDD_vs_HODL |
|--------:|--------------------|--------:|------------:|
| 2022    | PROD_E1_Y2_2022    | 2.921250| 0.104540    |
| 2023    | PROD_KISSv1_2023   | 2.641397| 0.936073    |
| 2024    | PROD_KISSv1_2024   | 1.613240| 0.768424    |
| 2025H1  | PROD_KISSv1_2025H1 | 1.138462| 0.741494    |

## KPIs OOS etiquetados (origen de los números)
- 2022 · E1_Y2 · sats_mult 2.916092 · mdd_vs_hodl 0.104540 · flips 8 · CSV: `base_v0_1_20251014_1509_kpis__OOS_2022_REGIME.csv`
- 2023 · V1 TOP · sats_mult 2.641397 · CSV: `base_v0_1_20251014_1509_kpis__OOS_2023_REGIME.csv`
- 2024 · V1 TOP · sats_mult 1.613240 · CSV: `base_v0_1_20251014_1509_kpis__OOS_2024_REGIME.csv`
- 2025H1 · V1 TOP · sats_mult 1.138462 · CSV: `base_v0_1_20251014_1509_kpis__OOS_2025H1_REGIME.csv`

## Producto acumulado (base 1 BTC a 2022-01-01)
- BTC fin 2024: 12.42607251  
- BTC fin 2025 (H1): 14.14661136  
- BTC fin 2025 neutral (H2≈H1): 16.10537946

## Stress rápido (sanity de robustez)
- Headroom recomendado: `E1_S_MIN=3.0 V1_M_MAX=0.9 make -s stress-head`
- Probabilidad de mantener KPIs (agregado): `make -s stress-prob`
- Guardados:
  - Headroom: `make -s stress-head-save` → `reports/mini_accum/STRESS_HEAD.txt`
  - Probabilidades: `make -s stress-prob-save` → `reports/mini_accum/STRESS_PROB.txt`

## Trazabilidad
- Hashes de datos: `reports/mini_accum/_freezes/DATA_HASHES.md`
- Freezes por año: `reports/mini_accum/_freezes/*`
- Tags Git: `git tag -n` incluye NetBTC/MDD y hashes en el mensaje

## Runbook semanal
- Ver `docs/miniaccum/readme.md` (Checklist, KPIs a registrar, criterios de aceptación y flags).
<!-- KISS-AUTO:END -->
