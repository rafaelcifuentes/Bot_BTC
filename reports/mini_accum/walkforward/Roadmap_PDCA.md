# KISS v1 — Walk-Forward & Robustez (autogenerado)

**Candidato base:** `DD15_RB1_H30_G200_BULL0`  · **Decisión:** ✅

## Resumen por ventana (mejor por sats_mult)
| window    | config_id               |   sats_mult |   mdd_vs_hodl |     fpy |   flips_total | run_ok   |
|:----------|:------------------------|------------:|--------------:|--------:|--------------:|:---------|
| WF_2023   | DD15_RB1_H30_G200_BULL0 |     2.8526  |      0.922658 | 7.01923 |             7 | True     |
| WF_2024   | DD15_RB1_H30_G200_BULL0 |     2.96861 |      0.773794 | 6       |             6 | True     |
| WF_2025H1 | DD14_RB1_H30_G200_BULL0 |     1.34835 |      0.732803 | 8.11111 |             4 | True     |

## Criterios de aceptación
- Mediana(sats_mult): **2.8526**
- Fail rate (sats<1): **0.00%**
- Δ vs vecindario ≥ min: **0.17437923344299255**
- MDD_vsHODL no peor que vecindario: **True**
- FPY no peor que vecindario: **True**

## Nota rápida 2021–2022
En bull markets largos, SMA200 (gate de venta) puede mantener posición y dar `run_ok=False`.

## Próximos pasos
- Confirmar stress de costes y tests anti-overfitting (PBO/CSCV, DSR, Reality/SPA).
- Si pasa: congelar baseline y versionar `configs/mini_accum/kiss_v1.yaml`.

## Stress de costes (mediana sats_mult por config)
| config_id               |     -20 |     -10 |      -5 |       0 |       5 |      10 |      20 |
|:------------------------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| DD14_RB1_H30_G200_BULL0 | 2.63159 | 2.58994 | 2.56933 | 2.54887 | 2.52855 | 2.50837 | 2.46844 |
| DD14_RB1_H31_G200_BULL0 | 2.63159 | 2.58994 | 2.56933 | 2.54887 | 2.52855 | 2.50837 | 2.46844 |
| DD14_RB1_H32_G200_BULL0 | 2.63159 | 2.58994 | 2.56933 | 2.54887 | 2.52855 | 2.50837 | 2.46844 |
| DD14_RB2_H30_G200_BULL0 | 2.5299  | 2.48986 | 2.47005 | 2.45038 | 2.43084 | 2.41144 | 2.37305 |
| DD14_RB2_H31_G200_BULL0 | 2.5299  | 2.48986 | 2.47005 | 2.45038 | 2.43084 | 2.41144 | 2.37305 |
| DD14_RB2_H32_G200_BULL0 | 2.5299  | 2.48986 | 2.47005 | 2.45038 | 2.43084 | 2.41144 | 2.37305 |
| DD15_RB1_H30_G200_BULL0 | 2.93344 | 2.89278 | 2.87263 | 2.8526  | 2.83269 | 2.81291 | 2.77368 |
| DD15_RB1_H31_G200_BULL0 | 2.93344 | 2.89278 | 2.87263 | 2.8526  | 2.83269 | 2.81291 | 2.77368 |
| DD15_RB1_H32_G200_BULL0 | 2.93344 | 2.89278 | 2.87263 | 2.8526  | 2.83269 | 2.81291 | 2.77368 |
| DD15_RB2_H30_G200_BULL0 | 2.82008 | 2.781   | 2.76163 | 2.74237 | 2.72323 | 2.70421 | 2.6665  |
| DD15_RB2_H31_G200_BULL0 | 2.82008 | 2.781   | 2.76163 | 2.74237 | 2.72323 | 2.70421 | 2.6665  |
| DD15_RB2_H32_G200_BULL0 | 2.82008 | 2.781   | 2.76163 | 2.74237 | 2.72323 | 2.70421 | 2.6665  |
| DD16_RB1_H30_G200_BULL0 | 2.86483 | 2.82512 | 2.80544 | 2.78588 | 2.76644 | 2.74711 | 2.7088  |
| DD16_RB1_H31_G200_BULL0 | 2.86483 | 2.82512 | 2.80544 | 2.78588 | 2.76644 | 2.74711 | 2.7088  |
| DD16_RB1_H32_G200_BULL0 | 2.86483 | 2.82512 | 2.80544 | 2.78588 | 2.76644 | 2.74711 | 2.7088  |
| DD16_RB2_H30_G200_BULL0 | 2.75412 | 2.71595 | 2.69703 | 2.67822 | 2.65953 | 2.64095 | 2.60413 |
| DD16_RB2_H31_G200_BULL0 | 2.75412 | 2.71595 | 2.69703 | 2.67822 | 2.65953 | 2.64095 | 2.60413 |
| DD16_RB2_H32_G200_BULL0 | 2.75412 | 2.71595 | 2.69703 | 2.67822 | 2.65953 | 2.64095 | 2.60413 |

**Correlación de ranking (Spearman) vs Δbps=0**
|   delta_bps_side |   spearman_corr |
|-----------------:|----------------:|
|              -20 |               1 |
|              -10 |               1 |
|               -5 |               1 |
|                0 |               1 |
|                5 |               1 |
|               10 |               1 |
|               20 |               1 |


## Tests anti-overfitting
- **PBO/CSCV**: p̂ = 0.311
- **DSR**: N/A (DSR N/A (faltan columnas))
- **Reality/SPA**: N/A (Datos insuficientes)

**Criterios orientativos de significancia**
- PBO ≤ 0.20; DSR>0 significativo; Reality Check/SPA no rechazan al 5–10%.

**Candidato:** `DD15_RB1_H30_G200_BULL0`
## Baseline & Lock-in
- Versión: `KISSv1_BASE_20250915_1309_provisional`
- Candidato: `DD15_RB1_H30_G200_BULL0`
- Estado: **Provisional** (quitar cuando DSR>0 y PBO ≤ 0.25, ideal ≤ 0.20, y 2 semanas OOS sin regresión).
- Copia YAML: `configs/mini_accum/kiss_v1_BASE_20250915_1309.yaml`
