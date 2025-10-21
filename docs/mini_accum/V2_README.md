# Mini-Accum V2 — Cierre y decisiones (KISS, trazable)

**Preset Año-2 post-halving:** `E1_Y2`  
**Barra:** 1D  
**Señales:** EMA 12/26 + RSI(14, 35/65) + ADX(14, min=22)  
**Régimen:** macro_sma 200 (ON)  
**Anti-whipsaw:** dwell_bars_min_between_flips = 3  
**Exit ATR:** OFF (toggle defensivo, no default)  
**Costes (backtest):** fee_bps_per_side=2, slip_bps_per_side=1

## Sanity 2022 (OOS) — E1_Y2
- **NetBTC (sats_mult):** ~**2.921**  
- **MDD vs HODL:** ~**0.1055**  
- **Flips:** **8**  
- KPI: `*kpis__OOS_2022_E1_Y2*.csv`

> Comparado con CTRL (dwell2, macro200 ON, ATR OFF):
> - NetBTC: 2.8966 → **2.9213** (**+0.85%**)  
> - MDD: **0.1172 → 0.1055** (↓ **0.0116**)  
> - Flips: **10 → 8** (↓ **2**)

## Mini-grid 2022 (resumen)
| Var | NetBTC | Lift vs CTRL | MDD | ΔMDD | flips | PASS |
|---|---:|---:|---:|---:|---:|:--:|
| **D (dwell3)** | 2.9213 | +0.85% | 0.1055 | **−0.0116** | **8** | ✅ |
| D+E (exit_atr ON) | 2.9213 | +0.85% | 0.1055 | −0.0116 | 8 | ✅ |
| D+F (macro OFF) | 2.9213 | +0.85% | 0.1055 | −0.0116 | 8 | ✅ |
| **CTRL** (dwell2) | 2.8966 | — | 0.1172 | — | 10 | ✅ |

## Sensibilidad (2022)
- **dwell4:** NetBTC 2.9288 (+0.26% vs D), MDD 0.1020 (−0.0036 vs D), flips 8 → **NO default** (mejora de MDD < **0.005** acordado).
- **adx_min=20:** igual que D (indiferente) → mantenemos **22**.

## Decisiones
- ✅ Adoptar **dwell3** como estándar del preset Año-2 post-halving (**E1_Y2**).  
- ✅ Mantener **macro_sma: 200** activada.  
- ✅ Mantener **adx_min: 22** (A20 no aporta).  
- ✅ **exit_atr: OFF** por defecto (dejar como toggle defensivo).  
- ❌ No subir a **dwell4** como default (beneficio marginal, por debajo del umbral de adopción).

## Uso operativo
- Años **Y2** (2º año post-halving): usar `configs/mini_accum/v2_0/E1_Y2.yaml` (1D).  
- Resto de años: baseline **KISS v1 TOP** (DD15·RB1·H30·G200·BULL0), 1D.  
- Si el mercado entra en régimen extra-choppy, considerar variante `E1_Y2_D4` (no default).