# One-Pager — OOS 2025H1 · KISS v1 (PROD)

**Freeze**: `KISSv1_BASE_20251010_freeze_NETBTC_4p340727`  
**Preset**: `configs/mini_accum/presets/CORE_2025.yaml`  
**Config (ganadora)**: G200 · DD15 · RB1 · H30 · BULL0 · *sin SL/TP*

## ✅ KPI OOS 2025H1 (2025-01-01 → 2025-06-30)
| Periodo | Tipo | sats_mult | ROI NetBTC | mdd_vs_hodl | flips_total | Fuente |
|:--|:--:|--:|--:|--:|--:|:--|
| 2025H1 | OOS | **1.138462** | **+13.85% (6m)** · ≈**29.9%** anual simple | **0.741494** | **2** | `reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv` |

## Estado OOS (KISS v1 · CORE_2025 · EMA21/55 · G200 · DD15 · RB1 · H30)

- **OOS_2023_REGIME**: flips_total = **76** (BUY=38, SELL=38) · artefactos `base_v0_1_20251029_2112_*__OOS_2023_REGIME.*`
- **OOS_2024_REGIME**: flips_total = **18** (BUY=9, SELL=9) · artefactos `base_v0_1_20251029_2112_*__OOS_2024_REGIME.*`
- **OOS_2025H1_REGIME**: flips_total = **9** · artefactos `base_v0_1_20251029_1535_*__OOS_2025H1_REGIME.*`

### Auditoría REGIME (consolidados del 2025‑10‑29)

| Tag              | flips_total |  netBTC | mdd_vs_hodl |    fpy | Artefactos |
|:-----------------|------------:|-------:|------------:|-------:|:-----------|
| OOS_2023_REGIME  |          76 |  1.3596 |      0.3138 |  68.204 | `base_v0_1_20251029_2112_*__OOS_2023_REGIME.*` |
| OOS_2024_REGIME  |          18 |  1.0190 |      0.5687 | 156.536 | `base_v0_1_20251029_2112_*__OOS_2024_REGIME.*` |
| OOS_2025H1_REGIME|           9 |  0.8711 |      0.9211 |  65.745 | `base_v0_1_20251029_1535_*__OOS_2025H1_REGIME.*` |

*Nota:* estos valores son de auditoría (corridas REGIME del 2025‑10‑29). La tabla “KPI OOS 2025H1” de arriba sigue el freeze del 2025‑10‑11 como referencia de PROD.

> Fuente de KPIs vs REGIME: la tabla “KPI OOS 2025H1” usa el freeze del **2025‑10‑11** (`__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv`, 2 flips). Los artefactos `__OOS_2025H1_REGIME` del **2025‑10‑29** (9 flips, netBTC≈0.871) corresponden a otra corrida. Mantenemos el freeze como referencia de PROD y listamos REGIME como auditoría.

> Nota: los `*_kpis__*.csv` llevan encabezados variables según versión. Usa `scripts/mini_accum/summarize_oos.py` para leer `sats_mult / netBTC / mdd_vs_hodl / fpy` de forma robusta, y validar que `flips_total(CSV) > 0`.

> ### Micro-barridos H31/H32 — 2025-10-29 (OOS 2025H1)
- H31: net≈0.9883 (−13.19% vs BASE=1.1385), flips=39 → FPY≈113.88 → **FAIL** (Contrato B.3 y D.7).
- H32: net≈0.9883 (−13.19% vs BASE=1.1385), flips=39 → FPY≈113.88 → **FAIL** (Contrato B.3 y D.7).
- mdd_vs_hodl: n/d en CSV candidato (no requerido para fallar: lift<+5% y fricción≫2×BASE).
**Acción:** mantener KISS v1 base (DD15/RB1/H30/G200/BULL0) en PROD; H31/H32 quedan OFF (experimento).

> *Anualización simple:* \((1.138462^2-1) ≈ 29.9\%\).

## 🧭 Contexto (WF del freeze, 1D)
| Ventana | Tipo | sats_mult | ROI NetBTC | mdd_vs_hodl | flips_total |
|:--|:--:|--:|--:|------------:|--:|
| 2022 | WF | 1.018661 | +1.87% |         n/a | 0 |
| 2023 | WF | 2.641397 | +164.14% |    0.936073 | 7 |
| 2024 | WF | 1.613240 | +61.32% |    0.768424 | 6 |

**Producto WF 2022–2024:** **4.340727**  
**Compuesto con OOS 2025H1 (6m):** **4.941751** *(indicativo)*

## 🧪 Gate & Decisión
- **NetBTC_OOS > 0** ✔︎  
- **Riesgo consistente** (mdd_vs_hodl ≈ 0.74) ✔︎  
- **Baja rotación** (flips_total = 2) ✔︎  
- **Overlays estáticos (10×30 y 12×40) → lift ≈ −20.67%, mismo/menor riesgo ⇒ quedan en experimento (OFF en PROD).

**Conclusión**: **Promover KISS v1 a `PROD_KISSv1_2025H1` sin SL/TP.**  
**Siguiente**: micro-barridos H31/H32 (RB1; RB2 solo referencia) y repetir gate.

---

## Decisión PROD 2025H1 — KISS v1
- Base: DD15/RB1/H30/G200/BULL0 (canónico)
- OOS 2025H1: sats_mult=1.138462 (+13.85% 6m), mdd_vs_hodl=0.741494, flips=2
- Overlay SL/TP 12×24: lift=+0.00%, MDD Δ=0.000000 → **FAIL gate (≥+5% requerido)**
- Acción: mantener v1 base en PROD; SL/TP queda **OFF (experimento)**.

## 2025-10-11 — KISS v1 (RB1/H30) mantiene liderazgo
- v1.1 (H29/H31/RB2): lift ≤ 0% → FAIL gate (≥+5%).
- v1.2 (SL/TP estático): sin candidatos válidos; 12×24 = +0.00% → queda OFF.
- Decisión: conservar v1 base en PROD (DD15/RB1/H30/G200/BULL0).
