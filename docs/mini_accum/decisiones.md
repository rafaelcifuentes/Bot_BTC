
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
