
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
