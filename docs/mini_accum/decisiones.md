
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

##2025-10-26 ## # DECISIONS (ADRs)

---

## 2025-10-29 — DECISIÓN: Política de preset por ciclo (“KISS estacional”)

**Resumen:** Adoptamos una regla única, simple y trazable para orquestar presets según el año del ciclo de halving.

- **ADR-006 — Política por ciclo (Aprobado):**  
  - **Año +2 post‑halving** ⇒ preset **E1_Y2** (`configs/mini_accum/presets/E1_Y2.yaml`): EMA **12/26**, **RSI 35/65**, **ADX≥22**, **dwell=3**, **bar=1d**, **macro_sma=200 ON**, `exit_atr=OFF`.  
  - **Otros años** ⇒ preset **CORE_2025** (`configs/mini_accum/presets/CORE_2025.yaml`): **DD15 / RB1 / H30 / G200 / BULL0**, *sin SL/TP*.
  - **Rationale (evidencia):** 2022 con E1≈**2.92×**, **mdd_vs_hodl≈0.10**, ~6–8 flips; 2023–2024 con v1≈**2.64×** y **1.61×** con **mdd_vs_hodl&lt;1**; 2025H1 v1≈**1.138×** con **2 flips**.

- **ADR-007 — Overlays SL/TP (Estado: OFF en PROD):**  
  Permanecen **OFF**. Solo opt‑in en canario con **SPA/RC ≥ 0.60**, **ΔMDD ≤ 0**, **ΔFPY ≤ +2/año**, **ΔROI_anual ≥ −4%**.

- **ADR-008 — Gate de promoción (Reafirmado):**  
  Promover solo si **NetBTC&gt;HODL al mismo o menor MDD**; además, Δ`sats_mult` ≥ **+0.02** en OOS y FPY dentro de **±2** del baseline.

- **ADR-009 — Trazabilidad:**  
  Etiquetar freezes y reportes con sufijos (`OOS_${TAG}_REGIME`) y conservar **FREEZE_DAILY_YYYYMMDD** al cierre UTC.

**Acción:** Documentado en `docs/mini_accum/Progreso.md` (sección “SANTO GRIAL”) y aquí en ADRs. Sin cambios en cron ni guardarraíles.

- ADR-001: KISS v1 sin cambios de lógica durante canario.
- ADR-002: Trading live bloqueado (DO_TRADE!=1) durante canario.
- ADR-003: Criterio canario KISS = última corrida del día `ready+done`.
- ADR-004: LAB4 y selector corren en sombra (no alteran señales).
- ADR-005: Empaquetado diario con evidencia “suficiente”: REPORT.md, últimos canarios, cron.log, latest.json, status.## '"$TODAY"' — Micro-barridos H31/H32 (DWELL96) — FAIL
Base (freeze OOS 2025H1): net=1.138462, mdd_vs_hodl=0.7415, FPY≈4
H31_DWELL96: net≈0.7790, mdd≈1.0085, FPY≈19.67 → FAIL (Gate + D.7)
H32_DWELL96: net≈0.7790, mdd≈1.0085, FPY≈19.67 → FAIL (Gate + D.7)
Decisión: mantener PROD CORE_2025 (H30·RB1). No promover. Ni un satoshi cedido.
Notas: TTL/dwell respetados; warning de pandas resuelto (ffill()).
## '"$TODAY"' — Micro-barridos H31/H32 (DWELL96) — FAIL
Base (freeze OOS 2025H1): net=1.138462, mdd_vs_hodl=0.7415, FPY≈4
H31_DWELL96: net≈0.7790, mdd≈1.0085, FPY≈19.67 → FAIL (Gate + D.7)
H32_DWELL96: net≈0.7790, mdd≈1.0085, FPY≈19.67 → FAIL (Gate + D.7)
Decisión: mantener PROD CORE_2025 (H30·RB1). No promover. Ni un satoshi cedido.
Notas: TTL/dwell respetados; warning de pandas resuelto (ffill()).
