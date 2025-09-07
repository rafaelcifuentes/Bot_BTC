# Mejoras de Alto ROI — Priorización y Detalle

> **Nota:** La mejora “Auto‑tuning de `ATR_MAX`” que propuse recién **no estaba** en la lista original. En este documento **no la incluyo** para mantener fidelidad con lo acordado. Si quieres, la añadimos luego como **A4 (param-only)**.

---

## A) Acciones sin cambiar código (solo parámetros / toggles)

| # | Idea                                        | Beneficio | Esfuerzo | Tiempo | ¿Código? | ¿Solo parámetros? | KPI de aceptación |
|---|---------------------------------------------|-----------|----------|--------|----------|-------------------|-------------------|
| 1 | Corazón slim (EMA200+ATR%)                  | Alto      | 2/10     | 2/10   | No       | Sí                | **MDD_overlay ≤ 0.85×** base y **ΔPF ≥ −0.05** |
| 2 | Grace TTL (dwell / TTL de estado)           | Medio-alto| 2/10     | 2/10   | No       | Sí                | **−20%** flips <24–48h **sin perder PF** (ΔPF ≥ −0.05) |
| 3 | Nudge por alineación (régimen ≈ ON + mom 4h)| Medio     | 3/10     | 2/10   | No       | Sí                | ↑ Net BTC con ΔMDD ≈ 0; Δturnover ≤ +5% |
| 5 | Perla con filtro EMA200*                    | Medio-alto| 3/10     | 3/10   | No*      | Sí*               | OOS PF ↑ (o estable) y turnover ↓; oos_net > 0 *(aplica cuando Perla esté activa)* |

---

## B) Requieren algo de código (pequeño a medio)

| # | Idea                                 | Beneficio | Esfuerzo | Tiempo | ¿Código? | ¿Solo parámetros? | KPI de aceptación |
|---|--------------------------------------|-----------|----------|--------|----------|-------------------|-------------------|
| 4 | Exec adaptativo 2-niveles (agresivo/pasivo por vol) | Medio | 5/10 | 5/10 | Menor–Medio | Parcial | ↓ coste/slippage **sin ↑ rejects** |
| 6 | Beta-cap suave (cap por beta a BTC)  | Medio     | 5/10     | 5/10   | Medio    | Parcial            | MDD ↓ ~10% con ΔPF ≥ −0.05 |
| 7 | Turnover budget semanal              | Medio     | 5/10     | 4/10   | Menor–Medio | Parcial         | Turnover ↓ ≥15% con Net BTC ≥ baseline |
| 8 | Perla ensemble chico (3–5 configs)   | Medio     | 6/10     | 6/10   | Medio    | Parcial            | ↓ var PnL y MDD, PF ≈ estable |
| 9 | Funding tilt en extremos             | Variable  | 6/10     | 6/10   | Medio    | Parcial            | Mejora en clusters extremos sin dañar régimen normal |

---

## C) Ideas más ambiciosas (planear, no urgentes)
- Corazón **advanced** (LQ + correlación + guardarraíles) como perfil macro v0.2.
- Cerebro/Allocator v0.2 (vol targeting, Kelly-cap, correlación activa).
- Meta-pesos por régimen (pequeño meta-modelo) tras recolectar logs.

---

## D) Proceso / Claridad (rápidos de gobernanza)
- Runner semanal único (✅ listo: `run_heart_slim_pipeline.sh`).
- Decision log + snapshot tras cada freeze (✅).
- Política “no tocar selected salvo gate + no-regresión” (vigente).

---

## Tabla consolidada de la “última lista” (#1–#9, P1)

| #  | Idea                                                    | Beneficio  | Esfuerzo | Tiempo | ¿Código?     | ¿Solo parámetros? | KPI de aceptación |
|----|---------------------------------------------------------|------------|----------|--------|--------------|-------------------|-------------------|
| 1  | Corazón slim (EMA200+ATR%)                              | Alto       | 2/10     | 2/10   | No           | Sí                | MDD_overlay ≤ 0.85× base y ΔPF ≥ −0.05 |
| 2  | Grace TTL (dwell/TTL de estado)                         | Medio-alto | 2/10     | 2/10   | No           | Sí                | −20% flips <24–48h sin perder PF |
| 3  | Nudge por alineación (sesgo cuando régimen=señal)       | Medio      | 3/10     | 2/10   | No           | Sí                | ↑Net BTC con ΔMDD ≈ 0; Δturnover ≤ +5% |
| 4  | Exec adaptativo 2-niveles (agresivo/pasivo por vol)     | Medio      | 5/10     | 5/10   | Menor–Medio  | Parcial           | ↓ coste/slippage sin ↑ rejects |
| 5  | Perla con filtro EMA200                                 | Medio-alto | 3/10     | 3/10   | No*          | Sí*               | oos_pf ↑ (o estable) y turnover ↓; oos_net>0 |
| 6  | Beta-cap suave (cap por beta a BTC)                     | Medio      | 5/10     | 5/10   | Medio        | Parcial           | MDD ↓ ~10% con ΔPF ≥ −0.05 |
| 7  | Turnover budget semanal                                 | Medio      | 5/10     | 4/10   | Menor–Medio  | Parcial           | Turnover ↓ ≥15% con Net BTC ≥ baseline |
| 8  | Perla ensemble chico (3–5 configs)                      | Medio      | 6/10     | 6/10   | Medio        | Parcial           | Vol PnL ↓ y MDD ↓ con PF ≈ estable |
| 9  | Funding tilt en extremos                                | Variable   | 6/10     | 6/10   | Medio        | Parcial           | Mejora en clusters extremos sin dañar régimen normal |
| P1 | Mini-BOT Acumulación (BTC/Stable, sin short) (hilo aparte) | Medio-alto | 4/10  | 4/10   | Medio        | —                 | — |

---

### Notas
- **Beneficio:** Alto / Medio / Bajo / Variable
- **Esfuerzo / Tiempo (1–10):** menor es más fácil/rápido
- **¿Código?:** No (solo param), Menor (toques chicos), Medio (funciones pequeñas), Alto (módulos nuevos)
- **KPI de aceptación (ejemplos):** mantener PF (ΔPF ≥ −0.05), bajar MDD (≤ −10/15%), bajar turnover, etc.

