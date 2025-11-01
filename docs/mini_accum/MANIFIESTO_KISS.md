# Mini-Accum KISS — ACCUM = acumular SATS

## Manifiesto KISS
- **North star:** NetBTC / `sats_mult` > HODL en ventanas 3–6–12m.
- **Riesgo:** MDD vs HODL igual o menor a la referencia.
- **Operativa sobria:** sin apalancamiento, sin “magia”, sin curvas a medida.
- **Gobernanza:** canarios con guardrails, A/B contra V1.0, promociones por evidencia.
- **Transparencia:** presets versionados, artefactos reproducibles, SPA/RC publicados.

## Cómo medimos éxito
- **NetBTC (SATS):** `sats_mult` vs HODL (3–6–12m).
- **Riesgo:** MDD vs HODL (magnitud y deltas).
- **Disciplina:** FPY (flips por año) para evitar sobre-trading.
- **Consistencia:** SPA / Reality Check con objetivo **p_consistent ≥ 0.60** antes de promover.

## Lo que NO es
- No hace farming, no usa leverage, no persigue USD PnL; **sólo SATS**.
- No se congela por calendario: se congela por **evidencia** (guardrails + SPA/RC).

## Regla de oro ACCUM
> **Si no supera HODL en NetBTC (a igual o menor riesgo), no se promueve.**

## Señales de rollback (en cualquier etapa)
- Forzar **DO_TRADE=0** o **DRYRUN=1**.
- Parar cron de `:07` si aplica.
- Cuarentena de logs sospechosos en `evidence/quarantine/`.

---
**Estado operativo (hoy):**
- Canary horario: **streak 7/7 GREEN** (telemetría estable).
- KPI Guard: **OK** (FPY y drift dentro de presupuesto).
- H31/H32: **OFF** y vigilado (quarantine activa).
- FREEZE semanal (lunes 00:05 UTC): **configurado**.