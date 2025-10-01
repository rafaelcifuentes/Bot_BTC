# 🚀 Plan de Despliegue Modular Mini‑Accum – Etapas 3 a 6

## ✅ ETAPA 3 — Canario 🟠 (10–20% capital)

**Objetivo:** Iniciar operación limitada en entorno real, con capital controlado.

### Acciones clave:
1. Activar `RUN_MODE=canary` con sizing limitado (10–20% del capital).
2. Reutilizar wrapper actual (`run_mini_accum_live.py`), con ejecución real y logs activos.
3. Congelar cada lunes (`FREEZE`) y correr A/B automáticamente.
4. Activar manualmente `BULL_HOLD` si macro alcista fuerte (`EMA200 D1` + `ADX ≥ 20`).
5. Aplicar **guardarraíles estrictos**:
   - ❌ Sin leverage.
   - ❌ Sin shorts.
   - ❌ Sin ajuste dinámico de tamaño (`risk_pct` fijo).

### Criterio de pase (Promoción a Etapa 4):
✅ Dos semanas consecutivas con:
- NetBTC ≥ HODL
- MDD_vs_HODL ≤ 1
- FPY ≤ 26/año (máx. 2/mes)
- 0 incidentes críticos o alertas severas
- Tracking semanal dentro de ±3% vs backtest/paper

---

## ✅ ETAPA 4 — Operación Continua 🟢

**Objetivo:** Consolidar la operación en producción con monitoreo estructurado.

### Tareas de producción:
1. Generar reporte semanal (`estado_semanal.md`) con:
   - NetBTC
   - FPY
   - MDD_vs_HODL
   - Flips actuales vs permitidos
   - Fail rate (flips fallidos)
2. Post-run A/B automático:
   - Si Δ NetBTC ≥ +0.02 sin empeorar MDD/FPY → marcar para revisión.
3. SPA / Reality Check:
   - Correr si se activa algún módulo nuevo.
   - **Criterio**: no rechazo al 5–10%.
4. Verificaciones estructurales:
   - PBO positivo
   - DSR estable y en zona verde

---

## ✅ ETAPA 5 — Rollback / Seguridad 🧯

**Objetivo:** Garantizar que se puede pausar, recuperar y continuar sin pérdida de estado.

### Checklist técnico:
- Activar `override_mode=PAUSE` en cualquier momento con seguridad.
- Backup automático de:
  - `state/mini_accum.pkl`
  - `signals/latest.json`
  - Carpeta `reports/*` completa
- Runbook de recuperación:
  - Comando de replay desde último timestamp válido
  - Validación de `hash + versión` antes de reanudar live

---

## ✅ ETAPA 6 — Documentación 📘

**Objetivo:** Garantizar reproducibilidad y trazabilidad total del sistema.

### Archivos a mantener y actualizar:
- `docs/mini_accum/BULL_HOLD.md` → reglas de activación + ejemplos visuales
- `docs/mini_accum/Progreso.md` → tabla de estado semanal
- `docs/mini_accum/Roadmap_PDCA.md` → actualizaciones tras cada `FREEZE`
  - Snapshot
  - Resultados A/B
  - SPA/RC stub

---

## 📌 Recomendaciones de Ejecución Semanal

| Día     | Acción Prioritaria                                                                 |
|---------|-------------------------------------------------------------------------------------|
| Lunes   | ✅ Ejecutar FREEZE semanal + snapshot OOS                                           |
| Lunes   | 🔄 Ejecutar A/B automático (aunque no se active ningún módulo nuevo)               |
| Martes+ | 🟡 Iniciar Canario con 10% capital real                                             |
| Durante | 🧪 Monitorear watchdog, heartbeat y outputs (`latest.json`, `live_kpis.csv`)       |
| Domingo | 📋 Actualizar `docs/Progreso.md` y hacer check visual de KPIs                      |

---

## 🟠 Canario activado = Modo vigilancia total

Este modo NO es una versión beta, es una versión de producción con capital reducido. Su misión es **verificar el tracking real vs paper sin desvíos** y con posibilidad de PAUSE inmediata.

¡Satochi Canario listo para despegar! 🕊️🌕

---