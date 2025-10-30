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

# Stages
- Shadow (DRYRUN) — actual
- Pilot Live (DO_TRADE=1 con size min, sólo tras 5/5 sucesivos y gates OK)
- Go-Live (post-pilot, sin cambios de lógica)

## Gates de promoción
- ≥5/5 días GREEN (criterio KISS), sin alertas duras.
- MDD no empeora vs baseline; flips↓; NetBTC≈.

## 📈 Estado de avance — corte UTC 2025-10-28 (Bogotá 2025-10-28)

> Esta sección se **añade** sin borrar nada del contenido previo.

### 🔎 Resumen operativo
- **Canario GREEN** validado hoy: `.../evidence/dayN_2025-10-28/canary_live.20251028T180700Z.log` incluye `ready (signal fresh)` + `done` (sin errores).
- **Gate** `gates_pilot_live`: detector GREEN robusto (acepta `→ **GREEN**` y patrón nuevo `ready+done`). **ATTEST** temporalmente en *warning* (no bloquea), manteniendo KISS.
- **Reporte diario** `bb_dailyreport.zsh`: `- ATTEST OK: 1` (cálculo robusto, fallback a `health/mini_accum.status`).
- **Freeze** etiquetado: `FREEZE_DAILY_20251028` publicado.
- **Soak Test (Shadow/DRYRUN)**: reiniciado el 2025-10-27. **Día 1** contado = 2025-10-28 (faltan 6 para 7/7).
- **Paquetes diarios**: `artifacts/canary_pack_20251028.tgz` generado.

### ✅ Done (ajustado a Etapas 3–6 donde aplica)
- **Shadow (pre-Etapa 3)** estable: canario horario DRYRUN con locks (rate-limit + process) activos.
- **Gates**: GREEN robusto; ATTEST calculado y **reportado** (modo informativo).
- **Evidencia**: `REPORT.md` diario + empaquetado por día.
- **Trazabilidad**: freeze diario etiquetado en remoto.

### 🧭 ToDo inmediato (próximas 72 h)
- **7/7 GREEN**: mantener racha con **ATTEST OK** diario (≥1 log GREEN/día y reporte emitido).
- **Docs**: incorporar el blueprint EN actualizado a `docs/` y enlazarlo desde `README`.
- **Selector de régimen por ciclo**: añadir `scripts/mini_accum/dev/run_regime_year.sh` y preset `configs/mini_accum/presets/E1_Y2.yaml` (Y+2 ⇒ E1; resto ⇒ v1 TOP) para reproducibilidad.
- **Pilot readiness (Etapa 3)**: preparar `RUN_MODE=canary` con **size real 10–20%** y **PAUSE** instantáneo (activar sólo tras 7/7).
- **Backups / rollback (Etapa 5)**: validar backup automático de `state/`, `signals/latest.json`, `reports/*`; documentar comando de **replay** en el runbook.
- **Monitoreo semanal (Etapas 4/6)**: crear `docs/mini_accum/Progreso.md` (tabla semanal) y `estado_semanal.md` (KPIs).

### ⚠️ Riesgos / observaciones
- Cambios de formato en logs: el gate cubre `ready+done`; si cambia el prefijo, habrá que ajustar el detector.
- ATTEST en *warning*: reactivar modo “hard” cuando estabilicemos 7/7 sin falsos negativos.
- Cron a `:07`: revisar colisiones tras cambios (locks están activos, pero conviene mirar).

<!-- 2025-10-30 — Estado y siguientes pasos (KISS v1) -->
## 2025-10-30 — Estado del despliegue y próximos pasos (KISS v1)

**Resumen de hoy**
- Presets canónicos: **CORE_2025** (DD15/RB1/H30/G200/BULL0) y **E1_Y2** (ADX≥22, EMA12/26+RSI, 1D) fijados.
- **KPI Guard**: OK (FPY≈10.81, drift≈1.00%) con `launchd` activo.
- **Canario**: 7/7 días **GREEN** (smoke `scripts/mini_accum/smoke_canary.zsh`).
- **H31/H32**: artefactos en **cuarentena**; checks confirman **OFF**.

**Etapas de despliegue (estado)**
1) **Dev/Shadow** → ✅ Completo (reglas CORE v1, datos normalizados, renombrado de artefactos, evidencias).
2) **Canario** → ✅ Estable (7/7 GREEN), guardarraíl activo, KPI Guard en verde.
3) **Prod BASE** → ✅ CORE_2025 sin overlays; SL/TP queda en experimento.
4) **Régimen** → ⏳ Por correr:  
   - 2022 ⇒ `E1_Y2`  
   - 2023/2024/2025H1 ⇒ `CORE_2025`
5) **Governance** → 🟨 En curso: FREEZE semanal (lunes), cuarentena H31/H32, evidencia diaria.

**DoD por etapa (extracto)**
- *Canario (DoD)*: ≥5/7 GREEN + KPI Guard OK + sin violaciones de D.7 (fricción) → **CUMPLIDO**.
- *Régimen (DoD)*: runners por periodo con tabla OOS/WF en Progreso.md; fuentes (paths) citadas.

**Acciones siguientes**
- Ejecutar runners por régimen y **tabular KPIs** (sats_mult, mdd_vs_hodl, FPY, flips).
- Mantener **H31/H32 OFF** (check + cuarentena) y **FREEZE semanal** (snapshots de blobs y HEAD).
- Actualizar **Progreso.md** con la tabla OOS/WF y enlaces a artefactos.

> Nota: No se promueven overlays que **no** aporten lift ≥ +5% **y** respeten MDD/FPY (Gate + D.7). “Ni un satoshi cedido”.

<!-- 2025-10-30 — Etapas A→E snapshot + B4 sombra -->
## 2025-10-30 — Siguientes etapas (plan a rajatabla) — Snapshot de estado

**Etapa A — Canario DRYRUN (actual)**
- Objetivo: estabilidad + telemetría real con riesgo cero.
- Estado: ✅ 7/7 días GREEN; KPI Guard OK; evidencia diaria empaquetada.

**Etapa B — Pilot Live “armado, sin ordenar”**
- Config: DO_TRADE=1 DRYRUN=0, pero el ejecutor imprime *(armed) crear orden aquí* (no envía).
- Duración: 1–2 días.
- Gates: `gates_pilot_live OK`, **B4: sin storm (1/h)**, evidencia empaquetada diaria.
- **B4 en sombra:** ✅ PASS. Validado con canario DRYRUN y `scripts/mini_accum/check_storm.zsh` (≤1/h en 24 h).

**Etapa C — Pilot Live Testnet**
- Config: BINANCE_TESTNET=1; 1–2 ejecuciones manuales ~$10 sim.
- Gates: logs con `placed … status=closed … filled=…` (solo testnet); sin errores CCXT/red.
- Estado: ⏳ pendiente.

**Etapa D — Pilot Live Mainnet micro (capas duras)**
- Config: DO_TRADE=1 DRYRUN=0 USD_MAX=10 CAP=0.10; kill-switch listo; máx 1 orden/día.
- Gates: 100% órdenes micro exitosas o abortadas correctamente; sin duplicados, sin storm, sin slippage raro.
- Estado: ⏳ pendiente.

**Etapa E — Producción limitada**
- Escalonar tamaño/frecuencia con capas, manteniendo el canario DRYRUN como sentinela.
- Estado: ⏳ pendiente.

**Señales de rollback (cualquier etapa)**
- DO_TRADE=0 o DRYRUN=1.
- Parar cron de :07.
- Cuarentena de logs a `evidence/quarantine/`.

**Checklist de pase A → B (compacta)**
- [x] 7/7 días GREEN (1/h a :07, sin tormentas)  
- [x] 7/7 ATTEST OK (≥1/día)  
- [x] `gates_pilot_live` OK (sin errores tras write_status)  
- [x] `REPORT.md` diario presente  
- [x] `canary_pack` diario no vacío  
- [x] Cero DRYRUN=0 con `placed/filled` en mainnet

> Cuando todo está ✅, se autoriza pasar a **Etapa B**. “Ni un satoshi cedido.”
