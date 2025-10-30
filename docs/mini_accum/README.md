# Mini‑Accum — README

Versión actual: **KISS v1.1 (pre‑canario)**  
Objetivo: Acumulación de BTC con máxima simplicidad, robustez y control de riesgo.

---

## 📌 Filosofía

El bot sigue los principios KISS (“Keep It Simple, Satoshi”):

- Reglas simples, transparentes y auditables.
- Evaluación estricta de cada mejora.
- Preferencia por comportamiento robusto a performance marginal.

---

## ⚙️ Núcleo congelado

El core v1.1 está **congelado** y pasa semanalmente por validaciones:

- ❄️ FREEZE semanal (OOS 2020–2025)
- ✅ Evaluación de KPIs (NetBTC, MDD, FPY, etc.)
- 🔍 A/B testing con módulos experimentales
- 🧪 Tests de robustez (SPA / Reality Check, DSR)

---

## 🧪 Etapas de despliegue

| Etapa | Descripción | Estado |
|-------|-------------|--------|
| 1     | ✅ Validación backtest & walk‑forward OOS (2020–2025) | Hecho |
| 2     | ✅ Paper Trading con gobernanza estricta semanal | Hecho |
| 3     | ✅ Canary **sombra** (DRYRUN=1) con keep‑alive/attest/pack — **5/5 logrado 2025‑10‑25** | Hecho |
| 4     | 🟠 Pilot Live controlado (sesión manual, cap≤USD 10, DO_TRADE=1; cron sigue en DRYRUN=1) | Próximo |
| 5     | 🔵 Operación continua (post‑pilot), monitoreo, A/B y gobernanza | Planificado |
| 6     | 📚 Documentación final + reproducibilidad total | En curso |

> **Nota:** El paso 4 está descrito con detalle en **[Cómo promover a Pilot Live](#pilot-live)**. Revisa esos *gates* antes de cualquier sesión manual.


---

## 🔒 Módulos Opt‑In (OFF por defecto)

Los módulos listados como opt-in en esta tabla son componentes que **alteran el comportamiento operativo** del bot.  
Su activación está sujeta a validación A/B y criterios de adopción.

| Módulo                    | Objetivo principal                                               | Estado         | Versión             | Impacto estimado |
|---------------------------|------------------------------------------------------------------|----------------|---------------------|------------------|
| `xb_adaptive`             | Cross‑buffer adaptativo con bandas de ATR                       | 🧪 ON (A/B)    | v1.1+               | +2% a +8%        |
| `exit_atr_guardrail`      | Bloqueo de salidas si rebote está dentro de ATR                 | 💤 OFF         | v1.1+               | +1% a +5%        |
| `pause_after_flip_sell`   | Espera tras giro (flip) para evitar reentradas impulsivas        | 💤 OFF         | v1.1+               | −1% a +2%        |
| `age_valve_exit`          | TTL para salidas (esperar N velas para confirmar)               | 💤 OFF         | v1.1+               | +1% a +3%        |
| `adx_macro_dynamic`       | Umbral ADX/EMA/pendiente dinámico (por percentiles)             | 💤 OFF         | v1.2 (cola)         | +1% a +4%        |
| `hibernation_on_chop`     | Evita operar en rangos (ADX o slope bajo)                       | 💤 OFF         | v2                  | +2% a +5%        |
| `bull_hold`               | Mantener BUY en macro fuerte (EMA200 D1 + ADX ≥ 20)             | 💤 OFF         | v2                  | +3% a +7%        |
| `pullback_entry`          | Entrada alternativa en retroceso (ej. bollinger/ema)            | 💤 OFF         | v3                  | +5% a +12%       |
| `trailing_exit_bull`      | Salida trailing dinámica en bull markets                        | 💤 OFF         | v3                  | +3% a +10%       |
| `risk_sizing_by_score`    | Ajustar posición según score o riesgo estimado                  | 💤 OFF         | v4+                 | +3% a +8%        |
| `sl_tp_defensivo`         | Stop Loss defensivo basado en ATR para limitar drawdowns severos | 💤 OFF (prioridad post‑canario) | v1.1.1 | +4% a +12%        |

*`sl_tp_defensivo` es el **primer candidato** a validar tras el canario (A/B en sombra, luego gate a producción). No se activa en canario ni en Pilot Live por defecto.*

---

<a id="pilot-live"></a>
## 🚀 Cómo promover a Pilot Live (KISS, manual)

> **Resumen:** Pilot Live es una sesión manual y acotada con **capital mínimo** y **controles explícitos**. No modifica el `crontab` (que sigue en DRYRUN=1). Solo procede cuando los **gates** están en verde.

### ✅ Requisitos previos (gates)

- **Streak 5/5** de canarios **DRYRUN=1 GREEN** en los últimos ≤7 días  
  _Comando de verificación_: `bb_streak_canary_kiss 5` → debe imprimir `5/5`.
- **ATTEST del día = OK** (sin FAIL) antes de promover  
  _Comando_: `bb_today` → `Hoy: attest_ok=1 | canary_dryrun_ok=1 → GREEN`.
- **Salud vigente**: `health/mini_accum.status` existe y reciente (age_h &lt; 2h, health=ok).
- **Evidencia diaria**: existe `evidence/dayN_YYYY-MM-DD/REPORT.md` del día.
- **Paquete diario**: existe y no vacío `artifacts/canary_pack_YYYYMMDD.tgz`  
  _Comando_: `tar -tzf artifacts/canary_pack_YYYYMMDD.tgz | sed -n '1,20p'`.
- **Sin DRYRUN=0** fuera de sesiones controladas: no hay `canary_live.* DRYRUN=0` en `logs/` (o están en `evidence/quarantine/`).
- **Code-seal OK**: sin `CODE SEAL MISMATCH` ni errores de watchdog en `logs/cron.log`.

> Si cualquiera de los gates falla, no promover. Revisión primero.

### 🧭 Procedimiento (30–60 min, manual)

1) **Congelar código**
   - `git status` limpio, `git rev-parse --short HEAD` anotado
   - (Opcional) tag: `git tag -a pilot_YYYYMMDD -m "Pilot Live (KISS v1.1)"`

2) **Revisar cron (sigue en sombra)**
   - Verifica que las entradas de canario sigan con `DRYRUN=1` y `EXCHANGE=binance` (global).
   - No cambies el cron para Pilot Live.

3) **Preparar entorno del Pilot**
   - Asegura API keys/permiso de trading en el exchange elegido.
   - Define parámetros mínimos:
     - `EXCHANGE=binance` (o `binanceus` según tu cuenta)
     - `DRYRUN=0`  ·  `DO_TRADE=1`  ·  `USD<=10`  ·  `cap=0.10`
   - ⚠️ **Nota**: `bb_day.zsh` tiene *guardias de sombra* y aborta con `DRYRUN!=1`. Para Pilot ejecuta **directamente** el entrypoint de `canary_live` (el mismo que genera `logs/canary_live.YYYY...log`), evitando `bb_day.zsh`.

   **Ejemplo ilustrativo** (ajusta al entrypoint real de tu repo):
   ```bash
   # Sesión manual (Pilot Live) — capital mínimo
   EXCHANGE=binance DRYRUN=0 DO_TRADE=1 USD<=10 cap=0.10 \
     python -m mini_accum.canary_live 2>&1 | tee -a logs/canary_live.$(date -u +%Y%m%dT%H%M%SZ).log
   ```

4) **Supervisión en tiempo real**
   - `tail -f logs/canary_live.*.log`
   - Señales sanas esperadas en el log:
     - `ready (signal fresh)`
     - (si aplica) trazas de orden **LIVE** con `DO_TRADE=1`
     - `canary_live: done`
   - Si aparece un WARN/ERROR o `CODE SEAL MISMATCH`, aborta y vuelve a sombra.

5) **Cierre y evidencia**
   - Restablece variables: `DO_TRADE=0` y `DRYRUN=1`.
   - `./scripts/mini_accum/bb_dailyreport.zsh` → genera `REPORT.md`.
   - `./scripts/mini_accum/pack_canary.zsh` → genera `artifacts/canary_pack_YYYYMMDD.tgz`.
   - Registra el resultado en tu bitácora (streak, ATTEST, hash git, enlace al paquete).

### ♻️ Rollback y recuperación controlada
   - No cambiaste el cron (sigue en sombra). Simplemente no repitas Pilot Live.
   - Si hubo DRYRUN=0, mueve el log a `evidence/quarantine/` y documenta el incidente.
   - Abre un ADR con la causa y el plan correctivo.

---

**Preguntas frecuentes**

- **¿Pilot Live cambia la lógica?** No. Es una **sesión manual** sobre el core congelado.
- **¿Hay que editar el cron?** No. El cron permanece con `DRYRUN=1` para el canario en sombra.
- **¿Cuenta como 6/5?** No. La condición para promoción permanente es **5/5** (sombra) + **gates OK**; Pilot Live es un *ensayo controlado* adicional.

Canonical doc (EN): docs/mini_accum/Mini-Accum_KISS_Executive_Summary_and_Reconstruction_Blueprint_EN.md

Orchestration rule:
- Year +2 post-halving (2014, 2018, 2022, 2026…): preset E1_Y2 (1D: EMA12/26, RSI 35/65, ADX≥22, Macro 200D ON, dwell=3).
- Other years: preset KISS v1 TOP (1D: EMA21/55 + ADX + Macro 200D; DD15 • RB1 • H30 • G200 • BULL0).

Philosophy: long/flat, no leverage; hard gate = GREEN; ATTEST = informational.