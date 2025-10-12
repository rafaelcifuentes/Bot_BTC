<file name=Progreso.md path=docs/mini_accum>
>>> Atajos: [Ver histórico V1.0](Progreso_HISTORICO.md#v10--histórico-recuperado) · [Ver histórico V2.0](Progreso_HISTORICO.md#v20--histórico-recuperado)

# Progreso al 2025-09-09 — V1.0 (prudente‑xbuf25) y plan V2.0

## Resumen ejecutivo

✅ DONE
- Core v0.1 congelado y replicable.
- Sufijo de reportes automatizado + diagnóstico de cap semanal.
- Din-ATR (2/1) funcionando, sin violaciones.
- Anti-microcruces (`signals.cross_buffer_bps=25`) incorporado.
- Logging, freeze, checksums; OOS 2022–2023 corrido.

🔜 TO-DO (acción inmediata)
1) Correr OOS por ventanas del plan y tabular KPIs (2022H2 / 2023Q4 / 2024H1).
2) Ablations para bajar flips: `dwell 6→4/8` y `cross_buffer_bps 25→35`.
3) Probar `macro_persist` ligero (1–2 días > EMA200) y/o `exit.confirm_bars=2`.
4) Exponer `flips_blocked_hard` en el summary y consolidar `--suffix` en CLI.
5) Push remoto + tag.

## DONE (detalle)
- Núcleo v0.1 implementado y congelado (macro D1 EMA200, 21/55 4h, salida activa confirmada, pasiva por cruce, dwell, costes).
- Normalización de tiempo (timestamp/ts UTC, orden y deduplicación) para evitar `KeyError: 'timestamp'` y duplicados.
- Runner y artefactos:
  - CLI mini-accum-backtest operativo (start/end) + renombrado con sufijo vía REPORT_SUFFIX y/o script `rename_last_reports.py`.
  - Diagnóstico de presupuesto semanal (BUY/semana, cap por semana, violaciones = 0).
  - Presupuesto dinámico por ATR (2-verde / 1-resto): activo y validado (sin violaciones del cap).
  - Buffer de cruce anti-microcruces `signals.cross_buffer_bps` (probado 0/10/15/25; preset actual xbuf25).
  - Trazabilidad: `experiments_log.csv` + freeze de entorno (`env/requirements-YYYYMMDD.txt`) + checksums OHLC.
- OOS ejecutado 2022–2023 (sin violaciones de cap).
- Tag local `v0.1-prudente-xbuf25` creado.

## NOTAS DE DESEMPEÑO (últimos runs)
- Variantes xbuf (dinATR + dwell6): `net_btc_ratio ≈ 0.59–0.61`, `mdd_model ≈ 0.232–0.243`, `flips/año ≈ 58–69`.
- Cumple MDD vs HODL (`≈0.75–0.80 ≤ 0.85`).
- No cumple aún: `net_btc_ratio ≥ 1.05` ni `flips/año ≤ 26` (objetivos del plan).

## TODO (por prioridad)
1) OOS formal por ventanas del plan (guardar KPIs por ventana):
   - 2022H2, 2023Q4, 2024H1 → tabla con `net_btc_ratio`, `mdd_model`, `mdd_vs_hodl`, `flips/año`.
2) Reducir turnover manteniendo MDD:
   - Ablations rápidas: `dwell 4 vs 6` (actual) y `xbuf 25/35`.
   - Probar confirmación de salida más estricta (p. ej. `confirm_bars=2`) y/o `macro_persist` (N días > EMA200).
   - Enforzar hard 26/año en CLI (ya está en core sim; exponer `flips_blocked_hard` en summary).
3) Módulos opt-in (ablation con KPIs OOS):
   - ATR “pausa amarilla” (slim): debe bajar flips ≥10% o MDD ≥10% con Net_BTC_ratio ≈.
   - Grace TTL: cooldown suave tras flip; objetivo: turnover −10% con ratio ≈.
   - Hibernación por chop (≥2 cruces 21/55 en 40 barras).
4) Documentar preset “prudente-xbuf25” en el plan (snippet YAML) y dejar BASE separado.
5) Integración final del sufijo en CLI (`--suffix`) y remover duplicado de `_rename_last_reports` en el runner.
6) CI mínima (lint + test de humo) y tests de I/O/EMA/merge D1→4h.
7) Git remoto y push del tag (o crear …-r1 si re-anclas).
8) Resumen de KPIs en markdown: incluir `flips_blocked_hard` y deltas vs baseline.

## Presets
- Preset actual (prudente‑xbuf25): `dinATR (2/1)`, `dwell=6`, `cross_buffer_bps=25`, `yb=5`, `p=40`.  
  **Objetivo:** bajar aún más `flips/año` sin romper MDD; mejorar `net_btc_ratio` hacia 1.05.

---

## ¿Cómo vamos?
- **Infra/packaging & reproducibilidad:** ~85%  
  Paquete instalable, CLI funcionando, runner con sufijo (rename), logging de experimentos, freeze de entorno, comprobaciones de datos, weekly cap dinámico por ATR y cross_buffer_bps activos.
- **Core v0.1 (reglas congeladas):** ~90%  
  Macro D-1, 21/55 4h, salida activa confirmada, dwell, costes, presupuesto hard.
- **Validación cuantitativa (ablation + OOS):** ~30–40%  
  Corridas 2024–2025 y 2022–2023 hechas; falta batería OOS formal (2022H2 / 2023Q4 / 2024H1), consolidar KPIs y anotar en el log.
- **Docs/CI/tests:** ~40%  
  Plan y progreso empezados; faltan tests unitarios (cap semanal, cross buffer, integridad datos) y CI simple.

**Progreso global aproximado:** ~60% del proyecto v0.1 “prudente”.

---

## ¿Rinde? (honesto y directo)
- Con el preset prudente (dyn-ATR + dwell=6 + xbuf=25) los últimos KPIs: `net_btc_ratio ≈ 0.60–0.61` y `MDD_model ≈ 0.23` vs HODL 0.306 (**−24%** de MDD frente a HODL).
- Qué pasa: Mejoramos el drawdown (bien), pero no superamos HODL y el turnover anual sigue por encima del soft/hard (≈57–70/año vs 26 objetivo).
- **Conclusión hoy:** 1/3 de umbrales pasa (MDD ✔️), pero Net_BTC_ratio y flips/año no. Aún no es un bot “rentable vs HODL” según el criterio del plan.

## Qué falta para “candidato a promoción”
1) OOS formal (2022H2, 2023Q4, 2024H1) con el preset prudente y 1–2 variantes (p/yband, `dwell=8`, `xbuf=15/25`).
2) Reducir flips sin perder MDD: probar `grace_TTL` o endurecer `cross_buffer_bps` y/o pausa amarilla (p40±yb3→5).
3) Sanity de costes y sensibilidad (±5–10 bps) para ver robustez.
4) Tests mínimos (cap semanal ≤, microcruces, datos sin duplicados).

## V1.0 — Resumen y aprendizajes

**Objetivo / alcance (V1.0)**
- Baseline “prudente‑xbuf25” con enfoque conservador para reducir drawdown y controlar rotación.
- Reglas congeladas (core v0.1): Macro D1 EMA200; cruces 21/55 en 4h; salida activa confirmada + salida pasiva por cruce; dwell; costes; presupuesto semanal hard; `signals.cross_buffer_bps`.

**KPIs (baseline V1.0)**
- `net_btc_ratio ≈ 0.60–0.61`
- `MDD_model ≈ 0.232–0.243`
- `flips/año ≈ 58–69`
- Cumple MDD vs HODL (`≈0.75–0.80 ≤ 0.85`), pero **no** cumple `net_btc_ratio ≥ 1.05` ni `flips/año ≤ 26`.

**Decisiones cerradas (V1.0)**
- Congelado **core v0.1** y preset **prudente‑xbuf25** (dinATR 2/1, dwell=6, `cross_buffer_bps=25`, `yb=5`, `p=40`).
- Normalización de timestamps (UTC), orden y deduplicación para evitar errores de I/O.
- Runner/CLI con sufijo y trazabilidad: `experiments_log.csv`, freeze de entorno y checksums OHLC.
- OOS 2022–2023 corrido sin violaciones del cap semanal.

**Lecciones y pendientes para V2.0**
- El control de drawdown funciona, pero la **rotación** sigue alta y el **ratio vs HODL** no alcanza el umbral del plan.
- Acciones: bajar flips sin empeorar MDD (ajustar dwell/xbuf, *grace TTL*, pausa amarilla por ATR), y formalizar OOS por ventanas con KPIs tabulados.

---

# 2025-10-04 — V1.1 (SL/TP defensivo ATR) — Canario opt‑in

## V1.1 — Resumen (2025-10-05)

**Backtests de estrés**  
- **Q3-2024:** ΔROI≈0 (ambas ≈ +2.4%), MDD≈−17.24% (igual), FPY≈23.9/a (igual).  
- **Q2-2025:** ΔROI≈0 (ambas ≈ −42.4%), MDD≈−20.4% (igual), FPY≈19.0/a (igual).  
➡️ En ambas ventanas, **V1.1 no degrada** vs CORE.

**SPA / Reality Check**  
- Ago-2023, Q3-2024, Q2-2025: `p_consistent = 0.545` → **FAIL** (objetivo ≥ 0.60).  
➡️ **Promoción bloqueada** por criterio estadístico, no por guardrails.

**Guardrails canario (última corrida)**  
- **PASS**: ΔMDD=0, ΔFPY=0, ΔROI=0.  
- Cron: `crontab -l | grep canary_guardrails.sh`  
- Log: `reports/mini_accum/guardrails.log`

**KPIs resumidos (en cristiano)**

| Versión | Módulos activos principales | ROI anual (NetBTC) | Kraken < $10k | FPY estimado | MDD esperada |
|---|---|---:|:---:|---:|---|
| **v1 (baseline)** | EMAs, macro filter, anti-whipsaw TTL, rotación BTC↔USDC solo con cruce | **~18–24%** | – | **~18–22/año** | **Moderado (DD16–DD20)** |
| **v1.1** | + SL/TP defensivo (ATR14; SL=2×, TP=3×, fix_on_entry) | **Neutral vs v1** (sin mejora medida) → **usar ~18–24%** | – | **~19–24/año** *(= v1 en Q3-2024 y Q2-2025; puede subir puntualmente en crash por SELL_SLTP)* | **≈ igual a v1** → **DD17–DD21** *(ΔMDD≈0 en Q3-2024 y Q2-2025)* |

**Estado operativo**  
- Overlays **congelados** (ATR 2×3).  
- Preset **2.5×** subido como **estudio** (no productivo).  
- Tag: **`mini-accum-v1.1-canary`**.  
- README del canario al día (**solo doc**).

**Artefactos clave**  
- Base (EQ): `reports/mini_accum/*_equity__Q3_2024.csv`, `reports/mini_accum/*_equity__Q2_2025.csv`  
- Canary (EQ): `reports/mini_accum/*_equity__Q3_2024_ATR2x3.csv`, `reports/mini_accum/*_equity__Q2_2025_ATR2x3.csv`  
- SPA/RC: `reports/mini_accum/spa_*.json`

**Siguientes pasos**  
- Medir **ATR 2.5×** y/o **`reentry_ttl=8–12`** en Ago-2023 y Q3-2024; repetir SPA/RC ⇒ objetivo **`p_consistent ≥ 0.60`**.  
- Mantener canario **10–20%** con guardrails; V1.0 en **sombra** para A/B.  
- Si **guardrails OK** + **SPA/RC PASS** en ≥1 ventana ⇒ proponer promoción.

**Método:** ATR(14), `SL = 2×ATR`, `TP = 3×ATR`, `fix_on_entry=true`. (Variación a evaluar: `SL=2.5×`, `reentry_ttl=8–12`)

**Ventanas probadas:**
- 2025‑Q3 → sin activación SL/TP; Δ≈0 (neutral).
- Ago‑2023 (mini‑crash) → SL/TP activa; Δmult≈−0.0068 (PASS SLO), ΔROI_anual≈−3.41% (FAIL SLO estricto); `ΔMDD ≤ 0` y `ΔFPY` dentro de límite.
- Q3‑2024 / Q2‑2025 → re‑ejecuciones limpias con sufijos únicos, sin `NaN`.

**SLO v1.1 (criterios de aceptación)**
- Pérdida acotada: `Δmult ≥ −0.010` ✅ ; `ΔROI_anual ≥ −0.03` ❌ (−0.0341 en Ago‑2023)
- Riesgo: `ΔMDD ≤ 0` ✅ ; `ΔFPY ≤ +2` ✅
- Consistencia: **SPA/RC ≥ 0.60** — *Pendiente correr en Ago‑2023 y Q3‑2024*.

**Guardrails canario (30d):**
- `ΔMDD ≤ 0`, `ΔFPY ≤ +2`, `ΔROI_anual ≥ −4%` → si viola, **rollback** a CORE.

**Decisión:** Promover **V1.1** a **canario opt‑in**; mantener CORE intacto hasta SPA/RC ≥ 0.60 y 1–2 semanas sin violar guardrails.

## V2.0 — Estado y gaps

**Meta V2.0 (tentativa)**
- Superar HODL de forma robusta manteniendo riesgo controlado:  
  `Net_BTC_ratio ≥ 1.05`, `MDD_model ≤ 0.85 × MDD_HODL`, `flips/año ≤ 26`, **SPA/RC PASS ≥ 0.60** en multisets.

**Gaps vs V1.1**
- Rotación: aún por encima del objetivo (necesario ≤ 26/año).
- Consistencia estadística: falta SPA/RC formal en múltiples ventanas (Ago-2023, Q3-2024, etc.).
- Robustez de SL/TP: evaluar **SL=2.5×** (TP=3×) y `reentry_ttl` para recortar ΔFPY sin degradar MDD.

**Experimentos planificados**
- [ ] SPA/RC multisets (Ago‑2023, Q3‑2024) — criterio PASS ≥ 0.60.
- [ ] Afinado SL ATR **2.5×** (TP=3×) y/o `reentry_ttl=8–12` velas — meta: **ΔFPY ≤ +2/año** manteniendo **ΔMDD ≤ 0**.
- [ ] Barridos adicionales de ventanas con *chop* e inclinaciones macro mixtas.
- [ ] Seguimiento canario 30d (guardrails): `ΔMDD ≤ 0`, `ΔFPY ≤ +2`, `ΔROI_anual ≥ −4%` antes de promover a CORE.

**Notas operativas**
- Mantener CORE sin cambios hasta cumplir SPA/RC ≥ 0.60 y 1–2 semanas sin violar guardrails.
- Documentar en cada PR: rutas de equities/flips usadas, overlays congelados y changelog.

## Progreso Mini-Accum KISS
- 2025-10-05: V1.0 Shadow Certified (sin canario). Baseline de referencia congelado para evaluación de V1.1.
- 2025-10-06 v1.1 listo para A/B y canario

## 2025-10-11 — Decisión KISS v1 (RB1/H30/G200/DD15/BULL0): mantener BASE

**OOS 2025H1 — BASE canónica:**  
- sats_mult = 1.138462  (≈ +13.85% en 6m)  
- mdd_vs_hodl = 0.741494  
- flips_total = 2  

**Overlays estáticos v1.2 (10×30 y 12×40) — OOS 2025H1:**  
- sats_mult ≈ 0.903167  → **lift ≈ −20.67% vs BASE**  
- mdd_vs_hodl ≈ 0.394  (mejor riesgo), flips ≈ 5  
- Gate: **FAIL** (requerido lift ≥ +5% y MDD ≤ BASE)

**v1.1 (H29/H31 y/o RB2):** lift ≤ 0% o negativo → **FAIL**.

**Acción:** mantener **KISS v1 BASE** en PROD; overlays quedan **OFF** (experimento).  
**Regla de oro ACCUM:** promover solo si NetBTC &gt; HODL **al mismo o menor MDD**.
</file>
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
- **Acción**: mantener **RB1/H30/G200/DD15/BULL0** en PROD; overlays quedan **OFF**.
- **Regla de oro ACCUM**: promover solo si NetBTC > HODL **al mismo o menor MDD**.
# Progreso — Mini Accum (KISS)

**Fecha:** 2025-09-14  
**Ventanas OOS fijas:** H2-2024 (2024-07-01→2024-12-31), Q1-2025 (2025-01-01→2025-03-31)  
**Preset:** `configs/mini_accum/presets/CORE_2025.yaml` (datos pinned 4h/D1)

---

## Resumen técnico (último corte)

- **Baseline adoptado:** **XB fijo = 20 bps** (CORE_2025 slice).
- **KPIs (OOS):**
  - **H2-2024 / XB20:** netBTC=**0.8865**, MDD=**0.625**, FPY=**20.07**.
  - **Q1-2025 / XB20:** netBTC=**1.0291**, MDD=**0.885**, FPY=**20.75**.
- **XB adaptativo (ATR) inicial:** peor que XB20 en slice (Δnet≈−0.04) y sin mejora en FPY/MDD.
- **Barrido XB 18–23:** XB19≈1.005 (Q1) pero **OOS** peor que XB20; mantenemos **XB20**.
- **ADX 18/20/24:** invariante en este slice; decisión: **ADX=20**.

---

## Roadmap PDCA (foco: subir NetBTC sin subir FPY ni empeorar MDD)

Leyenda prob.: **Alta (≥70%)**, **Media (40–60%)**, **Baja (≤30%)**.

| Etapa / Palanca | Estado (%) | Últimos hallazgos | Prob. ↑Net sin ↑FPY/MDD | Próxima acción |
|---|---:|---|---:|---|
| **Baseline XB=20 (fijo)** | **100** | CORE_2025: net>1; OOS mantiene perfil (H2 0.8865 / Q1 1.0291) | — | Etiquetado y documentado ✅ |
| **Validación OOS (H2-2024 & Q1-2025)** | **90** | Artefactos OK, runs consistentes con datos pinned | **Alta** | Mantener como guardarraíl en cada cambio |
| **XB adaptativo (ATR)** | **60** | (quiet/yellow/loud)=20/30/40 → Δnet≈−0.04 vs XB20; OOS sin mejora | **Media-baja (~35–40%)** | Probar tiers 19/25/32 + suavizado (EWMA) |
| **Tuning `exit_margin` (30–35 bps)** | **30** | Q1 M35 generó flips; falta tabla OOS completa | **Media (~45–55%)** | Correr M30/M35 en H2/Q1 y comparar vs XB20 |
| **Barrido fino XB (18–23)** | **80** | Q1: XB19≈1.005; OOS < XB20 (H2); XB20 sigue ganando | **Baja-media (~30–40%)** | Cerrar informe y fijar 20 como pivote |
| **Guardarraíl salida ATR (k=1.5/2.0)** | **80** | No mejora en slice; FPY estable | **Baja (~20%)** | Mantener OFF por defecto; re-test con más historia |
| **Age-valve / Pausa SELL** | **70** | No movieron KPIs en slice; controlan ping-pong | **Baja-media (~30–35%)** | OFF por defecto; re-evaluar tras tuning margin |
| **ADX dinámico (percentiles)** | **10** | Aún no implementado (sólo thresholds fijos) | **Media (~50%)** | Prototipo: ADXmin = pXX(vol) por régimen |
| **Macro verde con pendiente/distancia EMA200** | **10** | Idea en backlog | **Media (~45–55%)** | Feature flag + test OOS |
| **Automatización reportes A/B** | **70** | CLI guardrail (renombre condicionado) + helpers | — | Script A/B tabular por patrón (make_run_report.sh) |

---

## Decisiones vigentes

- **XB=20** es el baseline actual (mejor netBTC y FPY↓ vs PIN, MDD no peor).
- **ADX=20** (invariancia → prioridad a simplicidad).
- **Guardarraíl ATR y Age/Pausa:** presentes en código pero **OFF** por defecto.

---

## Reglas de paro (anti-overfitting)

1. **Una palanca por commit** + etiqueta con reporte.  
2. **Adopción:** ΔnetBTC ≥ **+0.02** en slice **y** OOS no peor que BASE por **−0.01**; **FPY** dentro de **±2**; **MDD** ≤ BASE + **0.05** abs.  
3. **Dos fallos seguidos** → archivar palanca, volver a hipótesis previa.  
4. **Datos pinned** y ventanas OOS fijas para todas las corridas.  
5. **Costes constantes** durante la serie de pruebas.

---

## Changelog (últimas 24h)

- ✅ **Baseline CORE_2025 con XB=20** adoptado y etiquetado.  
- ✅ **CLI guardrail**: no renombra si `netBTC==0` o `flips==0` → artefactos limpios.  
- ✅ **Whitelist de artefactos `__DEMO_PASS`** en `.gitignore`.  
- 🔄 **XB adaptativo (ATR)**: primer prototipo **no supera** XB20; se probarán tiers 19/25/32.  
- 🔄 **`exit_margin` 30/35 bps**: corridas iniciales en Q1; falta consolidar OOS.  
- 🔍 Scripts de resumen: `showkpi`, resúmenes A/B y helpers zsh para KPIs más rápidos.

---

## Siguiente sprint (orden sugerido)

1) **Completar M30/M35 (H2 y Q1)** y A/B vs XB20.  
2) **XB adaptativo 19/25/32 con suavizado** (evitar saltos entre bandas).  
3) **Prototipo ADX percentil** por régimen de volatilidad.  
4) **Automatizar tabla A/B** en `scripts/mini_accum/make_run_report.sh`.

> **Meta del sprint:** mantener **FPY ≤ BASE±2**, **MDD ≤ BASE+0.05**, y buscar **ΔnetBTC ≥ +0.02** vs XB20.
## Checklist OOS / Walk-Forward — Candidato `DD15 • RB1 • H30 • G200 • BULL0`

**Objetivo:** hacer consistentes los sats en el tiempo y respaldar estadísticamente la elección.

### 1) Batería de validación
- **Walk-forward (no solapado):** bull / bear / rango y ventanas rodantes (entrenar→avanzar→test) con parámetros *fijos*.
- **OOS fijas:** mantener las ya definidas (H2-2024, Q1-2025) + añadir al menos 2 ventanas históricas adicionales.
- **Stress de costes:** +5 / +10 / +20 bps sobre el baseline de ejecución.
- **Barrio ±1 (robustez):** `DD {14–16}`, `RB {0–2}` (o 1–2 si 0 no aplica), `H {30–32}` con `G200 sell, BULL0`.

### 2) Métricas a registrar (por ventana)
- `sats_mult`, `USD_net`, `MDD` (y `mdd_vsHODL`), `FPY`, `flips`.
- **CAGR** (USD y sats), **mediana** e **IQR** de `sats_mult` en el conjunto de ventanas.
- **Tasa de fallos**: veces que `sats_mult < 1.0`.
- Sensibilidad a costes (Δ vs baseline por +5/+10/+20 bps).

### 3) Anti-overfitting (PBO/DSR + Reality Check/SPA)
- **PBO** (Probability of Backtest Overfitting) sobre el grid del barrio.
- **Deflated Sharpe Ratio (DSR)** para significancia bajo múltiples pruebas.
- **Reality Check / SPA** para controlar *data snooping* en la batería.

### 4) Criterios de aceptación (cualquiera que no cumpla, descarta)
- **Desempeño OOS:**  
  - Δ`sats_mult` OOS ≥ **+0.02** vs baseline **y**  
  - `MDD` OOS ≤ **baseline + 0.05** (abs) **y**  
  - `FPY` dentro de **±2** de baseline.
- **Robustez barrio:** mediana(`sats_mult`) ≥ **1.02** y tasa de fallos ≤ **25%**.
- **Costes:** con **+10 bps** extra, `sats_mult` OOS **≥ 1.00** (no erosiona por completo).
- **Estadística:** PBO **bajo** (ideal < 0.2), DSR **> 0**, y pasa Reality Check/SPA al 5%.

### 5) Entregables (para pegar en Progreso.md)
- Tabla por ventana (WF/OOS) + resumen (mediana/IQR, tasa de fallos, sensibilidad a costes).
- Informe PBO/DSR + Reality Check/SPA del grid del barrio.
- Decisión: **Adoptar / Mantener en estudio / Descartar**, con justificación breve.

## 2025-10-11 — KISS v1 (RB1/H30) mantiene liderazgo
- v1.1 (H29/H31/RB2): lift ≤ 0% → FAIL gate (≥+5%).
- v1.2 (SL/TP estático): sin candidatos válidos; 12×24 = +0.00% → queda OFF.
- Decisión: conservar v1 base en PROD (DD15/RB1/H30/G200/BULL0).

## 2025-10-11 — v1.2 (SL/TP estático) aparcado
- Faltan helpers (mk_overlay/merge_cfg) y no hubo sufijo SLTP en artefactos.
- Decisión: mantener v1 base en PROD; overlays quedan en **experimento**.

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
