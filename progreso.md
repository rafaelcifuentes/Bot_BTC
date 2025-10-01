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
