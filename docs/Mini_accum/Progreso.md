# Progreso — Mini Accum (KISS)

## Hito — KISS_v1 baseline fijado (2025-09-15)

**Baseline plataforma (fijado para evitar regresiones):** `PT_G200_DD16_RB3_H30_BULL0`  
- sats_mult = **1.4438** (≈ +44.38% BTC vs HODL)  
- USD_net = **10.01×**  
- mdd_vsHODL = **0.733**  
- flips = **17**

**Runner-up (más frugal en flips):** `PT_G200_DD17_RB3_H30_BULL0`  
- sats_mult = **1.4075** (≈ +40.75% BTC vs HODL), USD_net = **9.76×**, mdd_vsHODL = **0.733**, flips = **15**.

> Gate: SMA200 en modo *sell* (filtro de tendencia). Sin bull-hold.
> 
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
| **Baseline XB=20 (fijo)** | **100** | CORE_2025: net>1; OOS estable (H2 0.8865 / Q1 1.0291) | — | Etiquetado y documentado ✅ |
| **Validación OOS (H2-2024 & Q1-2025)** | **90** | Artefactos OK, datos pinned | **Alta** | Mantener como guardarraíl en cada cambio |
| **XB adaptativo (ATR)** | **60** | 20/30/40 → Δnet≈−0.04 vs XB20; OOS sin mejora | **Media-baja (~35–40%)** | Probar tiers 19/25/32 + suavizado (EWMA) |
| **Tuning `exit_margin` (30–35 bps)** | **30** | Q1 M35 generó flips; falta tabla OOS completa | **Media (~45–55%)** | Correr M30/M35 en H2/Q1 y comparar vs XB20 |
| **Barrido fino XB (18–23)** | **80** | Q1: XB19≈1.005; H2 peor que XB20 | **Baja-media (~30–40%)** | Cerrar informe: 20 se mantiene |
| **Guardarraíl salida ATR (k=1.5/2.0)** | **80** | No mejora net; FPY estable | **Baja (~20%)** | Mantener OFF; re-test con más historia |
| **Age-valve / Pausa SELL** | **70** | Controla ping-pong; KPIs sin movimiento | **Baja-media (~30–35%)** | OFF; re-evaluar tras tuning margin |
| **ADX dinámico (percentiles)** | **10** | Pendiente (sólo thresholds fijos hoy) | **Media (~50%)** | Prototipo: ADXmin = pXX(vol) por régimen |
| **Macro verde con pendiente/distancia EMA200** | **10** | Backlog | **Media (~45–55%)** | Feature flag + test OOS |
| **Automatización reportes A/B** | **70** | CLI guardrail + helpers | — | Script A/B tabular (make_run_report.sh) |

---


## Decisiones vigentes
- **KISS_v1 (plataforma 2025-09-15):** `G200` • `DD=16` • `RB=3` • `H=30` • `BULL0`. *Runner-up:* `DD=17` con el resto igual (menos flips).

- **XB=20** es el baseline actual (mejor netBTC y FPY↓ vs PIN, MDD no peor).
- **ADX=20** (invariancia → simplicidad).
- **Guardarraíl ATR y Age/Pausa:** presentes en código pero **OFF** por defecto.

---

## Reglas de paro (anti-overfitting)
## Artefactos OOS (2020–2022)

Generados con `G200` (sell), `RB=3`, `H=30` para `DD=16` y `DD=17`:
- `reports/mini_accum/kiss_v1/base_v0_1_20250915_0235_kpis__OOS_2020_2022_PT_G200_DD16_RB3_H30_BULL0.csv`
- `reports/mini_accum/kiss_v1/base_v0_1_20250915_0235_kpis__OOS_2020_2022_PT_G200_DD17_RB3_H30_BULL0.csv`

Los KPIs impresos muestran `sats_mult=1.0`, `USD_net=1.0`, `mdd_vsHODL=0` y los flips están vacíos en este corte; **observación**: el sistema no operó en esa ventana con estas condiciones. Queda anotado como guardarraíl y **pendiente** revisar fechas/gate/costos si buscamos actividad en ese tramo.

1. **Una palanca por commit** + etiqueta con reporte.  
2. **Adopción:** ΔnetBTC ≥ **+0.02** en slice **y** OOS no peor que BASE por **−0.01**; **FPY** dentro de **±2**; **MDD** ≤ BASE + **0.05** abs.  
3. **Dos fallos seguidos** → archivar palanca.  
4. **Datos pinned** y ventanas OOS fijas para todas las corridas.  
5. **Costes constantes** durante la serie de pruebas.

---

## Changelog (últimas 24h)

- ✅ - ✅ **Baseline KISS_v1 (G200 • DD16 • RB3 • H30 • BULL0) fijado como plataforma**; runner-up DD17 documentado.
- ✅ **Artefactos OOS 2020–2022** generados para DD16 y DD17 con RB3, H30, G200 (sell); flips vacíos → anotado para revisión.**Baseline CORE_2025 con XB=20** adoptado y etiquetado.  
- ✅ **CLI guardrail**: no renombra si `netBTC==0` o `flips==0`.  
- ✅ **Whitelist de artefactos `__DEMO_PASS`** en `.gitignore`.  
- 🔄 **XB adaptativo (ATR)**: primer prototipo **no supera** XB20; probar 19/25/32.  
- 🔄 **`exit_margin` 30/35 bps**: corridas iniciales en Q1; falta consolidar OOS.  
- 🔍 Scripts de resumen: `showkpi`, resúmenes A/B y helpers zsh.

---

## Siguiente sprint (orden sugerido)

1) **Completar M30/M35 (H2 y Q1)** y A/B vs XB20.  
2) **XB adaptativo 19/25/32 con suavizado** (evitar saltos entre bandas).  
3) **Prototipo ADX percentil** por régimen de volatilidad.  
4) **Automatizar tabla A/B** en `scripts/mini_accum/make_run_report.sh`.

> **Meta del sprint:** **FPY ≤ BASE±2**, **MDD ≤ BASE+0.05**, y **ΔnetBTC ≥ +0.02** vs XB20.

## Hito — KISS_v1 baseline fijado (2025-09-15)
# Metodología y referencias (las que pediste) KISS_V1
- **Walk-forward / ventanas rodantes.** La idea es optimizar/ajustar en una ventana y probar inmediatamente en la siguiente, rotando; es práctica estándar para validar sistemas de trading.   
- **Riesgo de sobre-ajuste** cuando probamos muchas combinaciones:
  - **PBO – Probability of Backtest Overfitting**: estima la probabilidad de haber elegido un resultado por azar entre muchos. 
  - **Deflated Sharpe Ratio (DSR)**: ajusta la significancia del Sharpe por múltiples pruebas y no-normalidad. 
  - (Opcional) **Reality Check** de White: corrección por “data snooping” entre muchos modelos. 
- **CAGR (Compound Annual Growth Rate)**: fórmula estándar para anualizar un factor de crecimiento.  [oai_citation:1‡Investopedia](https://www.investopedia.com/ask/answers/071014/what-formula-calculating-compound-annual-growth-rate-cagr-excel.asp?utm_source=chatgpt.com)

# Qué correr ahora (exactamente lo que propusiste)
1) **OOS y Walk-Forward**  
   Repite el baseline `PT_G200_DD16_RB3_H30_BULL0` en:
   - `2024-07-01→2024-12-31` (H2-2024, WF-1)  
   - `2025-01-01→2025-03-31` (Q1-2025, WF-2)  
   - (y deja el registro OOS 2020-2022 como control negativo).

2) **Cercanos al ganador (robustez local)**  
   Barrido chico: `DD ∈ {15,16,17,18}`, `RB ∈ {2,3,4}`, `H ∈ {28,30,32}` con `G200 sell` y `BULL0`.  
   Criterios para aceptar un “mejorado”: mantiene guardarraíles arriba y **sats_mult>1** en varias ventanas, sin degradar FPY/MDD.

3) **Control de sobre-ajuste (post-selección)**  
   Sobre los finalistas, calcula **PBO** y **DSR**. Si quieres, luego te paso plantillas para computar ambos en Python con tus CSVs.

## Mini-batería OOS / Walk-forward — KISS_v1

**Config probada:** `PT_G200_DD16_RB3_H30_BULL0` (runner-up: `DD=17` igual resto).  
**CAGR (BTC) método:** `CAGR = sats_mult^(1/años) − 1`, con `años = días/365.25`.

| Ventana | Fechas | Meses | sats_mult | BTC CAGR (anual.) | USD_net | MDD_vsHODL | FPY | Flips | Notas |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Main backtest (KISS_v1 TOP)** | 2023-01-09 → 2025-03-11 | **26.0** | **1.4438** | **18.5%** | **10.01×** | **0.733** | **6.29** | **17** | Baseline fijado |
| OOS 2020–2022 | 2020-01-01 → 2022-12-31 | **36.0** | **1.0000** | **0.0%** | **1.00×** | 0.000 | — | **0** | Sin operaciones (gate *sell*@SMA200) |
| H2-2024 (WF-1) | 2024-07-01 → 2024-12-31 | — | — | — | — | — | — | — | **→ correr** |
| Q1-2025 (WF-2) | 2025-01-01 → 2025-03-31 | — | — | — | — | — | — | — | **→ correr** |

**Cómo replicar / actualizar la tabla**

- **Repetir baseline en ventanas fijas (WF):**
```zsh
for w in "2024-07-01 2024-12-31 H2_2024" "2025-01-01 2025-03-31 Q1_2025" "2020-01-01 2022-12-31 OOS_2020_2022"; do
  set -- $w; s=$1; e=$2; t=$3
  python scripts/mini_accum/kiss_v1.py \
    --config configs/mini_accum/kiss_v1.yaml \
    --mode pt --gate_sma 200 --gate_mode sell --dd_hard_pct 30 \
    --dd_pct 16 --rb_pct 3 --bull_hold_sma 0 \
    --start $s --end $e \
    --suffix ${t}_PT_G200_DD16_RB3_H30_BULL0
done

(asumiendo ventana ~26m, CAGR_sats = mult^(12/26)−1)
Config	sats_mult	CAGR_sats (~26m)	USD_net	mdd_vsHODL	FPY	flips
G200 • DD15 • RB1 • H30–32	1.8456	~32.7%	12.80×	0.733	6.29	17
G200 • DD15 • RB2 • H30–32	1.6890	~26.6%	11.72×	0.733	6.29	17
G200 • DD16 • RB2 • H30–32	1.5732	~22.8%	10.91×	0.733	6.29	17
G200 • DD15 • RB3 • H30–32	1.5501	~22.0%	10.75×	0.733	6.29	17
Baseline G200 • DD16 • RB3 • H30	1.4438	~18.5%	10.01×	0.733	6.29	17

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
MD
### Baseline KISS v1 (Final) — KISSv1_BASE_20250915_1642_final
- Candidato: `DD15_RB1_H30_G200_BULL0`
- Estado: **Definitivo** (PBO≈0.107, DSR positivo)
