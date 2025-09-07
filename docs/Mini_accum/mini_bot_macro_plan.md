# Mini-BOT de Acumulación BTC — **Plan Macro (Sandbox)**
*Versión:* V3-lite · *Fecha:* 2025-09-05 11:11:54 UTC

---


## 0) Contexto y Alcance
- **Hilo dedicado SOLO** a este mini-bot (sandbox). No toca Diamante/Perla/Corazón directamente.
- **Filosofía:** Rotación **binaria** BTC ↔ Stable (USDC), **sin shorts**.
- **Numeraire:** BTC (objetivo principal = **acumular satoshis**).
- **Modo de trabajo:** El mini-bot funciona como **campo de pruebas** para reglas simples portables a BOT BTC (Corazón/Allocator) **si** superan KPIs OOS.

---

## 1) Resumen Ejecutivo
- **Misión:** Superar HODL en BTC a 6–12 meses con **menor MDD** y **bajo turnover**; operar primero en **MODO SOMBRA**.
- **Modelo operativo:** 3 estados **claros** y auditables:
  - **Acumulativo (BTC 100%)**, **Defensivo (Stable 100%)**, **Hibernación (PAUSA)**.
- **Reloj:** Evaluación **cada 4h (UTC)** al cierre; ejecución al **open** de la siguiente vela.
- **Regla base:** Macro-filtro **EMA200 (D1)**; anti-whipsaw con **dwell** y **TTL**; **flip-budget** (hard 26/año; soft 2/mes).
- **Costes realistas:** **fee 6 bps + slip 6 bps por lado** (≈ 24 bps round-trip).
- **Sandbox de mejoras (opt-in, solo si pasan ablation + KPIs):**
  1) **Señal 21/55 (4h)** para entrada/salida.
  2) **Salida activa** bajo EMA21 con **confirmación** (2 velas).
  3) **Corazón slim:** regímenes por **ATR% (p40)** → verde/amarillo/rojo.
  4) **Grace TTL / cooldown** de flips.
  5) **ATR-adaptativo (2 niveles)**: dwell/flip-throttle.
  6) **Turnover budget** semanal (soft).

---

## 2) KPIs de Aceptación y Gobernanza (Congelados)
- **Win vs HODL (BTC):** `Net_BTC_ratio ≥ 1.05` (con costes).
- **MDD:** `MDD_model ≤ 0.85 × MDD_HODL`.
- **Turnover:** `flips/año ≤ 26` (soft cap **2/mes**).
- **Robustez:** estabilidad en **2–3 ventanas OOS** fijas: **2022H2, 2023Q4, 2024H1** (y una ventana de **estrés** sugerida: **2021Q2–Q3**).
- **Política de promoción (no-regresión):** un módulo entra **solo** si mejora MDD/turnover **sin** romper Net_BTC_ratio **ni** el presupuesto de flips.

---

## 3) Arquitectura Base y Módulos Opt-in (V3-lite)
### Capa Base (línea de referencia)
- **Macro:** Precio D1 vs **EMA200_D1** ⇒ BTC 100% ó Stable 100%.
- **Frecuencia:** decisiones **4h UTC**; ejecutar en el **open** siguiente.
- **Anti-whipsaw:** `dwell=4` velas 4h, `ttl=1` vela de confirmación tras cruce macro.
- **Presupuesto:** **26/año** (hard) y **2/mes** (soft).

### Módulos Opt-in (solo si aportan)
1. **Señales 4h (EMA21/55)**  
   - **Entrada:** macro verde **y** `EMA21_4h > EMA55_4h`.  
   - **Salida (pasiva):** `EMA21_4h < EMA55_4h`.
2. **Salida activa (confirmada)**  
   - En BTC: si `close_4h < EMA21_4h` **y** la **siguiente** vela **no** cierra > EMA21_4h ⇒ **salir** a Stable.
3. **Regímenes ATR% (Corazón slim)**  
   - **Verde:** `close_4h > EMA200_D1` **y** `ATR%_D1 ≥ p40`.  
   - **Rojo:** `close_4h < EMA200_D1` **y** `ATR%_D1 ≥ p40`.  
   - **Amarillo:** resto ⇒ **Hibernación/PAUSA**.
4. **Grace TTL (cooldown suave)**  
   - Tras un flip, durante `ttl_grace=1` vela: exigir señal **fuerte** para revertir (p.ej. `|precio_d1 − EMA200_d1| / ATR_d1 ≥ 0.30`).
5. **ATR-adaptativo (2 niveles)**  
   - `ATR% ≥ p60` ⇒ `dwell=5`, **máx 1 flip/24h**.  
   - `ATR% < p60` ⇒ `dwell=3`, **máx 2 flips/24h`.
6. **Turnover budget semanal (soft)**  
   - Techo blando **8 flips/semana**; si se excede, **solo** cambios verde↔rojo.

> **Criterios de promoción por módulo:**  
> - **ATR% (slim):** MDD↓ ≥10% **o** flips↓ ≥10% con `Net_BTC_ratio ≥ baseline − 0.5 pp`.  
> - **Grace TTL:** turnover↓ ≥10% con `Net_BTC_ratio ≈` (±0.5 pp).  
> - **Nudge/alineación (implícito en 21/55+macro):** `dwell` −1 del lado alineado; flips ≤ +10%.  
> - **ATR-adaptativo:** costes/turnover↓ con `Net_BTC_ratio ≈`.  
> - **Budget semanal:** costes↓ ≥15% con `Net_BTC_ratio ≈`.

---

## 4) V3-lite por Fases (Investigación, Sandbox y Robustez)
### **Fase A — Base mínima (benchmark)**
- Regla EMA200_D1 con `dwell=4`, `ttl=1`, flip-budget.  
- Objetivo: **curva de referencia** vs HODL (BTC) y mapa de estados.

### **Fase B — Señales 4h (tendencia)**
- Añadir **EMA21/55_4h** (entrada/salida) y **salida activa** (2 velas).  
- Aceptar si **MDD↓** o **turnover↓**, Net_BTC_ratio ≥ baseline y flips en presupuesto.

### **Fase C — Regímenes de volatilidad**
- **ATR% p40** para habilitar Verde/Rojo; Amarillo ⇒ **PAUSA**.  
- (Opcional) Hibernación por “chop”: ≥2 cruces 21/55 en 40 velas 4h.  
- Aceptar si **MDD↓ ≥10%** o **flips↓ ≥10%** con Net_BTC_ratio ≥ baseline − 0.5 pp.

### **Fase D — Micro-operativa (suavizado)**
- **Grace TTL** y **ATR-adaptativo (2 niveles)**; **budget semanal** (8 flips).  
- Aceptar si **costes/turnover↓ (≥10–15%)** y Net_BTC_ratio ≈ baseline.

### **Fase E — Estrés y robustez final**
- OOS: **2022H2, 2023Q4, 2024H1** + **estrés 2021Q2–Q3**.  
- Sensibilidad ±10–20% en `dwell`, `ttl`, p40/p60.  
- Congelar **v0.1-selected** si pasa KPIs en ≥2 ventanas y respeta flip-budget.

---

## 5) Fases Operativas 1→4 (Roadmap de Entrega)
### **Fase 1 — Backtesting riguroso (Datos + Ablation)**
- **Datos (3–5 años):** BTC-USD **D1** (EMA200) y **4h** (decisiones), en **UTC**; sin huecos/duplicados.  
- **Benchmarks a simular (en BTC):**  
  1) **HODL** (1 BTC inicial).  
  2) **Base (Capa A)**: EMA200_D1 + dwell/ttl + flip-budget.  
  3) **+Capa 1**: EMA21/55_4h.  
  4) **+Capa 2**: **salida activa**.  
  5) **+Capa 3**: **ATR% p40** (slim) y/o **hibernación**.  
  6) **Experimentos:** **ATR-adaptativo** y **budget semanal**.  
- **Costes:** **6 + 6 bps** por lado.  
- **KPIs (congelados)** y **criterio de promoción** a Fase 2: Base o Base+Capa1 cumplen KPIs en ≥2 ventanas OOS; ningún módulo añadido rompe MDD/turnover.

### **Fase 2 — Desarrollo e Implementación (Skeleton + Sombra)**
- **Estructura (monorepo):**  
  - `docs/mini_accum_plan.md`, `configs/mini_accum.yaml`.  
  - `btcbot/mini_accum/{engine.py,rules.py,kpis.py}` (máquina de estados y reglas).  
  - `scripts/mini_accum/backtest.py` (runner sobre CSV, modo sombra).  
- **Reloj y datos:** D1/4h en **UTC**; ejecutar al **open** siguiente.  
- **DoD Fase 2 → 3:** reproducir KPIs de Fase 1 (±2–3%), respetar flip-budget; artefactos en `reports/mini_accum/*` y logs completos.

### **Fase 3 — Paper Trading / Testnet (2–4 semanas)**
- **Ingesta viva read-only** con ccxt (BinanceUS) y scheduler 4h UTC.  
- **Sombra operativa:** estados y órdenes simulados (o testnet con `DRYRUN=1`); costes aplicados.  
- **Observabilidad:** dashboard básico (estado, balance BTC, flips), alertas por flip/error.  
- **DoD Fase 3 → 4:** 2–4 semanas estables, KPIs ≈ backtest, flip-budget respetado.

### **Fase 4 — Despliegue controlado (capital mínimo)**
- **Rollout canario:** **10–20%** del capital; **kill switch** (`override_mode`).  
- **Infra 24/7:** VPS, watchdog, backups, monitor de KPIs vs HODL.  
- **Gobernanza:** no-regresión; revisión mensual/trimestral; revertir a **PAUSA** si falla 2 cortes seguidos.

---

## 6) Parámetros Iniciales (YAML guía)
```yaml
asset: BTC-USD
stable: USDC
exchange: binanceus
timezone: UTC
decision_freq: 4h
freeze_end: null

trend_rule:
  ema_d1_period: 200

anti_whipsaw:
  dwell_bars_4h: 4
  grace_ttl_bars_4h: 1

flip_budget:
  hard_cap_year: 26
  soft_cap_month: 2

signals_4h:
  enabled: true
  ema_fast: 21
  ema_slow: 55

exit_active:
  enabled: true
  confirm_next_bar: true

regime_slim:
  enabled: false
  atr_percentile_threshold: 40
  use_hibernation_on_yellow: true

hibernation_chop:
  enabled: false
  lookback_bars_4h: 40
  min_crosses_21_55: 2

cooldown:
  enabled: false
  ttl_grace_bars: 1
  strong_signal_dist_atr: 0.30

atr_adaptive:
  enabled: false
  p60_threshold: 60
  dwell_high_vol: 5
  dwell_low_vol: 3
  max_flips_per_24h_high: 1
  max_flips_per_24h_low: 2

turnover_weekly:
  enabled: false
  soft_cap_flips: 8

costs:
  fee_bps_per_side: 6
  slippage_bps_per_side: 6

kpis:
  net_btc_ratio_min: 1.05
  mdd_ratio_max_vs_hodl: 0.85
  flips_per_year_max: 26
  robustness_windows: [2022H2, 2023Q4, 2024H1]

paths:
  d1_csv:      data/ohlc/1d/BTC-USD.csv
  h4_csv:      data/ohlc/4h/BTC-USD.csv
  reports_dir: reports/mini_accum/
  logs_dir:    logs/mini_accum/
```

---

## 7) Rutas, Estructura y Runbook (solo plan)
- **Monorepo** con paquete aislado `btcbot.mini_accum`; reutiliza loaders y costes comunes (`btcbot.lib`).  
- **Directorios de salida:** `reports/mini_accum/` y `logs/mini_accum/`.  
- **Runbook S0/S1 (documental, sin ejecutar):**
  1) Congelar este plan macro y `configs/mini_accum.yaml`.  
  2) Preparar `docs/mini_accum_plan.md` con KPIs y ventanas OOS.  
  3) Bocetar `scripts/mini_accum/backtest.py` (skeleton).  
  4) Definir `--freeze_end` para cortes semanales reproducibles cuando toque.

---

## 8) Decisiones Congeladas y Pendientes
- **Congeladas:** numeraire BTC; binario 100%/0%; costos 6+6 bps; flip-budget 26/año (2/mes); OOS: 2022H2, 2023Q4, 2024H1; reloj 4h UTC; EMA200_D1; 21/55_4h + salida activa (Base B).  
- **Pendientes (activar solo tras ablation):** `regime_slim (p40)`, `cooldown`, `atr_adaptive`, `turnover_weekly`, `hibernation_chop`.

---

## 9) Definition of Done (Global)
- **Backtest (F1):** KPIs cumplidos en ≥2 ventanas OOS; artefactos completos.  
- **Sombra (F2):** resultados ≈ backtest (±2–3%), flip-budget respetado.  
- **Paper (F3):** 2–4 semanas estables, alertas OK.  
- **Live (F4):** capital canario, KPIs mantenidos, sin incidentes; gobernanza de cambios aplicada.

---

**Nota:** Este documento es *solo plan*; no activa ejecución ni modifica componentes existentes.
