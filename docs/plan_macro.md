# Proyecto Bot BTC — **Plan Macro** (v2025-09-01)
**Actualizado:** 2025-09-01 03:35 UTC  
**Zona horaria canon:** UTC (todas las marcas de tiempo en UTC; resample 4h para mezcla)  
**Objetivo primario:** **acumular BTC** (numeraire BTC). Todas las métricas de riesgo y PnL se evalúan en términos de BTC.

---
Checklist de activación (go/no-go)
	•	✅ Diamante pasa gates OOS (PF/WR/trades/MDD).

	•	✅ Perla pasa gates y muestra baja corr con Diamante en rango.

	•	✅ Corazón (sombra) reduce MDD/Vol sin matar PF (>−5%).

	•	✅ Allocator (sombra) reproduce mezcla estable, turnover dentro de presupuesto, vol error < 5%.

	•	✅ Circuit breakers y kill-switch probados en stress tests (crashes, gaps).

	•	✅ Timestamps/freshness en UTC comprobados end-to-end.
---

## 0) Principios rectores
- **Arquitectura por capas:** Diamante (4h, oportunista/breakout) + Perla Negra (semanal, estable/contracíclica) → **Corazón** (semáforo suave + LQ + correlación) → **Cerebro/Allocator** (sizing global, ξ*, vol targeting, caps).
- **Modularidad y seguridad:** filtros **suaves** (no ON/OFF), **histéresis** y **dwell**; **freeze semanal** de ξ*; **circuit breakers**; **TTL** de señales; **cap** de exposición; **kill-switch** de MDD/vol.
- **BTC-numeraire:** el éxito se mide en BTC; evitar métricas/decisiones en USD.
- **Reproducibilidad:** walk-forward/holdout, “same ENV”, costos realistas (fee=6 bps, slip=6 bps), logs por capa (base→corr→rr→vol→signed).
- **Gobernanza de “selected” (Diamante):** no cambiar `configs/diamante_selected.yaml` salvo que el candidato supere **gate + no-regresión** en las mismas ventanas OOS y con costes durante **4–6 semanas**, y **solo con confirmación**. Snapshot en `configs/diamante_selected_YYYYMMDD_HHMM.yaml` + baseline en `reports/baseline/diamante_week0.json`.

---

## 1) Organización por hilos (ownership)
- **Proyecto Bot BTC — Diamante:** señales 4h, micro-grid, gates, política de no-regresión. Artefactos: `reports/diamante/*`.
- **Proyecto Bot BTC — Perla Negra:** señales semanales (exposure/ret), estabilidad y **baja correlación** con Diamante. Artefactos: `reports/perla/*`.
- **Proyecto Bot BTC — Corazón:** semáforo suave (ADX/EMA50/ATR), **LQ** con histéresis, gate de **correlación**, pesos `weights.csv`, overlay y **ξ*** (freeze). Artefactos: `reports/heart/*`.
- **Proyecto Bot BTC — Cerebro (Allocator):** mezcla con pesos de Corazón, corr-gate, **ξ***, vol targeting, caps, reporter. Artefactos: `reports/allocator/*`.

> **Regla:** cada script vive y se itera en su **hilo** correspondiente. La activación en vivo se aprueba al cierre de cada fase.

---

## 2) Fases (secuenciación y criterios de salida)

### Fase A — **Diamante** (Semanas 0–6)
**Meta:** edge claro en tendencia/expansión; estabilidad OOS con costes.
- **Gates (OOS, con costes):** PF ≥ **1.6**, WR ≥ **60%**, **≥30** trades por fold, MDD ≤ **8%** (BTC).
- **Validaciones por régimen:** bins de ADX (<15, 15–20, >20), |slope(EMA50)|, percentiles de ATR.
- **Correlación (diagnóstico):** medir corr con Perla (si hay serie); `--perla_csv`, `--max_corr` (filtrado opcional).
- **Entregables:** `reports/diamante/kpis_regimen.md`, `reports/kpis_grid.csv`, `reports/selected_vs_bh.md`, snapshots “selected”.

**Criterio de salida Fase A → B:** gates cumplidos en 2–3 ventanas OOS con fricción; no-regresión vs “selected”.

---

### Fase B — **Corazón** (C1–C2, **modo diagnóstico**)
**Meta:** pesos **suaves** por régimen + LQ + correlación que **mejoren riesgo** sin matar PF.
- **Semáforo:** 3 estados (Verde/Amarillo/Rojo) con **dwell 6–8** y **maxΔ 0.2** por barra.
- **LQ:** `m_lq ∈ (0.7, 1.0)` con histéresis 2 velas (riesgo de cluster opuesto cercano/fuerte).
- **Gate de correlación:** lookback 60–90d, umbral **0.35–0.40**, penalización máxima **30%** a la pierna de **peor performance 30d**.
- **Salidas:**  
  - `corazon/weights.csv` → `timestamp,w_diamante,w_perla` (∈[0,1], suma≈1).  
  - `corazon/lq.csv` → `timestamp,lq_flag` ∈ {HIGH_RISK,NORMAL}.  
  - `corazon/heart_rules.yaml` (**v0.2**; canonical).  
  - Overlay **ξ***: `reports/heart/xi_star.txt` (freeze **lunes**).
- **Criterios (overlay):** reducir **MDD ≥ 15%** **o** **Vol ≥ 10%** vs 50/50, con ΔPF ≥ −5% y ΔTurnover ≤ +20%.

**Criterio de salida B → C:** pesos estables; overlay cumple criterios sin aumento de costos excesivos.

---

### Fase C — **Cerebro / Allocator v0.1-R (modo sombra)**
**Meta:** simular mezcla y sizing con pesos de Corazón; **no** enviar órdenes aún.
- **Componentes:** corr-gate; **ξ*** (freeze semanal, cap 1.70×); **circuit breakers** (vol diaria > p98 o DD día ≤ −6% → ξ*=1.0); **vol targeting** 20% (clamp 0.5–1.2); **w_cap_total = 1.0**; **exec_threshold** por turnover (p.ej. 2%).
- **Entradas:**  
  - `signals/diamante.csv` → `timestamp,sD[-1..1],w_diamante_raw[0..1],retD_btc` (4h).  
  - `signals/perla.csv` → `timestamp,sP[-1..1],w_perla_raw[0..1],retP_btc` (weekly→4h ffill).  
  - `corazon/weights.csv`, `corazon/lq.csv`, `reports/heart/xi_star.txt`.
- **Salidas:**  
  - `reports/allocator/sombra_kpis.md` (PF, WR, MDD, Sortino, turnover, corr_dp, vol_error<5%).  
  - `reports/allocator/curvas_equity/{eq_base.csv, eq_overlay.csv}`.
- **Go/No-Go a Fase D:** tracking consistente, **vol_error < 5%**, costes esperados, overlay mejora riesgo.

Fases y entregables (secuenciación)

Fase 0 — Estrategias base “OK”

Diamante (4h)
	•	Validación OOS por régimen (bins ADX/ATR/slope).
	•	Gates (con costes): PF ≥ 1.6, WR ≥ 60%, ≥30 trades en ambos folds, MDD ≤ 8%.
	•	Artefactos: reports/diamante/* (kpis_grid, selected_vs_bh, baseline snapshots).

Perla Negra (semanal)
	•	Validación OOS en rango/bear/transición; estabilidad de señal y bajo turnover.
	•	Gates: PF ≥ 1.2 en rangos, MDD ≤ 15%, corr(D,P) ≤ 0.35–0.40 en neutro/rango.
	•	Artefactos: reports/perla/* + signals/perla.csv (resample 4h con ffill para mezcla).

Fase 0.5 — Corazón “diagnóstico” (sin tocar ejecución)
	•	Semáforo suave 3 estados (Verde/Amarillo/Rojo) con dwell 6–8 velas y maxΔ peso 0.2 por barra.
	•	LQ con histéresis 2 velas: m_lq ∈ {0.70, 1.00}.
	•	Gate correlación (ventana 60–90d, umbral 0.35–0.40): penalizar hasta 30% a la pierna con peor performance reciente.
	•	Salida: reports/corazon_weights/BTC-USD_weights.csv (timestamp, w_diamante, w_perla).
	•	Meta: overlay que baje MDD ≥ 15% o Vol ≥ 10% sin degradar PF > 5% ni subir turnover > 20%.

Fase 1 — Allocator v0.1-R “modo sombra”
	•	Mezcla con pesos de Corazón, corr-gate, ξ* (freeze lunes) y circuit breaker (vol diaria > p98 o DD día ≤ −6% → ξ*=1.0).
	•	Vol targeting anual 20% (clamp 0.5–1.2), w_cap_total = 1.0, exec_threshold por turnover (p. ej. 2%).
	•	Salidas:
	•	reports/allocator/sombra_kpis.md (PF, WR, MDD, Sortino, turnover, corr_dp, vol_error < 5%).
	•	reports/allocator/curvas_equity/*.csv (eq_base vs eq_overlay).
	•	reports/heart/xi_star.txt (congelado semanal).
	•	Criterio de “Go” a Fase 2: reproducibilidad OOS, costes dentro de presupuesto, vol error < 5%, overlay mejora riesgo.

Fase 2 — Live con guardarraíles
	•	Activar órdenes con w_cap_total = 0.30–0.50 y maxΔ peso = 0.10 por 2–3 semanas.
	•	Monitoreo de slippage y divergencia vs sombra. Kill-switch por MDD rolling 30d o vol/turnover anómalos.

Fase 3 — Pleno (si pasa Fase 2)
	•	Levantar caps gradualmente; habilitar ξ* > 1.0 si el overlay demostró consistentemente menor MDD/vol.
	•	Mantener freeze semanal y circuit breakers.

Interfaces (contratos de datos)
	•	Zona horaria: UTC; timestamps alineados por cierre 4h.
	•	signals/diamante.csv: timestamp, sD[-1..1], w_diamante_raw[0..1], retD_btc.
	•	signals/perla.csv: timestamp, sP[-1..1], w_perla_raw[0..1], retP_btc (resample 4h con ffill).
	•	corazon/weights.csv: timestamp, w_diamante[0..1], w_perla[0..1] (suma ≈ 1).
	•	corazon/lq.csv: timestamp, lq_flag ∈ {HIGH_RISK,NORMAL}.
	•	Freshness: TTL=4h; señal fuera de TTL → peso 0.

Reglas de régimen (semilla)
	•	Verde: ADX≥20 y |slope(EMA50)|>umbral; ATR% medio/alto → D=0.7–0.9, P=0.1–0.3.
	•	Amarillo: transición (ADX subiendo  <15→>20) → 50/50 (dwell 6–8).
	•	Rojo: rango (ADX<18 y ATR%<p40) → P=0.7–0.9, D=0.1–0.3.
	•	LQ: riesgo alto → multiplicar ambos por 0.7.

KPIs & Gates de aceptación (resumen)
	•	Diamante: PF≥1.6, WR≥60%, ≥30 trades/fold, MDD≤8%.
	•	Perla: PF≥1.2 (rango), MDD≤15%, corr(D,P)≤0.35–0.40.
	•	Corazón (overlay): −MDD≥15% o −Vol≥10% con ΔPF ≥ −5% y ΔTurnover ≤ +20%.
	•	Allocator sombra: vol_error <5%, PF no peor que 50/50 en igual fricción, costos acordes.
	•	Live (fase 2): tracking vs sombra dentro del rango de slippage esperado.

Controles de riesgo y operativa
	•	ξ*: xi_star = clamp(min(mdd_ratio, vol_ratio)*0.85, 1.0, 1.70), freeze lunes; CB resetea a 1.0.
	•	Vol targeting: objetivo 20% (clamp 0.5–1.2), asimétrico.
	•	Correlación: ventana 60–90d, umbral 0.35–0.40, penalización máx 30% a la pierna débil.
	•	Cap total: |w_total| ≤ 1.0; exec_threshold por costos.
	•	Kill-switch: a cash si MDD 30d < −X% (definir X) o señales obsoletas.

Artefactos y rutas (estándar)

project/Bot_BTC/
  signals/{diamante.csv, perla.csv}
  corazon/{weights.csv, lq.csv, heart_rules.yaml}
  reports/
    diamante/*, perla/*, heart/*, allocator/*
    allocator/curvas_equity/{eq_base.csv, eq_overlay.csv}
    allocator/sombra_kpis.md
    heart/xi_star.txt

Cadencia semanal (sugerida)
	•	Lunes: refresh de ξ* (freeze) y corte de KPIs semana previa.
	•	Martes–Jueves: validaciones OOS, stress tests, revisión de correlación y costos.
	•	Viernes: snapshot de “selected”, checks de no-regresión, aprobación Go/No-Go.
	•	Diario (cada 4h): generación de pesos Corazón y corrida del Allocator en modo sombra.

---

### Fase D — **Perla Negra V2 (semanal)**
**Meta:** amortiguar rangos/bear/transición; **baja correlación** con Diamante.
- **Gates:** PF ≥ **1.2** en rangos, MDD ≤ **15%**, corr(D,P) ≤ **0.35–0.40** en neutro/rango.
- **Entrega de serie para mezcla:** `reports/perla/perla_exposure_latest.csv` y consolidado `signals/perla.csv` (4h ffill).
- **KPIs:** estabilidad, bajo turnover, consistencia en bins de rango/bear/transición.

---

### Fase E — **Live con guardarraíles** (rampa 2–3 semanas)
**Meta:** ejecutar órdenes con caps prudentes y monitoreo estricto.
- **Parámetros de rampa:** `w_cap_total = 0.30–0.50`, `maxΔ peso = 0.10` por barra.
- **Kill-switch:** a cash si MDD rolling 30d < −X% (definir X) o señales obsoletas (TTL=4h).
- **Monitoreo:** slippage, tracking vs sombra, costos; auditoría diaria de frescura de señales.

---

### Fase F — **Pleno**
**Meta:** levantar caps y habilitar **ξ\* > 1.0** si el overlay demostró **menor MDD/vol** de forma consistente.
- Mantener **freeze semanal** y **circuit breakers**; revisión periódica de corr-gate y vol targeting.
- Registrar cambios versionados en `configs/` y snapshots en `reports/baseline/`.

---

## 3) Interfaces (contratos de datos)
**Zona horaria:** **UTC** (todas las series).  
**Rejilla de mezcla:** 4h (Perla resampleada con ffill).

- `signals/diamante.csv`:  
  `timestamp,sD[-1..1],w_diamante_raw[0..1],retD_btc`
- `signals/perla.csv`:  
  `timestamp,sP[-1..1],w_perla_raw[0..1],retP_btc`  *(weekly → 4h ffill)*
- `corazon/weights.csv`:  
  `timestamp,w_diamante[0..1],w_perla[0..1]`  *(suma≈1)*
- `corazon/lq.csv`:  
  `timestamp,lq_flag ∈ {HIGH_RISK,NORMAL}`
- `reports/heart/xi_star.txt`:  
  texto simple `1.00–1.70` (actualiza **lunes**).

**Freshness/TTL:** 4h. Si una señal/peso/archivo está fuera de TTL → peso 0 por seguridad.

---

## 4) Reglas de régimen (semáforo, semilla)
- **Verde:** ADX≥20 y |slope(EMA50)|>umbral; ATR% medio/alto → **D=0.7–0.9**, **P=0.1–0.3**.
- **Amarillo:** transición (ADX subiendo de <15→>20) → **D=0.5**, **P=0.5** (dwell 6–8).
- **Rojo:** rango (ADX<18 y ATR%<p40) → **P=0.7–0.9**, **D=0.1–0.3**.
- **LQ:** riesgo alto → multiplicar ambos por **0.70** + histéresis 2 velas.

---

## 5) Controles de riesgo (globales)
- **ξ\***: `xi* = clamp(min(mdd_ratio, vol_ratio) * 0.85, 1.0, 1.70)`; **freeze** semanal (lunes).  
  **Circuit breakers:** vol diaria > **p98** **o** DD día ≤ **−6%** → `xi*=1.0`.
- **Vol targeting:** objetivo **20%** anual; clamp **[0.5, 1.2]** (asimétrico, prudente).
- **Correlación:** ventana 60–90d, umbral **0.35–0.40**, penalización máx **30%** a la pierna con **peor performance 30d**.
- **Cap total:** |w_total| ≤ **1.0**; **exec_threshold** por turnover (p.ej., **2%**).  
- **Kill-switch:** a cash si MDD 30d < −X% (definir X) o señales obsoletas/archivo corrupto.

---

## 6) Cadencia operativa (semana tipo)
- **Lunes:** refrescar `xi_star.txt`; publicar KPIs semana previa; sanity de frescura (TTL).  
- **Mar–Jue:** OOS/stress; validar corr-gate; revisar costos/turnover; ajustes en `heart_rules.yaml` si aplica.  
- **Viernes:** snapshots “selected”, reportes de no-regresión; Go/No-Go de fase/sprint.  
- **Cada 4h:** generar `corazon/weights.csv` y correr **Allocator en modo sombra**.

---

## 7) Estructura de carpetas estándar
```
project/Bot_BTC/
  signals/
    diamante.csv
    perla.csv
  corazon/
    heart_rules.yaml
    weights.csv
    lq.csv
  configs/
    allocator_sombra.yaml
    diamante_selected.yaml
  reports/
    diamante/*
    perla/*
    heart/
      xi_star.txt
    allocator/
      sombra_kpis.md
      curvas_equity/
        eq_base.csv
        eq_overlay.csv
```

---

## 8) KPIs & gates de aceptación (resumen)
- **Diamante:** PF≥1.6, WR≥60%, ≥30 trades/fold, MDD≤8% (BTC).  
- **Perla:** PF≥1.2 (rango), MDD≤15%, corr(D,P)≤0.35–0.40.  
- **Corazón (overlay):** −MDD≥15% **o** −Vol≥10% con ΔPF ≥ −5% y ΔTurnover ≤ +20%.  
- **Allocator (sombra):** vol_error <5%, PF no peor que 50/50 en igual fricción; costos dentro de presupuesto.  
- **Live (rampa):** tracking vs sombra dentro del rango de slippage esperado.

---

## 9) Cambios vs versión anterior (changelog)
- Se introduce **Corazón v0.2** con semáforo suave, **LQ** con histéresis, y **gate de correlación** (lookback 60–90d, thr 0.35–0.40, penalización ≤30%).
- Se añade **Allocator v0.1-R en modo sombra** con **ξ*** (freeze), **circuit breakers**, **vol targeting** (20% con clamp 0.5–1.2) y **w_cap_total=1.0**.
- Se formaliza la **organización por hilos** (scripts/iteraciones por chat dedicado).
- Se refuerza la **gobernanza de “selected”** de Diamante y snapshots versionados.
- Se estandarizan **interfaces** CSV/TTL y estructura de carpetas.

---
## 10) Decisiones recientes — 2025-08-31 (Corazón en sombra)
- **Reporter**: comparar **baseline vs overlay** (no la misma columna).
  - **Baseline**: `ret_4h`; **Overlay**: `ret_4h_overlay`; usar `--ts_col timestamp`.
  - Alineación por timestamp con tolerancia **2h**.
- **Hallazgo**: MDD base → overlay: −1.7351% → −0.8694% (~2× menos) en consola; KPIs corregidos con la columna overlay.
- **Regla de re-riesgo (ξ\*) adoptada**: `xi* = min(MDD_base/MDD_overlay, vol_base/vol_overlay) * 0.85` con **cap 1.70×** y **freeze semanal (lunes)**.
- **Política**: Diamante intocado por defecto (export por-barra opt-in `CORAZON_EXPORT_BARS=1`).
- **Operativa**: Corazón en sombra (régimen + LQ soft-veto); si la mejora de MDD se sostiene 1–2 semanas con FREEZE → **guardrail opt-in en W5** sin interferir con Perla/Diamante.
- **Validación what-if**: escalar `ret_overlay` por **1.6964×** y re-correr KPIs (esperado: MDD≈baseline, NET↑).
