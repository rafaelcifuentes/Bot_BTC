# Proyecto Bot BTC — Plan Macro (v2025-09-02)
**Actualizado:** 2025-09-03 01:54 UTC  
**Zona horaria canon:** UTC (todas las marcas de tiempo en UTC; resample 4h para mezcla)  
**Objetivo primario:** **acumular BTC** (numeraire BTC). Todas las métricas de riesgo y PnL se evalúan en términos de BTC.

---

## 0) Foto actual (corte 2025-09-02)
- **Allocator (CEREBRO):** ✅ congelado con el perfil estable; `tests_overlay_check.py` alinea NET con la curva (Diff≈0).
- **Corazón (semáforo):** ✅ saneado; listo para operar **en sombra** o activarse cuando Perla quede validada OOS.
- **Perla (ojo de mercado):** ✅ edge utilizable. En pruebas recientes (Donchian 40/10, `longflat`), NET overlay de +2.09% con costes (12 bps) y turnover ≈27.2. En configuraciones anteriores ("Opción A"), con otra ventana/config, llegó a +16.3% — se mantiene como referencia, no como baseline.
- **Diamante (ojo de mercado):** ⚠️ prioridad de auditoría/mejora. Raw audit flojo; requiere rediseño/fine‑tuning antes de activar en mezcla.

---

## 1) Principios rectores
- **Arquitectura por capas:** Diamante (4h, oportunista/breakout) + Perla (semanal/4h, contracíclica/estable) → **Corazón** (semáforo suave + LQ + correlación) → **Cerebro/Allocator** (sizing global, ξ*, vol targeting, caps).
- **Modularidad y seguridad:** filtros **suaves** (no ON/OFF), **histéresis** y **dwell**; **freeze semanal** de ξ*; **circuit breakers**; **TTL** de señales; **cap** de exposición; **kill‑switch** de MDD/vol.
- **BTC-numeraire:** el éxito se mide en **BTC**; evitar métricas/decisiones en USD.
- **Reproducibilidad:** walk‑forward/holdout, mismo ENV, **costos realistas** (fee=6 bps, slip=6 bps), logs por capa (base→corr→rr→vol→signed).
- **Gobernanza "selected" (Diamante):** `configs/diamante_selected.yaml` solo cambia si el candidato supera **gate + no‑regresión** (4–6 semanas OOS con costes), y con snapshot versionado.

---

## 2) Ownership por hilos
- **Diamante:** señales 4h, micro‑grid, gates, política de no‑regresión. Artefactos: `reports/diamante/*`.
- **Perla:** señales semanales/4h (exposure/ret). Baja correlación con Diamante. Artefactos: `reports/perla/*`.
- **Corazón:** semáforo suave (ADX/EMA50/ATR), **LQ** con histéresis, gate de **correlación**, pesos `weights.csv`, overlay y **ξ***. Artefactos: `reports/heart/*`.
- **Cerebro/Allocator:** mezcla con pesos de Corazón, corr‑gate, **ξ***, vol targeting, caps, reporter. Artefactos: `reports/allocator/*`.

---

## 3) Fases y criterios de salida

### Fase A — **Diamante** (Semanas 0–6)
**Meta:** edge claro en tendencia/expansión; estabilidad OOS con costes.  
**Gates (OOS, con costes):** PF ≥ **1.6**, WR ≥ **60%**, ≥**30** trades/fold, MDD ≤ **8%** (BTC).  
**Validaciones por régimen:** bins de ADX (<15, 15–20, >20), |slope(EMA50)|, percentiles de ATR.  
**Salida A→B:** gates cumplidos en 2–3 ventanas OOS con fricción + no‑regresión.

### Fase B — **Corazón** (diagnóstico)
**Meta:** pesos **suaves** por régimen + LQ + correlación que **mejoren riesgo** sin matar PF.  
Semáforo 3 estados, **dwell 6–8**, **maxΔ 0.2**/barra. Gate corr: ventana **60–90d**, thr **0.35–0.40**, penalización **≤30%** a la pierna más débil (últimos 30d).  
**Criterio:** reducir **MDD ≥15%** **o** **Vol ≥10%** vs 50/50, con ΔPF ≥ −5% y ΔTurnover ≤ +20%.

### Fase C — **Allocator v0.1‑R (modo sombra)**
**Meta:** simular mezcla y sizing con pesos de Corazón (sin órdenes).  
Componentes: corr‑gate; **ξ*** (freeze semanal, cap 1.70×); **circuit breakers** (vol día>p98 o DD día ≤−6%→ ξ*=1.0); **vol targeting** 20% (clamp 0.5–1.2); **w_cap_total=1.0**.  
**Go/No‑Go C→D:** tracking consistente, **vol_error <5%**, costes esperados, overlay mejora riesgo.

### Fase D — **Perla V2**
**Meta:** amortiguar rangos/bear/transición con **baja correlación** vs Diamante.  
**Gates:** PF ≥ **1.2** (rango), MDD ≤ **15%**, corr(D,P) ≤ **0.35–0.40**, NET OOS > 0.

### Fase E — **Live con guardarraíles**
**Meta:** ejecutar con caps prudentes 2–3 semanas.  
Parámetros de rampa: `w_cap_total=0.30–0.50`, `maxΔ peso=0.10`/barra; freeze de ξ*, CB activos; monitoreo de slippage/divergencia.

### Fase F — **Pleno**
Levantar caps y habilitar **ξ* > 1.0** si el overlay demostró menor MDD/vol de forma consistente.

---

## 4) Interfaces (contratos de datos)
**Zona horaria:** **UTC**. **Rejilla mezcla:** 4h (Perla resample con ffill).

- `signals/diamante.csv`: `timestamp,sD[-1..1],w_diamante_raw[0..1],retD_btc`  
- `signals/perla.csv`: `timestamp,sP[-1..1],w_perla_raw[0..1],retP_btc` *(weekly → 4h ffill)*  
- `corazon/weights.csv`: `timestamp,w_diamante[0..1],w_perla[0..1]` *(suma≈1)*  
- `corazon/lq.csv`: `timestamp,lq_flag ∈ {{HIGH_RISK,NORMAL}}`  
- `reports/heart/xi_star.txt`: texto `1.00–1.70` (actualiza **lunes**).  
**Freshness/TTL:** 4h → señal/peso fuera de TTL ⇒ peso 0.

---

## 5) Reglas de régimen (semilla)
- **Verde:** ADX≥20 y |slope(EMA50)|>umbral; ATR% medio/alto → **D=0.7–0.9**, **P=0.1–0.3**.
- **Amarillo:** transición (ADX sube de <15→>20) → **D≈0.5**, **P≈0.5** (dwell 6–8).
- **Rojo:** rango (ADX<18 y ATR%<p40) → **P=0.7–0.9**, **D=0.1–0.3**.
- **LQ (riesgo alto):** multiplicar ambos por **0.70** + histéresis 2 velas.

---

## 6) Controles de riesgo (globales)
- **ξ***: `xi* = clamp(min(mdd_ratio, vol_ratio) * 0.85, 1.0, 1.70)`; **freeze** semanal (lunes).  
  **Circuit breakers:** vol diaria > **p98** **o** DD día ≤ **−6%** → `xi*=1.0`.
- **Vol targeting:** objetivo **20%** anual; clamp **[0.5, 1.2]** (asimétrico, prudente).
- **Correlación:** ventana **60–90d**, thr **0.35–0.40**, penalización máx **30%** a la pierna más débil (30d).
- **Cap total:** |w_total| ≤ **1.0**; **exec_threshold** por turnover (p.ej., **2%**).
- **Kill‑switch:** a cash si MDD 30d < −X% o señales obsoletas/corruptas.

---

## 7) Cadencia operativa (semana tipo)
- **Lunes:** refrescar `xi_star.txt`; KPIs semana previa; sanity de frescura (TTL).
- **Mar–Jue:** OOS/stress; validar corr‑gate; revisar costos/turnover; ajustes en `heart_rules.yaml` si aplica.
- **Viernes:** snapshots "selected", reportes de no‑regresión; Go/No‑Go.
- **Cada 4h:** generar `corazon/weights.csv` y correr **Allocator en sombra**.

---

## 8) Estructura de carpetas
```
project/Bot_BTC/
  signals/{{diamante.csv, perla.csv}}
  corazon/{{heart_rules.yaml, weights.csv, lq.csv}}
  configs/{{allocator_sombra.yaml, diamante_selected.yaml}}
  reports/
    diamante/*  perla/*  heart/*
    allocator/
      sombra_kpis.md
      curvas_equity/{{eq_base.csv, eq_overlay.csv}}
```

---

## 9) KPIs & gates de aceptación (resumen)
- **Diamante:** PF≥1.6, WR≥60%, ≥30 trades/fold, MDD≤8% (BTC).  
- **Perla:** PF≥1.2 (rango), MDD≤15%, corr(D,P)≤0.35–0.40.  
- **Corazón (overlay):** −MDD≥15% **o** −Vol≥10% con ΔPF ≥ −5% y ΔTurnover ≤ +20%.  
- **Allocator (sombra):** vol_error <5%, PF no peor que 50/50; costos dentro de presupuesto.  
- **Live (rampa):** tracking vs sombra dentro del slippage esperado.

---

## 10) Changelog (desde 2025‑09‑01)
- Corazón v0.2 (semáforo suave + LQ + corr‑gate) documentado; sigue **en sombra**.
- Allocator v0.1‑R (ξ* freeze, CB, vol targeting, caps) validado; perfil **congelado**.
- `allocator_sombra_runner_plus.py` lee **fee_bps/slip_bps del YAML** automáticamente.
- Homologados contratos CSV/TTL y estructura de carpetas.
