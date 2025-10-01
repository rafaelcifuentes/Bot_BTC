# Plan Macro — Bot BTC (Actualizado)

**Fecha:** 2025-09-16  
**Propósito:** Consolidar líneas maestras, contratos y criterios de aceptación para evolucionar **Diamante–Perla–Corazón–Cerebro** sin romper producción. Enfatiza **BTC numeraire** (satoshis) como **prioridad 0**.

---

## 0) Principios rectores
- **BTC numeraire por defecto**: todos los KPIs en **NetBTC**, **CAGR_sats**, **MDD_vs_HODL**; USD es auxiliar para control de costes.  
- **KISS**: simple → medible → promovible **solo** si gana **OOS** con fricción.  
- **Freeze & reproducibilidad**: cortes semanales, UTC estricto, artefactos versionados.  
- **Gobernanza**: cambios por **A/B** y **no-regresión**; snapshots y toggles reversibles.

---

## 1) Foto actual
- **Cerebro (Allocator)**: perfil congelado; reporter alineado (diff≈0) entre script y curva.  
- **Corazón (semáforo/overlay)**: saneado; corriendo en **modo sombra** con KPIs y freeze.  
- **Perla (semanal/4h)**: edge real; filtro **EMA soft** disponible (band **0.990**, ttl **1**) en shadow o prod vía *symlink*.  
- **Diamante**: pendiente de auditoría/rediseño para cumplir gates OOS.

---

## 2) Contratos de interfaz (evitar peleas de timestamps)
- **Zona horaria:** todo **UTC**. **TTL** de señales/pesos: **4h** (stale → peso 0).  
- **Archivos (rejilla 4h; resample con ffill donde aplique):**
  - `signals/diamante.csv` → `timestamp, sD∈{-1,1}, w_diamante_raw∈[0,1], retD_btc`  
  - `signals/perla.csv`   → `timestamp, sP∈{-1,1}, w_perla_raw∈[0,1], retP_btc`  
  - `corazon/weights.csv` → `timestamp, w_diamante, w_perla` (∈[0,1], suma≈1)  
  - `corazon/lq.csv`      → `timestamp, lq_flag∈{HIGH_RISK,NORMAL}` (histéresis 2 velas)  
- **Costes operativos para backtests:** 12 bps totales (6+6).  

---

## 3) Gates de aceptación (OOS con costes)
- **Diamante** (aspiracional tras rediseño): PF ≥ 1.6, WR ≥ 60%, ≥30 trades/fold, MDD ≤ 8% (BTC).  
- **Perla**: PF ≥ 1.15–1.25, MDD ≤ 15%, corr(D,P) ≤ 0.35–0.40, NetBTC OOS > 0.  
- **Corazón (overlay en sombra)**:  −MDD ≥ 15% **o** −Vol ≥ 10% **sin** ΔPF < −5% ni ΔTurnover > +20%.

---

## 4) Riesgo & operación (perfil actual)
- **Objetivo de vol anual**: mantener el del perfil congelado (que dio uplift en NET).  
- **xi\***: cap≈1.65; **freeze** semanal; **circuit breakers** activos.  
  - CB vigentes: `vol_daily_pctl = 0.965`, `dd_day = -0.05` (si se dispara → `xi* = 1.0`).  
- **Vol targeting y clamps**: clamp razonable (p.ej. 0.5–1.2), `w_cap_total = 1.0`.  
- **Ejecución (opcional sandbox)**: `round_step=0.15`, `max_delta_weight_bar=0.15` → solo si ↓turnover y NET≈.

---

## 5) Mejoras **Nivel 1** (sin tocar código)
**Objetivo:** ROI con cambios paramétricos y toggles, sin interferir pruebas.

### 5.1 Perla — filtro EMA soft (param-only)
- **Grid ya probado**: (band, ttl) ∈ {(0.995,2), (0.990,1)}.  
- **Recomendación**: dejar **(0.990,1)** en *shadow*; activar en prod vía `reports/Allocator/perla_for_allocator.csv → symlink` si el dashboard confirma **PF≈**, **Act 85–90%**, **NetBTC↑**.  
- **Toggles**: `toggle_perla {base|ema}` normaliza cabecera `timestamp,retP_btc`.

### 5.2 Corazón — overlay y guardarraíles
- **ATR_MAX**: mini-grid **0.07–0.09**. Mantener si **ΔPF ≥ −0.05** y **MDD_ratio ≤ 0.85** con actividad útil.  
- **Corr Gate**: `threshold=0.35`, `max_penalty=0.20` (no mejoró fuera de ese punto).  
- **w_floor_on_signal**: **0.45** (mantiene aportes cuando hay señal válida).  
- **Función operativa**: `heart_monday` con *fallback* si falla el pipeline legacy; genera `*_overlay.csv` + KPIs.

### 5.3 Gobernanza & reporter
- **A/B semanal** (baseline vs overlay); promoción **no automática**.  
- **Métricas de decisión** (BTC): NetBTC, CAGR_sats, MDD_vs_HODL, PF, Act y turnover.

---

## 6) Roadmap (NOW / NEXT / LATER)
- **NOW**
  1) Mantener Cerebro congelado y Corazón en **sombra** con reportes.  
  2) Perla EMA soft en **shadow** (o prod vía symlink si gana) y tracking en dashboard.  
  3) Mini-grid **ATR_MAX 0.07–0.09**; fijar **0.08** si sostiene KPI (PF≈, MDD_ratio≤0.85, Act≈54%).  
- **NEXT**
  4) Auditoría/rediseño de **Diamante** para cumplir gates OOS.  
  5) Si Perla queda blindada OOS → **blend real** (w_diamante/w_perla) + corr gate.  
- **LATER**
  6) Fine‑tuning de costes solo si no afecta NET; tercera señal si aparece **no correlacionada**.

---

## 7) Cadencia operativa
- **Lunes**: freeze (`xi*`), corte KPIs de la semana previa; correr `heart_monday` (y grids que toquen sóloparámetros).  
- **Miércoles**: **Perla Negra** y **Diamante** (IS→OOS) en sus hilos; no bloquear Corazón.  
- **Diario (cada 4h)**: pesos de Corazón (sombra), sanity TTL e integridad de CSV.  
- **Viernes**: A/B semanal y Go/No-Go de los cambios paramétricos.

---

## 8) Criterios de promoción (blend activo)
Promueve Corazón **real** cuando (con costes, BTC):  
- **MDD cartera** ↓ ≥ 15% vs sin blend; **Vol** ↓ ≥ 10%; **PF** no cae > 5–10%; **NetBTC ≥ baseline**.  
- Corr(D,P) controlada (0.35–0.40) penalizando ≤30% a la pierna más débil (ventana 60–90d, perf 30d).

---

## 9) Estructura de carpetas (resumen)
```
project/Bot_BTC/
  signals/{diamante.csv, perla.csv}
  corazon/{heart_rules.yaml, weights.csv, lq.csv}
  reports/{diamante/*, perla/*, heart/*, allocator/*}
    allocator/curvas_equity/{eq_base.csv, eq_overlay.csv}
    heart/xi_star.txt
```

---

## 10) Decisiones recientes (extracto)
- **CB ajustado**: `vol_daily_pctl=0.965`, `dd_day=-0.05` (más protector).  
- **Corr Gate** verificado: `thr=0.35`, `max_penalty=0.20`.  
- **w_floor_on_signal=0.45`** activado.  
- **ATR_MAX grid 0.07–0.09** → mantiene PF y ↓MDD (shadow).  
- **Perla EMA soft 0.990/1**: aceptable en shadow con PF≈ y Act↑.

---

## 11) Éxito esperado
- **Corto plazo**: −MDD ≥ 15–25% **y/o** −Vol ≥ 10–20% con NetBTC ≥ baseline en overlay.  
- **Medio**: Diamante re‑aprobado OOS; blend activo reduce riesgo sin matar PF; reporter en sats.

---

> **Nota:** Este plan consolida y actualiza los dos documentos previos del plan macro y lo alinea con las prioridades ROI vigentes y la métrica en **satoshis**.


---

## 12) Checklist semanal (de bolsillo) — Go/No-Go

> **Uso:** una sola página, apta para README. Todo en **UTC**. Métricas en **BTC** (NetBTC, CAGR_sats, MDD_vs_HODL, PF, Act, Turnover).

### Lunes — Freeze + Corazón (sombra)
1) Congelar periodo y correr overlay con `heart_monday` (perfil slim):
   ```bash
   heart_monday "YYYY-MM-DD 00:00" 0.07 cg20_t035   # o grid 0.07–0.09
   ```
2) KPI de aceptación (overlay vs baseline):
   - Aceptar si **|MDD_overlay|/|MDD_base| ≤ 0.85** **y** **ΔPF ≥ −0.05**, con **Actividad útil** (no-cero) razonable.  
   - Registrar `*_vs_base.md` y snapshot en `reports/heart/`.

### Miércoles — Perla & Diamante (IS→OOS)
1) Perla: correr semanal. Mantener **EMA soft** en *shadow* con `(band, ttl) = (0.990, 1)`.
   - Activar en prod (si gana) vía *symlink* normalizado:`reports/Allocator/perla_for_allocator.csv → perla_ema_norm.csv`.
   - Volver a base cuando quieras: `toggle_perla base`.
2) Diamante: ejecutar y recoger KPIs OOS (PF, MDD, NetBTC, FPY) con costes.

### Viernes — A/B semanal y decisión
- **Corazón**: promover parámetros sólo si (con costes):  
  **MDD ↓ ≥ 15%** *o* **Vol ↓ ≥ 10%**, **PF** no cae > 5–10%, **NetBTC ≥ baseline**.  
- **Perla EMA soft**: mantener si **PF≈** base, **Actividad ≥ 85%** y **NetBTC ≥ base**.  
- Si hay empate, conservar parámetros previos; segunda semana consecutiva ganando → pasar a revisión de producción.

### Monitoreo diario (mínimo)
- Sanidad de CSVs (sin huecos/duplicados), **TTL 4h** y **lq_flag**.  
- Reporter/overlay: equity base vs overlay en `reports/allocator/curvas_equity/*`.

### Comandos rápidos
```bash
# Corazón (slim) — lunes
heart_monday "2025-09-15 00:00" 0.08 cg20_t035   # ejemplo con ATR_MAX=0.08

# Perla — toggle de fuente para Allocator (cabecera normalizada a timestamp,retP_btc)
cd reports/Allocator
ln -sfn perla_ema_norm.csv  perla_for_allocator.csv   # activar EMA soft (0.990/1)
ln -sfn perla_base_norm.csv perla_for_allocator.csv   # volver a base

# Verificación rápida
echo "[link] $(readlink perla_for_allocator.csv)"; head -n1 perla_for_allocator.csv
```

> **Pocket snippet para README**: copia la **sección 12** tal‑cual al `README.md` si quieres tenerla a mano.


# Plan Macro — referencia

El documento vivo está en **docs/plan_macro.md**.  
(Mantenemos este archivo solo como puntero para evitar confusión.)
