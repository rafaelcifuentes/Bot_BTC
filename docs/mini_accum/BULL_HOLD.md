# BULL_HOLD (opt‑in) — Runbook
**Estado:** módulo opt‑in (OFF por defecto; no baseline) · **Última actualización:** 2025-10-01

**Objetivo:** en bull fuerte, reducir rotación inútil y capturar más tramo de tendencia **sin romper KISS**.

---

## 1) Regla operativa (KISS)
**Mantener BUY** cuando el **macro D1 es fuerte** y las señales 4h **no aportan edge claro**.

**Condición de bull fuerte (D1):**
- `close_D1 > EMA200_D1` **y** `ADX14_D1 ≥ 20` (Wilder).  
  *(Opcional, modo “estricto”)*: `close_D1 > EMA50_D1`.

**Efecto en la lógica 4h cuando BULL_HOLD = ON:**
- **Desactivar _salida activa_** por cruce bajo de EMA21 (evita “whipsaw”).  
- **Seguir permitiendo salida pasiva solo si el macro se degrada** (p.ej., `close_D1 < EMA200_D1` con histeresis) o si aplica **SL/TP defensivo** (si el módulo está activo).

> Regla de sobriedad: BULL_HOLD **no fuerza nuevas entradas**. Solo **extiende** una posición BUY ya abierta.

---

## 2) Guardarraíles
- **Sin leverage** por defecto. Si se habilita LEV como opt‑in en el futuro → **tamaño pequeño** y **stop operativo** obligatorio.
- **Histeresis macro:** no apagar/encender con un solo día. Exigir **2 días** consecutivos fuera del umbral para **desactivar**.
- **Ventana mínima de sostén:** {24–48} velas 4h desde la activación para evitar parpadeo (siempre que el macro siga fuerte).
- **Respeto al flip‑budget:** BULL_HOLD **no puede** incrementar FPY por encima del presupuesto; si lo hace, **rollback**.

---

## 3) Activación y desactivación
**Activación (manual, documentada):**
- Registrar **fecha/hora (UTC)**, **commit/tag**, ventanas afectadas y parámetros.
- Adjuntar razón (ej.: “macro D1 fuerte y whipsaw reciente por 21/55”).

**Desactivación (automática o manual):**
- Automática si: `ADX14_D1 < 18` **o** `close_D1 < EMA200_D1` durante **2 días consecutivos**.  
- Manual si: aumento de **FPY**, **MDD** o **tracking error** vs baseline fuera de tolerancia.

---

## 4) Parámetros sugeridos (overlay YAML)
```yaml
modules:
  bull_hold:
    enabled: false           # opt‑in
    adx_period_d1: 14
    adx_min_on: 20           # umbral de activación
    adx_min_off: 18          # umbral de salida (histeresis)
    require_above_ema50: false
    min_hold_bars_4h: 24     # evita parpadeo (si macro sigue fuerte)
    disable_exit_active: true
    allow_passive_exit_on_macro_flip: true
```

---

## 5) KPIs mínimos y Gates de adopción
- **sats_mult ≥ 1.00** (vs HODL) en ventanas OOS fijadas.
- **MDD_vs_HODL ≤ 1.00** (igual o menor drawdown que HODL).
- **FPY ≤ 26/año** (y **≤ 2/mes** soft cap). Ideal: FPY **igual o menor** que baseline.
- **ΔNetBTC ≥ +0.02** (si no mejora MDD/FPY, no asciende).
- **SPA / Reality Check:** no rechazo al 5–10% · **DSR** positivo.

---

## 6) Protocolo de pruebas (A/B + OOS)
- **Datos y costes:** mismo set OOS y **6+6 bps** por lado.
- **Ventanas OOS canónicas:** `2022H2`, `2023Q4`, `2024H1` (y `2025H1` si aplica).  
- **A/B semanal:** baseline vs baseline+`bull_hold` con **FREEZE**.  
- **Criterio de promoción:** 2 cortes consecutivos con **NetBTC↑** (≥+0.02) **y/o** **MDD↓**, sin romper FPY → **promover a v2**.

---

## 7) Observabilidad y run‑time
- Publicar en `live_kpis.csv` las columnas: `bull_hold_state`, `bull_hold_since_ts`, `adx_d1`, `macro_flag_d1`, `reason_code`.
- Logs: `logs/mini_accum/bull_hold.log` con **activaciones**, **desactivaciones** y **motivos**.
- Alertas: notificar por **flip** suprimido por BULL_HOLD (para auditar el “whipsaw evitado”).

---

## 8) Riesgos conocidos
- **Sobre‑permanencia** en alzas agotadas si ADX cae rápido → mitigado por **histeresis de salida** y/o **SL/TP defensivo**.
- **Sesgo optimista** en chops de baja vol → combinar con `hibernation_on_chop` (v2).
- **Aumento de exposición** sostenida → vigilar **mdd_vs_hodl** y **rolling drawdown**.

---

## 9) Rollback (si falla)
- **Revert inmediato** al baseline (commit anterior) si MDD/FPY se desborda o SPA/RC rechaza.  
- Registrar ticket de post‑mortem con: ventanas, métricas, razón de fallo y decisión.

---

## 10) Estado y enlace a Roadmap
- **Estado:** opt‑in **OFF** (hasta pasar A/B + OOS).  
- **Roadmap:** ver `docs/mini_accum/roadmap.md` (v2 → disciplina).

---

### TL;DR
BULL_HOLD mantiene la posición **cuando el bull es claro**, apaga la salida activa de 4h y deja que el **macro D1 gobierne** la permanencia, con **histeresis** y **guardarraíles** para no sobre‑exponerse.

