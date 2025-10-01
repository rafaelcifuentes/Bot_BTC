# hibernation_on_chop (opt-in) — Runbook
**Estado:** módulo opt-in (OFF por defecto; no baseline) · **Última actualización:** 2025-10-01

**Objetivo:** Pausar operaciones en **rangos sin dirección** para **reducir whipsaw** y preservar FPY/MDD.

---

## 1) Regla operativa (KISS)
**Activar Hibernación** cuando el mercado esté en lateralidad (**chop**).

**Detección de chop (D1 o 4h):**
- **ADX14_D1 < 20** *(umbral por defecto)*, **o**
- |pendiente(EMA55_4h)| < ε (p.ej., 0.05% por vela) durante N velas,
- *(opcional)* rango verdadero medio (ATR%) < p40 del histórico reciente.

**Efecto cuando hibernation_on_chop = ON:**
- No se abren **nuevas posiciones**.
- Se permiten **cierres** según lógica base/SL para evitar quedarse atrapado en rangos.

---

## 2) Guardarraíles
- **Histeresis:** salir de hibernación solo si **ADX14_D1 ≥ 22** (no parpadeo).
- **Time-cap:** evaluar salida de hibernación cada **24–48** velas 4h.
- **Respeto FPY:** la hibernación **debe disminuir** (o mantener) FPY vs baseline.

---

## 3) Activación / Desactivación (documentadas)
- **Activación:** registrar UTC, commit/tag, ventanas afectadas, umbrales usados.
- **Desactivación:** automática si ADX supera umbral + histeresis; manual si afecta negativamente NetBTC.

---

## 4) Parámetros sugeridos (overlay YAML)
```yaml
modules:
  hibernation_on_chop:
    enabled: false
    adx_period_d1: 14
    adx_max_on: 20         # entrar en hibernación si ADX cae por debajo
    adx_min_off: 22        # salir de hibernación si ADX recupera
    ema_slope_eps_4h: 0.0005   # 0.05% por vela
    ema_slope_lookback: 24
    atr_pct_quantile_cutoff: 0.4
    enforce_no_new_entries: true
    allow_exits_only: true
```

---

## 5) KPIs mínimos / Gates
- **FPY ↓** o = baseline; **MDD ↓** o = baseline.
- **ΔNetBTC ≥ +0.02** *o* mejora material en MDD/FPY.
- **SPA/RC** no rechaza al 5–10% · **DSR** positivo.

---

## 6) Protocolo de pruebas
- Mismas ventanas OOS y costes del baseline.
- A/B semanal con FREEZE.
- Promoción a v2 si **2 cortes** consecutivos cumplen gates.

---

## 7) Observabilidad
- `hibernation_state`, `hibernation_since_ts`, `adx_d1`, `ema_slope_4h`, `reason_code` en `live_kpis.csv`.
- Log dedicado: `logs/mini_accum/hibernation.log`.

---

## 8) Riesgos y mitigaciones
- **Falsos negativos** (no detectar chop) → combinar ADX + slope EMA + ATR%.
- **Salidas prematuras** del chop → histeresis y ventana mínima ON.
- **Oportunidad perdida** si ADX sube rápido → histeresis moderada + evaluación periódica.

---

## 9) Rollback
- Revert al baseline si FPY/MDD empeoran o SPA/RC rechaza.

---

## 10) Estado / Roadmap
- **Estado:** OFF (opt-in).  
- **Roadmap:** `docs/mini_accum/roadmap.md` (v2: disciplina).
