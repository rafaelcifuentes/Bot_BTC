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
| 1     | ✅ Validación backtest y walk-forward OOS (2020–2025) | Hecho |
| 2     | ✅ Paper Trading con gobernanza estricta semanal | Hecho |
| 3     | 🟠 Canary: live con 10–20% capital y kill-switch | En curso |
| 4     | 🔵 Operación continua, monitoreo, A/B y gobernanza | Próximo |
| 5     | 🔒 Rollback y recuperación controlada | Planificado |
| 6     | 📚 Documentación final + reproducibilidad total | Planificado |

---

## 🔒 Módulos Opt‑In (OFF por defecto)

Los módulos listados como opt-in en esta tabla son componentes que **alteran el comportamiento operativo** del bot.  
Su activación está sujeta a validación A/B y criterios de adopción.

| Módulo                    | Objetivo principal                                               | Estado | Versión | Impacto estimado |
|---------------------------|------------------------------------------------------------------|--------|---------|------------------|
| `xb_adaptive`             | Cross‑buffer adaptativo con bandas de ATR                       | 🧪 ON (A/B) | v1.1+   | +2% a +8%        |
| `exit_atr_guardrail`      | Bloqueo de salidas si rebote está dentro de ATR                | 💤 OFF | v1.1+   | +1% a +5%        |
| `pause_after_flip_sell`   | Espera tras giro (flip) para evitar reentradas impulsivas       | 💤 OFF | v1.1+   | −1% a +2%        |
| `age_valve_exit`          | TTL para salidas (esperar N velas para confirmar)               | 💤 OFF | v1.1+   | +1% a +3%        |
| `adx_macro_dynamic`       | Umbral ADX/EMA/pendiente dinámico (por percentiles)             | 💤 OFF | v1.2    | +1% a +4%        |
| `hibernation_on_chop`     | Evita operar en rangos (ADX o slope bajo)                       | 💤 OFF | v2      | +2% a +5%        |
| `bull_hold`               | Mantener BUY en macro fuerte (EMA200 D1 + ADX ≥ 20)             | 💤 OFF | v2      | +3% a +7%        |
| `pullback_entry`          | Entrada alternativa en retroceso (ej. bollinger/ema)            | 💤 OFF | v3      | +5% a +12%       |
| `trailing_exit_bull`      | Salida trailing dinámica en bull markets                        | 💤 OFF | v3      | +3% a +10%       |
| `risk_sizing_by_score`    | Ajustar posición según score o riesgo estimado                  | 💤 OFF | v4+     | +3% a +8%        |
| `sl_tp_defensivo`         | Stop Loss defensivo basado en ATR para limitar drawdowns severos | 💤 OFF | v1.1.1 (post‑canario) | +4% a +12%        |

*`sl_tp_defensivo` será el primer módulo en validación tras el canario.*