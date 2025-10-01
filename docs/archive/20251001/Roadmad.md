---
title: "Mini‑Accum KISS — Hoja de Ruta (PDCA) y Módulos Opt‑In"
version: "v1.0"
date: "2025-10-01"
status: "CANÓNICO (alineado con brochure.md)"
---

# 🔮 Hoja de Ruta Futura y Módulos — Opt‑In · Mini‑Accum KISS

**Filosofía:** Añadir inteligencia **sin perder** la esencia **KISS**.  
Cada módulo se activa **solo** si demuestra impacto **positivo** neto en **NetBTC**, **drawdown** o **estabilidad** (OOS).

---

## 🧭 Roadmap PDCA — Mini‑Accum KISS v1
**Versión congelada:** `KISSv1_BASE_20250915_1642_final`  
**Objetivo:** *Sleeving* prod‑ready **sin desviarse** del núcleo esencial v0.1.  
Este roadmap define los pasos futuros bajo PDCA (**Plan‑Do‑Check‑Act**), priorizando **estabilidad**, **gobernanza** y **acumulación de sats**.

> **Principio rector:** *“Solo mejoramos cuando algo más simple ya no es suficiente.”*

**Objetivo global:** Evitar errores innecesarios y **acumular satoshis con disciplina**, no con “magia”.

---

## 📌 RESUMEN EJECUTIVO (v1 Base)
Estrategia de **rotación BTC ↔ USDC** con reglas simples y gobernanza estricta:

- **Entrada:** Cruce **EMA21 > EMA55** (4h) con **macro** D1 **> EMA200**
- **Anti‑whipsaw:** TTL mínimo entre señales
- **Posición:** 0% o 100% (sin sizing dinámico)
- **Costes modelo:** fee **6 bps** + slip **6 bps** por lado
- **Sin** SL/TP, **sin** leverage, **sin** shorts
- **Benchmark:** Comparación semanal **NetBTC vs HODL**
- **Todo lo demás** queda **opt‑in**

**PDCA aplicado**
| Fase | Acción |
|---|---|
| **Plan** | Baseline simple, sin módulos adicionales |
| **Do** | Wrapper, paper trading, alertas, cron, FREEZE semanal |
| **Check** | A/B semanal, KPIs, comparación con HODL |
| **Act** | Gate de mejora: **Δ≥+0.02** sin empeorar **MDD** o **FPY** |

---

## 📍 Estado actual (KISS v1)
| Componente | Estado |
|---|---|
| Estrategia | Cruce **EMA21/55** + **macro D1>EMA200** |
| Marco temporal | **4h** (macro en **D1**) |
| Gestión de riesgo | Posición 100% o 0%, **sin** SL/TP ni sizing adaptativo |
| Costes | **6+6 bps** por lado (stress: **10/20 bps**) |
| Outputs | `latest.json`, `live_kpis.csv`, `health.status` |
| Gobernanza | **FREEZE** semanal + **A/B** + fail rate |
| Tracking vs HODL | **✅ Activo** y visible |
| SPA / RC / DSR | **🧪 En integración** (validación estadística) |

**Notas estratégicas:** Alta prioridad inmediata → **SL/TP defensivo**, **hibernation_on_chop**, **BULL_HOLD** (blindaje + edge sin complejidad).

**Componentes en producción**
- **Ingesta:** ccxt (BinanceUS), cron 4h UTC  
- **Señales:** cruce EMA21/55, macro D1>EMA200  
- **Validación temporal:** TTL entre flips (4 velas)  
- **Outputs:** `latest.json` (posición), `live_kpis.csv` (NetBTC, FPY, delta HODL), `health.status`  
- **Gobernanza:** FREEZE semanal + A/B automático  
- **Logs / Kill‑switch:** operativo

---

## 🧭 Roadmap PDCA — Mini‑Accum KISS v2+

### 🚀 Roadmap de versiones (extendido)
- **v1** → núcleo base: EMA21>55, macro EMA200, TTL mínimo, **sin** SL/TP  
- **v2** → **disciplina**: bull_hold, cooldown_after_loss, hibernation_on_chop  
- **v3** → **eficiencia**: ATR% adaptativo (márgenes/TTL), turnover_budget, reentry_buffer, *RSI gate (confirmador suave)*  
- **v4** → **control de riesgo**: sizing dinámico, SL/TP defensivo avanzado, *DCA adaptativo (experimental)*  
- **v5** → **precisión**: pullback_entry, trailing_exit_bull, confirmations_rsi_bias

> Cualquier módulo activado **debe** pasar **OOS**, **SPA/RC**, y no degradar **NetBTC/MDD/FPY**.

---

## 🗺️ Roadmap de Evolución (tabla)
| Versión | Objetivo estratégico | Módulos principales incluidos |
|---|---|---|
| **v1** | Núcleo base KISS | EMA21/55, macro D1, TTL, rotación BTC↔USDC, sin leverage/shorts |
|  | Wrapper producción estable | Wrapper productivo, FREEZE semanal, A/B, validación OOS |
| **v2** | Disciplina y sostenibilidad | **bull_hold** (EMA200+ADX), **cooldown_after_loss**, **hibernation_on_chop** |
| **v3** | Eficiencia y robustez | **atr_pct_adapt**, **turnover_budget**, **reentry_buffer** |
| **v4** | Control de riesgo avanzado | **risk_sizing_by_score**, **SL/TP defensivo** (evolución), **DCA adaptativo** (plan) |
| **v5** | Precisión fina de ejecución | **pullback_entry**, **trailing_exit_bull**, **confirmations_rsi_bias** |

---

## 🔒 Módulos Opt‑In (No activados en v1)
| Módulo | Objetivo Principal | Estado | Versión futura | Impacto estimado NetBTC |
|---|---|---:|:---:|---:|
| **bull_hold** | Mantener posición en bull markets (EMA200 + ADX≥20) | 😴 OFF | v2 | **+3% a +9%** |
| **cooldown_after_loss** | Evitar reentradas impulsivas tras pérdidas | 😴 OFF | v2 | **+1% a +3%** |
| **hibernation_on_chop** | Evitar operar en rangos (ADX bajo / slope plana) | 😴 OFF | v2 | **+6% a +14%** |
| **atr_pct_adapt** | SL/TP adaptativos vía ATR | 😴 OFF | v3 | **+2% a +5%** |
| **turnover_budget** | Penalizar rotación excesiva | 😴 OFF | v3 | **+2% a +4%** |
| **reentry_buffer** | TTL extendido tras salida reciente | 😴 OFF | v3 | **+1% a +2%** |
| **rsi_confirmation** | Sesgo suave (ej. RSI>50) | ❌ OFF | v3 | **0% a +3%** |
| **risk_sizing_by_score** | Tamaño variable por score/vol | 😴 OFF | v4 | **+3% a +8%** |
| **dca_adaptativo** | Aumentar en retrocesos fuertes | ❌ OFF | v4/v5 | **+5% a +10%** (alta varianza) |
| **exit_atr_guardrail** | Bloqueo de salida en rango ATR | 😴 OFF | v4 | **+2% a +5%** |
| **sl_tp_defensivo** | Stop‑loss defensivo (ATR) | 😴 OFF | v1.1.1 | **+4% a +12%** |
| **pullback_entry** | Entrada en retroceso (EMA/Bollinger) | 😴 OFF | v5 | **+2% a +5%** |
| **trailing_exit_bull** | Salida escalonada en bull runs | 😴 OFF | v5 | **+3% a +7%** |
| **confirmations_rsi_bias** | Freno extra en entradas débiles | 😴 OFF | v5 | **+1% a +3%** |

> **Nota:** Se corrigieron erratas del borrador original y se normalizaron rangos con el baseline actual.

---

## 📈 Estimación de ROI Anual Esperado — Mini‑Accum KISS
**Referencia canónica:** baseline **v1.0** validado ≈ **32.7%** anual (rango operativo **~30–35%**).  
Rangos indicativos (NetBTC), con **costes 6+6 bps/lado** y **FPY** dentro de presupuesto.

| Versión | Módulos activos principales | ROI anual (NetBTC) | FPY estimado | MDD esperada | Observaciones |
|---|---|---:|:---:|:---:|---|
| **v1 (baseline)** | EMAs, macro filter, TTL, rotación BTC↔USDC | **~30–35%** | ~18–22 | DD16–DD20 | Sistema robusto y simple; ya supera HODL |
| **v1.1** | + SL/TP defensivo (ATR) | **~32–38%** | 20–24 | ↓MDD −10% a −20% | Protección ante caídas abruptas |
| **v2** | + BULL_HOLD + hibernation_on_chop + cooldown | **~36–44%** | 16–22 | ↓MDD −35% a −50% | Mayor salto ajustado a riesgo |
| **v3** | + trailing_exit + turnover_budget + pullback + reentry | **~40–52%** | 14–20 | DD13–DD17 | Menos whipsaw, mejor timing |
| **v4** | + risk sizing + ATR% adaptativo | **~40–55%** | 12–18 | Ajustable | Requiere rigor anti‑overfit |
| **v5** | + DCA adaptativo + confirmaciones RSI | **~40–55%** (techo) | 10–16 | Ultra optimizado | Activar solo con evidencia sólida |

**Observaciones de consolidación**
- La curva de ROI es **no lineal**; hay **sinergias**, no suma directa.  
- **v2** suele aportar el mayor salto **ajustado a riesgo**.  
- **v3–v5** priorizan **calidad de ejecución** y **estabilidad**.

---

## 🧪 Validación estadística (SPA / RC / DSR / PBO)
> No son módulos de ejecución, sino **herramientas de validación**.

| Herramienta | ¿Qué es? | ¿Para qué sirve? | Estado |
|---|---|---|:---:|
| **SPA** (Superior Predictive Ability) | Test contra benchmarks nulos | Verificar ventaja no aleatoria | 🟡 |
| **Reality Check (White)** | Variante conservadora del SPA | Corregir por *multiple testing* | 🟡 |
| **DSR** (Deflated Sharpe Ratio) | Ajuste de Sharpe por múltiples pruebas | Evitar falsos positivos | ✅ |
| **PBO (CSCV)** | Probabilidad de sobreajuste | Robustez OOS | ✅ |

**Política de adopción de cambios**  
- Si **dos iteraciones** seguidas **no mejoran** bajo SPA/RC/DSR → **archivar** la palanca.  
- Datos **“pinned”** y ventanas OOS fijas (H2‑2024, Q1‑2025).  
- **Un cambio por commit** y reversión inmediata si **no** supera baseline.  
- Costes **fijos** durante toda la serie de pruebas.

---

## 🛡️ Criterios de adopción por versión
- ✅ **NetBTC ≥ baseline**  
- ✅ **MDD ≤ baseline + 0.05**  
- ✅ **FPY ≤ 26/año**  
- ✅ **SPA / Reality Check:** no rechazo al 5–10%  
- ✅ **DSR** positivo y estable en todas las ventanas  
- ✅ **Reproducibilidad** y estabilidad OOS  
- ✅ **Docs** claros y versionamiento controlado  
- ✅ **Mejora significativa** vs la versión anterior  
- ✅ **Drawdowns** dentro de límites aceptables

---

## 📁 Documentación relacionada
- `docs/Progreso.md` → Estado semanal y KPIs  
- `docs/mini_accum/BULL_HOLD.md` → Reglas y activación  
- `reports/mini_accum/walkforward/` → Freezes y comparativas  
- `configs/mini_accum/overlays/` → YAMLs opt‑in por módulo

---

## 🧭 Notas de ejecución (v1.1 → v2)
- No requieren hiperparámetros complejos.  
- No alteran la lógica base; **modulan** según el régimen.  
- Respetan 100% la filosofía **KISS**.  
- Preparan terreno para **A/B** limpio.

**Plan inmediato (1.1 → 2.0):**
- `SL/TP defensivo` (ATR×2.5) — **v1.1** (post‑canario)  
- `hibernation_on_chop` (ADX D1 < 20) — **v2**  
- `BULL_HOLD` (D1>EMA200 & >EMA50) — **v2**

---

**Hoja de ruta oficial KISS (1.1 a 2.0)** — *Plan de Implementación Modular*
| Módulo KISS | Objetivo estratégico | Implementación sugerida | Versión | Est. impacto NetBTC |
|---|---|---|:---:|---:|
| **SL/TP defensivo** | Protección flash‑crash | StopLoss dinámico **ATR(14) × 2.5** | ✅ v1.1 | **+4% a +12%** |
| **hibernation_on_chop** | Evitar operar en lateralidad | **ADX(D1) < 20** = Hibernación | ✅ v2 | **+5% a +10%** |
| **BULL_HOLD** | Capturar bull runs | D1>EMA200 y D1>EMA50 → desactiva salida activa | ✅ v2 | **+2% a +6%** |

---

### Cierre
> **No complicamos la base** hasta que algo más simple **ya no** sea suficiente.  
> KISS primero. Opt‑ins después. **Sats** siempre.
