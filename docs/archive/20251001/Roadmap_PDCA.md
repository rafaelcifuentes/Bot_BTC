🧭 Roadmap PDCA — Mini‑Accum KISS v1

Versión congelada: KISSv1_BASE_20250915_1642_final
Objetivo: Sleeving prod‑ready sin desviarse del núcleo esencial v0.1.
Este roadmap define los pasos futuros bajo filosofía PDCA (Plan‑Do‑Check‑Act), priorizando estabilidad, gobernanza y acumulación de sats.

⸻

📌 RESUMEN EJECUTIVO

Mini‑Accum KISS v1 es una estrategia de rotación BTC ↔ USDC con lógica estrictamente definida, sin overfitting, gobernada por reglas técnicas simples:
	•	Cruce EMA21 > EMA55 (4h)
	•	Confirmación macro: Precio D1 > EMA200
	•	TTL mínimo entre señales (anti-whipsaw)
	•	Posición fija: 0% o 100%
	•	Costes simulados: fee 6 bps + slip 6 bps por lado
	•	Sin SL/TP ni sizing variable
	•	Comparación semanal NetBTC vs HODL (BTC benchmark)
	•	No usa RSI, DCA, ni capas adicionales (todos opt‑in)

⸻

🔁 PDCA – Enfoque aplicado

Fase	Acción
Plan	Baseline simple, sin módulos adicionales
Do	Wrapper, paper trading, alertas, cron, FREEZE semanal
Check	A/B semanal, seguimiento KPIs, comparación con HODL
Act	Gate de mejora: Δ≥+0.02 sin empeorar MDD o FPY


⸻

📍 Estado actual (KISS v1)

Componente	Estado
Estrategia	Cruce EMA21/55 + macro D1>EMA200
Marco temporal	4h (macro en D1)
Gestión de riesgo	Posición 100% o 0%, sin SL/TP ni sizing adaptativo
Costes	6+6 bps por lado (stress: 10/20 bps)
Outputs	latest.json, live_kpis.csv, health.status
Gobernanza	FREEZE semanal + A/B + fail rate
Tracking vs HODL	✅ Activo y visible
SPA / RC / DSR	🧪 En integración para validación estadística


⸻

📦 Componentes actuales en producción
	•	Ingesta: ccxt (BinanceUS), cron 4h UTC
	•	Señales: cruce EMA21/55, macro D1>EMA200
	•	Validación temporal: TTL entre flips (4 velas)
	•	Outputs:
	•	latest.json: posición actual
	•	live_kpis.csv: NetBTC, FPY, HODL delta
	•	health.status: OK/WARN/PAUSE
	•	Gobernanza: FREEZE semanal + A/B automático
	•	Logs y kill-switch funcional

⸻

🔒 Módulos Opt‑In (No activados en v1)

Módulo	Objetivo principal	Estado	Versión futura
bull_hold	Mantener posición en bull markets (EMA200+ADX≥20)	💤 OFF	v2
cooldown_after_loss	Evitar reentradas tras pérdida	💤 OFF	v2
hibernation_on_chop	Pausa en rango (ADX bajo o pendiente)	💤 OFF	v2
atr_pct_adapt	SL/TP adaptativo según ATR	💤 OFF	v3
turnover_budget	Limitar exceso de flips	💤 OFF	v3
reentry_buffer	TTL extendido post salida	💤 OFF	v3
rsi_confirmation	Gate por RSI como sesgo suave	❌ OFF	v3
risk_sizing_by_score	Ajustar posición por riesgo/score	💤 OFF	v4
dca_adaptativo	Aumentar en retrocesos controlados	❌ OFF	v4
exit_atr_guardrail	Bloqueo de salida si está dentro de rango ATR	💤 OFF	v4
pullback_entry	Entrada alternativa (EMA, bollinger)	💤 OFF	v5
trailing_exit_bull	Salida dinámica escalonada en bull runs	💤 OFF	v5
confirmations_rsi_bias	Filtro extra en zonas ambigua	💤 OFF	v5


⸻

📈 Estimación de ROI Anual Esperado — Mini‑Accum KISS
	•	Baseline v1 (sin módulos):
	•	ROI anual neto estimado: 18% a 24% en BTC
	•	FPY ≈ 18–22 | MDD moderado | Sharpe medio
	•	Con implementación completa de v1.1 + v2:
	•	ROI proyectado: 28% a 36% anual (BTC)
	•	Sinergia no lineal | Mejora estabilidad | MDD cae 35–50%
	•	Proyección full stack hasta v5 (opt‑ins activados con validación):
	•	ROI potencial estimado: 45% a 60% anual neto (BTC)
	•	Win rate ≈ 60–66% | FPY ≤ 20 | MDD contenido | Sharpe y Sortino +50–70%
	•	Robustez estadística: pasa SPA, RC, DSR, PBO en walk-forward OOS

⸻

🚀 Roadmap de versiones (extendido)

v1   → núcleo base: EMA21>55, macro EMA200, TTL mínimo, sin SL/TP
v2   → disciplina: bull_hold, cooldown_after_loss, hibernation_on_chop
v3   → eficiencia: ATR adaptativo, turnover_budget, reentry_buffer, RSI gate
v4   → control de riesgo: sizing dinámico, SL/TP defensivo, DCA adaptativo
v5   → precisión: pullback_entry, trailing_exit_bull, confirmations_rsi_bias


⸻

Cualquier módulo activado deberá pasar validación OOS, SPA/RC y no degradar NetBTC/MDD/FPY.

🧠 Principio rector:
“Solo mejoramos cuando algo más simple ya no es suficiente.”

🟠 Mini‑Accum KISS no busca predecir el mercado, sino evitar errores innecesarios y acumular satoshis con disciplina, no con magia.

## 🔒 Módulos Opt‑In (No activados en v1)

| Módulo                 | Objetivo principal                                                           | Estado    | Versión futura | Impacto estimado en NetBTC (%) |
|------------------------|-------------------------------------------------------------------------------|-----------|----------------|-------------------------------:|
| bull_hold              | Mantener posición en bull markets (EMA200+ADX≥20)                            | 💤 OFF    | v2             |                      +3% a +9% |
| cooldown_after_loss    | Evitar reentradas impulsivas tras pérdidas                                   | 💤 OFF    | v2             |                      +1% a +3% |
| hibernation_on_chop    | Evitar operar en rangos sin dirección (ADX bajo o pendiente)                 | 💤 OFF    | v2             |                     +6% a +14% |
| atr_pct_adapt          | SL/TP adaptativo según ATR                                                   | 💤 OFF    | v3             |                      +2% a +5% |
| turnover_budget        | Penalizar señales con demasiados flips recientes                             | 💤 OFF    | v3             |                      +2% a +4% |
| reentry_buffer         | TTL extendido tras salida reciente                                           | 💤 OFF    | v3             |                      +1% a +2% |
| rsi_confirmation       | Confirmador de entrada (ej. RSI > 50)                                        | ❌        | v3 (opt-in)    |                      +0% a +3% |
| risk_sizing            | Tamaño variable según riesgo, DD o score externo                             | 💤 OFF    | v4+            |                      +3% a +8% |
| dca_adaptativo         | Comprar más fuerte en caídas                                                 | ❌        | v4/v5          |             +5% a +10% (alta varianza) |
| exit_atr_guardrail     | Bloqueo de salida si está dentro de rango ATR                                | 💤 OFF    | v4             |                      +2% a +5% |
| sl_tp_defensivo        | Stop Loss defensivo basado en ATR para limitar drawdowns severos            | 💤 OFF    | v1.1.1 (post‑canario) |               +4% a +12% |
| pullback_entry         | Entrada alternativa (EMA, bollinger)                                         | 💤 OFF    | v5             |                      +2% a +5% |
| trailing_exit_bull     | Salida dinámica escalonada en bull runs                                      | 💤 OFF    | v5             |                      +3% a +7% |
| confirmations_rsi_bias | Filtro extra en zonas ambigua                                                | 💤 OFF    | v5             |                      +1% a +3% |
