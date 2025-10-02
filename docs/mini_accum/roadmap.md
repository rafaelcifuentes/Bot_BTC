[FREEZE V1] 2025-10-02 — baseline sellado
---
title: "Mini-Accum KISS — Roadmap (Canónico)"
version: "v1.0"
date: "2025-10-01"
status: "CANÓNICO"
---

# 🧭 Roadmap PDCA — Mini-Accum KISS v1
**Principio rector:** “Solo mejoramos cuando algo más simple ya no es suficiente.”

## PDCA aplicado
| Fase | Acción |
|---|---|
| **Plan** | Baseline KISS (EMA21/55 4h + macro D1>EMA200), sin módulos |
| **Do** | Wrapper, cron 4h UTC, FREEZE semanal, paper/testnet |
| **Check** | A/B semanal, KPIs, NetBTC vs HODL, SPA/RC/DSR |
| **Act** | Gate: **ΔNetBTC ≥ +0.02** sin empeorar **MDD/FPY** |

## 🚀 Roadmap de versiones
- **v1.0** → núcleo base + wrapper productivo, FREEZE y OOS  
- **v1.1** → **sl_tp_defensivo** (ATR)  
- **v2.0** → **bull_hold**, **hibernation_on_chop**, **cooldown_after_loss**  
- **v3.0** → **atr_pct_adapt**, **turnover_budget**, **reentry_buffer**, **rsi_confirmation**  
- **v4.0** → **risk_sizing_by_score**, **exit_atr_guardrail**, **dca_adaptativo** (plan)  
- **v5.0** → **pullback_entry**, **trailing_exit_bull**, **confirmations_rsi_bias**

## 🔒 Módulos Opt-In (no activados en v1)
- bull_hold
- hibernation_on_chop
- cooldown_after_loss
- sl_tp_defensivo
- atr_pct_adapt
- turnover_budget
- reentry_buffer
- rsi_confirmation
- risk_sizing_by_score
- exit_atr_guardrail
- dca_adaptativo
- pullback_entry
- trailing_exit_bull
- confirmations_rsi_bias
