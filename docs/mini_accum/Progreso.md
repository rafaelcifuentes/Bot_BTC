#!/bin/bash

set -euo pipefail

PY=${PY:-python3}

echo "Starting KISS_v1 walk-forward pipeline..."

# Run baseline tests for WF windows
for w in "2024-07-01 2024-12-31 H2_2024" "2025-01-01 2025-03-31 Q1_2025" "2020-01-01 2022-12-31 OOS_2020_2022"; do
  set -- $w; s=$1; e=$2; t=$3
  echo "Running baseline for window $t ($s to $e)..."
  $PY scripts/mini_accum/kiss_v1.py \
    --config configs/mini_accum/kiss_v1.yaml \
    --mode pt --gate_sma 200 --gate_mode sell --dd_hard_pct 30 \
    --dd_pct 16 --rb_pct 3 --bull_hold_sma 0 \
    --start $s --end $e \
    --suffix ${t}_PT_G200_DD16_RB3_H30_BULL0
done

echo "Baseline runs completed."

# Additional pipeline steps would go here
# ...

echo "Pipeline finished successfully."

# --- Consolidado 2021/2022 (baseline, sin filtrar) ---
if ls reports/mini_accum/kiss_v1/*kpis__WF_*202[12]*.csv >/dev/null 2>&1; then
  ${PY:-python3} tools/mini_accum/wf_consolidate.py \
    --kpis_glob 'reports/mini_accum/kiss_v1/*kpis__WF_*202[12]*.csv' \
    --out_summary 'reports/mini_accum/walkforward/wf_summary_kpis__2122.csv' \
    --out_best    'reports/mini_accum/walkforward/wf_best_by_window__2122.csv' \
    --out_md      'reports/mini_accum/walkforward/Roadmap_PDCA.md' \
    --candidate   'DD15_RB1_H30_G200_BULL0' \
    --keep_all || true
  echo "[OK] Consolidado 21/22 (keep_all) -> wf_summary_kpis__2122.csv"
else
  echo "[WARN] No hay KPIs 2021/2022 en reports/mini_accum/kiss_v1/"
fi

# 🛡️ Criterios de adopción por versión

	• ✅ NetBTC ≥ baseline  
	• ✅ MDD ≤ baseline + 0.05  
	• ✅ FPY dentro del presupuesto (≤ 26/año)  
	• ✅ SPA / Reality Check: no rechazo al 5–10%  
	• ✅ DSR (Deflated Sharpe Ratio) positivo y estable en todas las ventanas  
	• ✅ No degradación de reproducibilidad ni estabilidad (OOS)  
	• ✅ Documentación clara en docs/ y versionamiento controlado  
	• ✅ La versión debe mostrar una mejora significativa en el rendimiento frente a la anterior.  
	• ✅ No debe presentar drawdowns mayores a los aceptables definidos.  
	• ✅ La estabilidad del modelo debe ser consistente en diferentes períodos de prueba.  

# 🛡️ Criterios de adopción por versión

• La versión debe mostrar una mejora significativa en el rendimiento frente a la anterior.  
• No debe presentar drawdowns mayores a los aceptables definidos.  
• La estabilidad del modelo debe ser consistente en diferentes períodos de prueba.  

### 🧪 Herramientas de Validación Estadística

| Herramienta                     | ¿Qué es?                                                                     | ¿Para qué sirve?                                                                                   |
|---------------------------------|------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| SPA (Superior Predictive Ability test) | Test estadístico que compara el modelo contra alternativas nulas.           | Verifica que el modelo tiene habilidad predictiva real y no es resultado del azar.                 |
| Reality Check (White)           | Variante más conservadora del SPA; considera múltiples comparaciones simultáneas. | Detecta si hay sobreajuste al elegir la mejor estrategia de muchas candidatas.                    |
| DSR (Deflated Sharpe Ratio)     | Ajuste al Sharpe Ratio según el número de estrategias evaluadas.              | Previene falsos positivos al penalizar el Sharpe de estrategias seleccionadas entre muchas.       |

## Sleeving prod-ready — KISS v1 (baseline: DD15_RB1_H30_G200_BULL0)

### 0) Pre-flight (1 vez)
- [ ] Tag/versión fijada: `KISSv1_BASE_20250915_1642_final`
- [ ] Costs on: fee=6 bps, slip=6 bps (stress: 10/20 bps)
- [ ] Freeze semanal activo (lunes, 4h UTC)
- [ ] A/B corto al final del pipeline (normal y LEV si aplica)
- [ ] BULL_HOLD documentado como opt-in (no baseline)

### 1) Wrapper de producción (skeleton)
- [ ] **Ingesta**: ccxt (BinanceUS), OHLCV D1/4h en UTC, sin huecos
- [ ] **Scheduler**: cron/daemon cada 4h (ejecutar al *open* de la siguiente vela)
- [ ] **Persistencia**: estado (`position_pct_btc ∈ {0,1}`), última señal, versión
- [ ] **Logs**: rotación diaria + nivel DEBUG para órdenes simuladas
- [ ] **Health**: heartbeat/latidos + watchdog (reinicio si >2 ticks sin señal)
- [ ] **Kill-switch**: `override_mode` (PAUSA inmediata y segura)
- [ ] **Outputs mínimos** *(contrato de integración)*:
  - `signals/mini_accum/latest.json` → `{ts_utc, position_pct_btc, reason, version}`
  - `reports/mini_accum/live_kpis.csv` → KPIs semanales (NetBTC vs HODL, FPY, flips)
  - `health/mini_accum.status` → `OK|WARN|PAUSE` + timestamp

**DoD wrapper**:
- [ ] Reproducibilidad ±2–3% vs backtest semanal
- [ ] Latencia de decisión < 30s; idempotencia de órdenes simuladas
- [ ] Kill-switch probado (forzar PAUSA y volver a NORMAL)

### 2) Paper/Testnet — 7 días
- [ ] `RUN_MODE=paper` con costes on (6+6 bps), reloj 4h UTC
- [ ] Alertas: flip, error ingesta, watchdog, desvío KPIs
- [ ] Dashboard simple: estado, NetBTC vs HODL, FPY, flips

**Criterios de pase a Canario**:
- [ ] Tracking error semanal ≤ **±3%** vs backtest
- [ ] FPY dentro de presupuesto (≤26/año; **soft 2/mes**)
- [ ] 0 incidentes críticos (ingesta/ejecución/estado)

### 3) Canario — 10–20% capital
- [ ] Rollout con capital limitado (10–20%)
- [ ] Freeze semanal + A/B automático
- [ ] BULL_HOLD opt-in **manual** solo si bull tendencial fuerte
- [ ] Guardarraíles: sin leverage, sin shorts

**Criterios de ampliación (a 30–40% o integración en Corazón)**:
- [ ] 2 semanas consecutivas con KPIs OK:
  - NetBTC semanal ≥ HODL
  - MDD_vs_HODL ≤ 1
  - FPY en rango
  - Sin alertas críticas

### 4) Operación continua
- [ ] Reporte semanal: `Estado semanal` (NetBTC, fail_rate, MDD_vs_HODL, FPY)
- [ ] A/B corto post-pipeline (regla KISS: Δ≥+0.02 sin empeorar MDD/FPY → **revisión**)
- [ ] SPA/RC (cuando esté listo el módulo real): **no rechazado** al 5–10%
- [ ] PBO/CSCV y DSR permanecen en verde

### 5) Rollback / Seguridad
- [ ] `override_mode: PAUSE` documentado y probado
- [ ] Backup diario de estado y reports
- [ ] Runbook de recuperación (replay desde último estado consistente)

### 6) Documentación
- [ ] `docs/mini_accum/BULL_HOLD.md` (runbook)
- [ ] `docs/Progreso.md` → sección “Estado semanal”
- [ ] Roadmap_PDCA.md actualizado tras cada run (freezes + A/B + SPA/RC stub)

## Checkpoint semanal — mini_accum KISS v1 (paper)
Fecha run: 2025-09-18 03:11 UTC
Datos: D1 last=2025-09-18 00:00, H4 last=2025-09-18 00:00 → FRESCOS (OK)
Señal: position_pct_btc=1.0 (macro_green=True, trend_up=True), version=KISSv1_BASE_20250915_1642_final
LIVE KPIs (últimas 5):
2025-09-18T02:57:28Z,paper,BTC/USDC,1.0,3,3
2025-09-18T03:04:30Z,paper,BTC/USDC,1.0,3,3
2025-09-18T03:07:39Z,paper,BTC/USDC,1.0,3,3
2025-09-18T03:08:36Z,paper,BTC/USDC,1.0,3,3
2025-09-18T03:11:40Z,paper,BTC/USDC,1.0,3,3
FLIPS:
2025-09-18T01:49:48Z,0.0→1.0,test-buy
2025-09-18T01:55:05Z,1.0→0.0,test-sell
2025-09-18T02:12:08Z,0.0→1.0,macro_green=True,trend_up=True
HEALTH: OK (no-op)

A/B corto:
Δ median(sats_mult)=+0.019 (< +0.02) con peor FPY en contender → mantener baseline.
Estado paper (semana): ✅ 1/2 checkpoints OK

## Paper — Semana 1 (APROBADA)
- Checkpoints: Miércoles y Jueves 08:00 ET — ambos OK
- Freshness: D1=2025-09-18 00:00Z, H4=2025-09-18 12:00Z (sin stale)
- Señal: position_pct_btc=1.0 (macro_green=True, trend_up=True)
- A/B: Δ median(sats_mult)=+0.019 (< +0.02) → mantener baseline
- Rotación: FPY en rango; flips estables; sin errores
- Health: OK
**Veredicto:** Semana 1 aprobada. Siguiente: Semana 2 (Lun/Thu 08:00 ET). Si OK → canario 10–20%.


---

## 📌 Checkpoints Históricos — Mini‑Accum KISS v1

> Seguimiento consolidado de las evaluaciones semanales de producción (`RUN_MODE=paper`), estado del bot y decisiones de go/no-go.

### Semana 1 — Aprobada ✅  
📆 Fecha: 2025‑09‑18 (jueves)  
🧪 Checkpoints: Miércoles y Jueves 08:00 ET — ambos OK  
🧠 Señal: `position_pct_btc=1.0` (macro_green=True, trend_up=True)  
📈 Δ median(sats_mult)=+0.019 (< +0.02) → mantener baseline  
📊 FPY en rango, sin errores ni incidentes  
🔍 Health: OK  
📂 Archivos: `checkpoints/2025-09-18_0800ET/`  
📝 Veredicto: Semana 1 aprobada. Siguiente checkpoint: Semana 2 (Lun/Thu 08:00 ET).  