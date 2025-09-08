Macro Plan (cerrado) — con organización por hilos

1) Proyecto Bot BTC – Diamante (4h, breakout-swing)

Meta: edge claro en tendencia/expansión.
Gates (OOS, con costes): PF ≥ 1.6, WR ≥ 60%, ≥30 trades/fold, MDD ≤ 8% (BTC).
Entregables semanales:
	•	KPIs por bins de régimen (ADX: <15, 15–20, >20; ATR pctl; |slope EMA50|).
	•	signals/diamante.csv (UTC, 4h):
timestamp,sD,w_diamante_raw,retD_btc
	•	Reporte: reports/diamante/kpis_regimen.md + snapshot "selected".

Checklist rápido:
	•	Timestamps UTC 4h aligned
	•	Sin lookahead/leaks
	•	Turnover y costes verificados
	•	No-regresión vs "selected"

⸻

2) Proyecto Bot BTC – Perla Negra (semanal, estable/contracíclica)

Meta: amortiguar rango/bear/transiciones; baja corr con Diamante.
Gates: PF ≥ 1.2 en rango, MDD ≤ 15%, corr(D,P) ≤ 0.35–0.40 en neutro/rango.
Entregables semanales:
	•	KPIs por bins de rango/bear/transición; turno estable.
	•	signals/perla.csv (resample a 4h con ffill):
timestamp,sP,w_perla_raw,retP_btc
	•	Reporte: reports/perla/kpis_regimen.md.

Checklist rápido:
	•	Baja corr en rango
	•	Señal estable (pocas vueltas)
	•	Resample 4h correcto (ffill)

⸻

3) Proyecto Bot BTC – Corazón (semáforo suave + LQ + correlación)

Meta: pesos suaves, no ON/OFF; histéresis y gate de correlación.
Entregables semanales (modo diagnóstico, no ejecuta):
	•	corazon/weights.csv: timestamp,w_diamante,w_perla (∈[0,1], suma≈1)
	•	corazon/lq.csv: timestamp,lq_flag ∈ {HIGH_RISK,NORMAL}
	•	corazon/heart_rules.yaml (plantilla abajo)
	•	Overlay/ξ*: reports/heart/xi_star.txt (freeze lunes)

Criterios:
	•	MDD ↓ ≥ 15% o Vol ↓ ≥ 10% vs 50/50, ΔPF ≥ −5%, ΔTurnover ≤ +20%.

Plantilla corazon/heart_rules.yaml:
enable: true
timezone: "UTC"

regime:
  adx_thr: 20
  ema50_slope_thr: 0.0
  atr_pct_window: 100
  vol_pctl_low: 0.40
  dwell_bars: 6
  max_delta_weight: 0.20

weights:
  verde:    {diamante: 0.80, perla: 0.20}
  amarillo: {diamante: 0.50, perla: 0.50}
  rojo:     {diamante: 0.20, perla: 0.80}

lq:
  enable: true
  high_risk_multiplier: 0.70
  hysteresis_bars: 2

corr_gate:
  lookback_bars: 60     # 60–90 recomendado
  threshold: 0.35
  max_penalty: 0.30

freeze_xi_star:
  update_day: "Monday"
  xi_star_cap: 1.70

circuit_breakers:
  vol_daily_pctl: 0.98
  dd_day: -0.06

Checklist rápido (Corazón):
	•	Histeresis (dwell 6–8) y maxΔ 0.2/barra
	•	LQ con histéresis 2 velas (0.7x/1.0x)
	•	Corr gate activo (penaliza la pierna más débil hasta 30%)
	•	Exporta pesos y ξ*

⸻

4) Proyecto Bot BTC – Cerebro / Allocator (v0.1-R, modo sombra)

Meta: mezclar con pesos de Corazón + corr-gate + ξ* (freeze) + vol-targeting + caps.
No ejecuta órdenes en Fase 1; solo simula y reporta.

Entradas estándar:
	•	signals/diamante.csv  (4h): timestamp,sD,w_diamante_raw,retD_btc
	•	signals/perla.csv     (4h): timestamp,sP,w_perla_raw,retP_btc
	•	corazon/weights.csv   (4h): timestamp,w_diamante,w_perla
	•	corazon/lq.csv        (4h): timestamp,lq_flag
	•	reports/heart/xi_star.txt  (texto: 1.00–1.70, actualizado lunes)

Salidas esperadas:
	•	reports/allocator/sombra_kpis.md (PF, WR, MDD, Sortino, turnover, corr_dp, vol_error<5%)
	•	reports/allocator/curvas_equity/{eq_base.csv,eq_overlay.csv}

Plantilla configs/allocator_sombra.yaml:
rebalance_freq: "4h"
timezone: "UTC"

costs:
  fee_bps: 6
  slip_bps: 6
exec:
  exec_threshold: 0.02
  max_delta_weight_bar: 0.20

alloc_base:
  diamante: 0.50
  perla: 0.50

risk:
  vol_target_ann: 0.20
  vol_clamp: {min: 0.50, max: 1.20}
  w_cap_total: 1.00
  kill_switch:
    mdd_30d: -0.10   # define X si quieres más conservador

corr_gate:
  lookback_bars: 60
  threshold: 0.35
  max_penalty: 0.30

files:
  diamante: "signals/diamante.csv"
  perla: "signals/perla.csv"
  heart_weights: "corazon/weights.csv"
  heart_lq: "corazon/lq.csv"
  xi_star: "reports/heart/xi_star.txt"

outputs:
  kpis: "reports/allocator/sombra_kpis.md"
  equity_dir: "reports/allocator/curvas_equity/"


Checklist rápido (Allocator sombra):
	•	Mezcla respeta w_cap_total=1.0
	•	Vol targeting 20% (clamp 0.5–1.2)
	•	ξ* freeze lunes + circuit breaker (vol p98 o DD día ≤ −6% → ξ*=1.0)
	•	Corr gate ≤ 30% a la pierna más débil
	•	vol_error < 5%, costos dentro del presupuesto
	•	Tracking eq_overlay vs eq_base consistente

⸻

Cadencia operativa (semana tipo)
	•	Lunes: refrescar xi_star.txt; publicar KPIs de la semana previa.
	•	Mar–Jue: tests OOS y stress; verificar corr gate y costos.
	•	Viernes: snapshots "selected" y no-regresión; Go/No-Go.
	•	Cada 4h: generar corazon/weights.csv y correr Allocator en modo sombra.

⸻
Estructura de carpetas estándar

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

