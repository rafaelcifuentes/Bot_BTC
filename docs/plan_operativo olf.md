# Plan operativo (6+2 semanas) — **Diamante primero**, **Corazón después**

**Estado:** Di# BTC
EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf \
  --symbol BTC-USD --period 730d --horizons 30,60,90 \
  --freeze_end "2025-08-05 00:00" \
  --out_csv reports/diamante_btc_week1.csv

# ETH
EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf \
  --symbol ETH-USD --period 730d --horizons 30,60,90 \
  --freeze_end "2025-08-05 00:00" \
  --out_csv reports/diamante_eth_week1.csv

# SOL
EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf \
  --symbol SOL-USD --period 730d --horizons 30,60,90 \
  --freeze_end "2025-08-05 00:00" \
  --out_csv reports/diamante_sol_week1.csvamante en curso · Corazón **AGENDADO** (no ejecutar en paralelo).  
**Premisa:** Mantener **V2 (Perla Negra)** intocable; benchmarks solo se registran.

---

## Alcance y orden`

1) **Fase A — Diamante (semanas 0–6)**  
   Swing 4h con ATR (SL/TP1/TP2 + parcial), validación OOS, costes realistas y guardrails.

2) **Fase B — Corazón (semanas C1–C2, después de Diamante)**  
   Intradía 4h por **probabilidad + gates (ADX/FG/Funding)**, **sin** TP/SL ATR, con **snapshot diario automatizado**.
`
> **Regla:** Durante Fase A **NO** se ejecutan tareas de Corazón.

---

## Fase A — Diamante (swing_4h_forward_diamond)

### 1) Objetivo
- Validar **estabilidad** (PF, WR, MDD, Sortino) y **competitividad** con costes realistas.
- Medir desempeño semanal vs **B&H**.
- Decidir en semanas 4–6 si pasamos a **paper trading** (conservador).

### 2) Principios
- **Reproducibilidad:** usar `--freeze_end` y/o `--max_bars`.
- **Conexión:** `EXCHANGE=binanceus` (evita 451), `--skip_yf` cuando aplique.
- **Datos:** `_fetch_ccxt` con paginación `limit=1500`.

### 3) Cronograma y hitos

**Semana 0 — ✅ completada**  
Parche JSON (Timestamp), Sortino, “Selected”, gate ADX1D (len=14), flags utilitarias.

**Semanas 1–2 — Robustez**
- **Día 1–2:** Validación **multi-activo** (BTC, ETH, SOL). Gate: **PF>1.6**, **WR>60%**, **trades ≥30** por horizonte.
- **Día 3:** **Micro-grid** por fold (threshold / tp / sl / partial) → elegir por **validación**.
- **Día 4:** **Costes duros** (2× SLIP + fee in/out). Gate semanal: **PF>1.5**.
- **Día 5:** **Cierre** (freeze viernes 00:00 UTC), consolidación CSV/JSON y dashboard.
Como vamos :
	•	Día 1–2 (multi-activo): BTC y ETH listos (sizing estable). Falta SOL → arriba te dejé cómo añadirlo.
	•	Día 3 (micro-grid): Pendiente; arriba tienes los bucles de H y TH. Podemos añadir también variantes de partial (100% TP1 vs 50/50).
	•	Día 4 (costes duros): Pendiente; define FEE_BPS y SLIP_BPS y reevalúa PF con el mismo grid.
	•	Día 5 (cierre & dashboard): ya generaste orders_preview_*.csv. Falta agregador de KPIs por corrida/ventana para cerrar la semana.

**Semanas 3–4 — OOS rolling & guardrails**  
Out-of-sample mensual encadenado, filtros de régimen y stress tests (spreads/latencia/fees).

**Semanas 4–6 — Decisión**  
Semáforo: 🟢 PF ≥ 1.15 & Sortino ≥ 0.50 & MDD ≤ 1.1× baseline → **paper** con riesgo reducido.


Nueva tarea ?
Qué hacen exactamente las piezas nuevas (para cuando toque)
	•	Loader diamante_selected.yaml: auto-inyecta tu última selección (p. ej. H=90, TH=0.66) sin tocar comandos.
	•	--perla_csv + --max_corr: (apagado por defecto) leen la serie de posición o señal binaria de Perla y calculan la correlación efectiva con la exposición de Diamante en la ventana OOS; si supera --max_corr, se descarta esa config. Esto es una forma práctica de forzar diversificación (evitar config “pegadas”).  ￼

Respuesta directa a tus preguntas
	•	¿Encaja en el Plan Macro? Sí: lo dejamos agendado para la etapa de evaluación (Semanas 3–6); hoy no se usa.


### 4) Setup de ejecución (variables sugeridas)
```bash
export EXCHANGE=binanceus
export FREEZE="2025-08-05 00:00""

5) Comandos de trabajo (Semana 1)

Hoy (Lunes · Día 1) — Validación multi-activo (alta prioridad)
bash : 
# BTC
EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf \
  --symbol BTC-USD --period 730d --horizons 30,60,90 \
  --freeze_end "2025-08-05 00:00" \
  --out_csv reports/diamante_btc_week1.csv

# ETH
EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf \
  --symbol ETH-USD --period 730d --horizons 30,60,90 \
  --freeze_end "2025-08-05 00:00" \
  --out_csv reports/diamante_eth_week1.csv

# SOL
EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf \
  --symbol SOL-USD --period 730d --horizons 30,60,90 \
  --freeze_end "2025-08-05 00:00" \
  --out_csv reports/diamante_sol_week1.csv

Martes · Día 2 — Revisión gate multi-activo
	•	Leer reports/diamante_*_week1.csv.
	•	Confirmar por activo y horizonte (30/60/90): PF>1.6, WR>60%, trades≥30.

Miércoles (Día 3) — Micro-grid por fold (validación)
bash :
EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf \
  --walk_k 6 --horizons 30,60,90 \
  --best_p_json '{"threshold":0.60,"sl_atr_mul":1.3,"tp1_atr_mul":0.8,"tp2_atr_mul":5.0,"partial_pct":0.70}' \
  --out_csv reports/diamante_microgrid_week1.csv

Jueves (Día 4) — Costes realistas
bash :
EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf \
  --slip 0.0002 --cost 0.0004 --horizons 30,60,90 \
  --out_csv reports/diamante_costes_week1.csv

Viernes (Día 5) — Cierre
	•	--freeze_end del viernes 00:00 UTC.
	•	Consolidar métricas multi-activo y snapshot semanal en reports/.

6) Reporterías y KPIs
	•	KPIs mínimos: Net, PF, Win%, MDD, Sortino.
	•	Adicionales (pendiente): Sharpe/Sortino por trade, time-in-market, exposición, curva vs B&H.

7) Costes y riesgo
	•	Fee en entrada+salida (parámetro --cost), SLIP en entrada (--slip).
	•	Riesgo por ATR (RISK_PERC) ya integrado.

8) Guardrails de régimen
	•	ADX1D (len=14, umbral configurable).
	•	Próximos: slope EMA/MACD (evitar shorts en bull o subir --threshold).

9) Paper trading (condicionado)
	•	Requisitos: 2 semanas seguidas 🟢 en ≥2 activos.
	•	Riesgo por trade 0.5–1.0%, logs y kill-switch por MDD semanal.

⸻

Fase B — Corazón (prob + gates · NO ATR) — Programado después de Diamante

No ejecutar durante Fase A. Preparado para arrancar al terminar Diamante.

C1) Objetivo
	•	Señal probabilística (RF) con umbral + gates (ADX/FG/Funding).
	•	Sin TP/SL por ATR ni parciales.
	•	Automatizar snapshot diario y ranking por pf_60d.

C2) Entradas y salidas
	•	Datos:
	•	Fear&Greed → ./data/sentiment/fear_greed.csv
	•	Funding → ./data/sentiment/funding_rates.csv
	•	Salidas:
	•	reports/corazon_metrics_auto_all.csv
	•	reports/corazon_metrics_auto_longv2.csv
	•	reports/corazon_auto_daily.csv (bitácora recomendación)

C3) Parámetros base (cuando toque arrancar)
	•	ALL (sin gates): --threshold 0.60, --no_gates, --max_bars 975, --freeze_end <lunes C1>.
	•	LONG_V2 (con gates):
	•	--fg_long_min -0.15  --fg_short_max 0.15
	•	--funding_bias 0.005
	•	--adx1d_len 14  --adx1d_min 22  --adx4_min 12
	•	--threshold 0.60

C4) Tareas de la Semana C1 (post-Diamante)
	1.	Ajustar FREEZE al lunes de arranque (C1).
	2.	Ejecutar snapshot diario (una corrida manual para verificar):
bash : 
source scripts/corazon_cmds.zsh
runC_auto_daily
runC_auto_status

	3.	Programar tarea diaria (cron/launchd) apuntando a runC_auto_daily.
	4.	Validar pf_60d, wr_60d, mdd_60d, trades_60d en corazon_auto_daily.csv.

C5) Tareas de la Semana C2
	1.	Sweep de umbral (ajuste fino):
	•	ALL: --sweep_threshold 0.56:0.64:0.01
	•	LONG_V2: --sweep_threshold 0.57:0.62:0.01
	2.	Fijar threshold definitivo (si cambia) y mantener automatización diaria.

Nota: Corazón ya tiene scripts/corazon_cmds.zsh y corazon_auto.py listos, pero se usan recién en C1.


Checklist semanal (enfoque)
	•	Diamante Semana 1 — ejecutar solo lo listado arriba (BTC/ETH/SOL hoy).
	•	No correr funciones de Corazón (postergar a C1).
	•	Viernes: consolidar y evaluar gates para plan de Semana 2.
	•	Al terminar Fase A (semanas 4–6): decidir paper y recién ahí activar Fase B (Corazón C1–C2).

-------------------------------------------------------------------------------------------------
Track: Corazón (NO-ATR) 
Anterior, el nuevo es el de arriba!
Core idea: Señal probabilística (RF) con umbral y gates (ADX / Fear&Greed / Funding).
Gestión: Sin SL/TP por ATR ni parciales.
Output principal: Snapshot diario de métricas + recomendación automática.

⸻

1) Objetivo operativo
	•	Ejecutar a diario un backtest congelado (freeze) y un snapshot live para comparar consistencia.
	•	Generar CSV de métricas y un log de recomendaciones con ranking por pf_60d.
	•	Disponer de funciones rápidas para:
	•	runC_all_freeze / runC_all_live (ALL, sin gates).
	•	runC_long_freeze_v2 / sweep para la variante LONG_V2 con gates.
	•	runC_auto_daily + runC_auto_status (pipeline diario automatizado).

⸻

2) Directorio, rutas y entorno
	•	Repo: ~/PycharmProjects/Bot_BTC (ajusta si aplica).
	•	Venv: .venv/ (Python 3.11).
	•	Datos de sentimiento:
	•	Fear & Greed: ./data/sentiment/fear_greed.csv
	•	Funding rates: ./data/sentiment/funding_rates.csv
	•	Reports (salidas): reports/

Variables recomendadas (por defecto ya cubiertas en los helpers):
zsh:
export EXCHANGE=binanceus
export FREEZE="2025-08-05 00:00"     # Congelado semanal (actualízalo cada lunes)
export FG=./data/sentiment/fear_greed.csv
export FU=./data/sentiment/funding_rates.csv
export OUT=reports

3) Scripts y comandos clave

3.1 Cargar helpers (zsh)
bash:
source scripts/corazon_cmds.zsh
# Ver funciones disponibles
runC_status

Funciones relevantes (ya definidas en tu scripts/corazon_cmds.zsh):
	•	ALL, sin gates
	•	runC_all_freeze → freeze a FREEZE (975 velas)
	•	runC_all_live   → live/rolling (975 velas)
	•	LONG_V2, con gates (ADX/FG/Funding)
	•	runC_long_freeze_v2 → parámetros gates “22/14/12”
	•	runC_sweep_long_v2 / runC_sweep_long_v2_22 → sweep de umbral con gates
	•	Sweep (ALL, sin gates)
	•	runC_sweep_freeze
	•	Auto diario
	•	runC_auto_daily   → ejecuta ALL (sin gates) + LONG_V2 (con gates) y append al CSV de recomendaciones
	•	runC_auto_status  → muestra últimas filas del CSV

3.2 Ejecución manual rápida

ALL (sin gates) — freeze
bash :
runC_all_freeze

ALL (sin gates) — live
bash : 
runC_all_live

Sweep de umbral (sin gates) — freeze
bash :
runC_sweep_freeze

LONG_V2 (con gates) — freeze
bash :
runC_long_freeze_v2

Sweep de umbral — LONG_V2 (gates 22/14/12)
bash :
runC_sweep_long_v2_22
4) Automatización diaria (snapshot + ranking)

4.1 Script: corazon_auto.py
	•	Corre dos pasadas:
	1.	ALL (sin gates) con --no_gates
	2.	LONG_V2 (con gates “22/14/12”: adx1d_len=14, adx1d_min=22, adx4_min=12, fg_long_min=-0.15, fg_short_max=0.15, funding_bias=0.005)
	•	Umbral por defecto: 0.60
	•	Congelado a FREEZE para comparabilidad semanal (ajústalo cada lunes).
	•	Escribe/append a: reports/corazon_auto_daily.csv
	•	Métrica de ranking: pf_60d (mayor es mejor).

Uso directo
bash :
python corazon_auto.py \
  --exchange binanceus \
  --fg_csv ./data/sentiment/fear_greed.csv \
  --funding_csv ./data/sentiment/funding_rates.csv \
  --max_bars 975 \
  --freeze_end "2025-08-05 00:00" \
  --compare_both \
  --adx_min 22 \
  --report_csv reports/corazon_auto_daily.csv

Con helpers:
bash :
runC_auto_daily
runC_auto_status

Salidas esperadas
	•	reports/corazon_metrics_auto_all.csv (ALL)
	•	reports/corazon_metrics_auto_longv2.csv (LONG_V2)
	•	reports/corazon_auto_daily.csv (bitácora con ranking por pf_60d)

Ejemplo de fila:
ts,exchange,symbol,freeze_end,label,threshold,pf_30d,pf_60d,pf_90d,wr_60d,mdd_60d,trades_60d,net_60d,rank

4.2 Programación (macOS)

Opción A — cron (simple)
bash :
crontab -e

Añade (ejecuta a las 09:30 todos los días; ajusta ruta/horario):

cron :
SHELL=/bin/zsh
30 9 * * * cd /Users/rafaelcifuentes/PycharmProjects/Bot_BTC && \
  source .venv/bin/activate && \
  source scripts/corazon_cmds.zsh && \
  runC_auto_daily >> logs/corazon_auto.log 2>&1

Opción B — launchd (más “mac-style”)
	•	Crea ~/Library/LaunchAgents/com.corazon.auto.plist con un ProgramArguments que invoque:
zsh -lc 'cd <repo> && source .venv/bin/activate && source scripts/corazon_cmds.zsh && runC_auto_daily'
	•	Carga: launchctl load ~/Library/LaunchAgents/com.corazon.auto.plist
	•	Logs: redirige stdout/err en el plist o usa runC_auto_status.

Nota: cada lunes, actualiza FREEZE para el nuevo período de comparación semanal.

⸻

5) Parámetros y defaults recomendados

ALL (sin gates)
	•	--threshold 0.60 (sweet spot típico: 0.60–0.61)
	•	--no_gates
	•	--max_bars 975 y --freeze_end "$FREEZE" para runs reproducibles

LONG_V2 (con gates)
	•	--fg_long_min -0.15  --fg_short_max 0.15
	•	--funding_bias 0.005
	•	--adx1d_len 14  --adx1d_min 22  --adx4_min 12
	•	--threshold 0.60

Sweep (para recalibrar umbral)
	•	ALL: --sweep_threshold 0.56:0.64:0.01
	•	LONG_V2: --sweep_threshold 0.57:0.62:0.01

⸻

6) KPIs y criterio de lectura
	•	pf_Xd (Profit Factor): > 1.6 fuerte; 1.3–1.6 aceptable; <1.3 débil.
	•	wr_Xd (Win Rate): > 60% sano; revisar si hay muy pocos trades.
	•	mdd_Xd (Max Drawdown): mantener bajo (Corazón suele estar < 3%).
	•	trades_Xd: ideal ≥ 20–30 en 60–90d para significancia.
	•	net_Xd: PnL agregado (unidades internas del backtest; úsalas comparativamente).

Ranking automático (bitácora diaria):
	•	Se ordena por pf_60d (descendente). Etiquetas: ALL vs LONG_V2.

⸻

7) Entregables y rutas
	•	Diario
	•	reports/corazon_metrics_auto_all.csv
	•	reports/corazon_metrics_auto_longv2.csv
	•	reports/corazon_auto_daily.csv (bitácora con ranking)
	•	Ad-hoc
	•	reports/corazon_metrics.csv / *_live.csv
	•	reports/corazon_*_sweep*.csv (sweeps)
	•	reports/corazon_trades_*.csv (detalle de trades)

⸻

8) Troubleshooting rápido
	•	Warning pkg_resources is deprecated (pandas_ta): inocuo.
	•	FG/Funding vacíos: pasa rutas explícitas (--fg_csv, --funding_csv) o define FG/FU.
	•	Found array with 0 sample(s): suele indicar que, tras gates + umbral, no hay muestras; prueba:
	•	bajar umbral o ampliar ventana (--max_bars),
	•	relajar gates (ej. adx1d_min), o ejecutar ALL sin gates para sanity check.
	•	Timezone patch: runC_all_live ya intenta parchear runner_corazon.py si detecta desfases.

⸻

9) Apéndice — Track Diamante (referencia rápida)

Diamante es swing 4h con ATR (SL/TP1/TP2 + parcial); independiente de Corazón.
Comandos base (Semana 1 — validación multi-activo):
bash:
# BTC
EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf \
  --symbol BTC-USD --period 730d --horizons 30,60,90 \
  --freeze_end "2025-08-05 00:00" \
  --out_csv reports/diamante_btc_week1.csv

# ETH
EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf \
  --symbol ETH-USD --period 730d --horizons 30,60,90 \
  --freeze_end "2025-08-05 00:00" \
  --out_csv reports/diamante_eth_week1.csv

# SOL
EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf \
  --symbol SOL-USD --period 730d --horizons 30,60,90 \
  --freeze_end "2025-08-05 00:00" \
  --out_csv reports/diamante_sol_week1.csv

10) Checklist semanal (Corazón)
	•	Lunes: actualizar FREEZE y correr runC_auto_daily.
	•	Revisar reports/corazon_auto_daily.csv → confirmar pf_60d, wr_60d, mdd_60d y trades_60d.
	•	Si pf_60d cae < 1.3 o trades_60d < 15, correr sweep y evaluar threshold.
	•	Mantener FG/FU al día; sincronizar fuentes si cambia el proveedor.

⸻
Perla Negra (plan y status actual)

¡claro!

Nombre de la etapa

Perla Negra V2 — Semana
(runner_conflict_guard_weekly_v2.py, “regla híbrida”)

Finalidad

Ajustar solo los pesos del portafolio (wL / wS) del Conflict-Guard de Andromeda una vez por semana, usando la regla híbrida:
	•	✅ Score sube (mejor riesgo/retorno),
	•	✅ MDD ≤ cap (cap = min_leg × 1.05*),
	•	✅ Net no cae más del 20% respecto al baseline (o sube).

Si se cumplen los tres, se hace autolock de los nuevos pesos; si no, se mantienen los anteriores. El objetivo es adaptarse al mercado sin sacrificar la brújula de riesgo y rentabilidad.

Plan macro (resumen)
	1.	Baseline anual estable (perfil de referencia).
	2.	Perla Negra V2 semanal (esta etapa) con autolock por regla híbrida.
	3.	Evaluación 4–8 semanas: comparar dinámico vs fijo (mirando Net, PF, MDD y Score).
	4.	Decisión: si el ajuste semanal gana rentabilidad ajustada a riesgo, se adopta como estándar; si no, volvemos al fijo.
	5.	Opcional: correr V3 (micro-grid fino) ad-hoc para intentar +2–3% Net manteniendo MDD bajo (no interfiere con V2).
	6.	Paso a paper/live con la configuración ganadora.

(Recordatorio operativo: correr los miércoles 08:10 Montreal ≈ 12:10 UTC; guardar logs en ./logs/ y revisar ./profiles/andromeda_conflict_guard_weights.json.)