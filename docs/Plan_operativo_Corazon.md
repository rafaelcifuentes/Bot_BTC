Plan Operativo — Corazón (intradía 4h, prob + gates)

Track: Corazón (NO-ATR)
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
zhs:
export EXCHANGE=binanceus
export FREEZE="2025-08-05 00:00"     # Congelado semanal (actualízalo cada lunes)
export FG=./data/sentiment/fear_greed.csv
export FU=./data/sentiment/funding_rates.csv
export OUT=reports


⸻

3) Scripts y comandos clave

3.1 Cargar helpers (zsh)
source scripts/corazon_cmds.zsh
# Ver funciones disponibles
runC_status

unciones relevantes (ya definidas en tu scripts/corazon_cmds.zsh):
	•	ALL, sin gates
	•	runC_all_freeze → freeze a FREEZE (975 velas)
	•	runC_all_live   → live/rolling (975 velas)
	•	LONG_V2, con gates (ADX/FG/Funding)
	•	runC_long_freeze_v2 → parámetros gates "22/14/12"
	•	runC_sweep_long_v2 / runC_sweep_long_v2_22 → sweep de umbral con gates
	•	Sweep (ALL, sin gates)
	•	runC_sweep_freeze
	•	Auto diario
	•	runC_auto_daily   → ejecuta ALL (sin gates) + LONG_V2 (con gates) y append al CSV de recomendaciones
	•	runC_auto_status  → muestra últimas filas del CSV

3.2 Ejecución manual rápida

ALL (sin gates) — freeze
Basch :
runC_all_freeze
runC_all_live
runC_sweep_freeze
runC_long_freeze_v2
runC_sweep_long_v2_22


⸻

4) Automatización diaria (snapshot + ranking)

4.1 Script: corazon_auto.py
	•	Corre dos pasadas:
	1.	ALL (sin gates) con --no_gates
	2.	LONG_V2 (con gates "22/14/12": adx1d_len=14, adx1d_min=22, adx4_min=12, fg_long_min=-0.15, fg_short_max=0.15, funding_bias=0.005)
	•	Umbral por defecto: 0.60
	•	Congelado a FREEZE para comparabilidad semanal (ajústalo cada lunes).
	•	Escribe/append a: reports/corazon_auto_daily.csv
	•	Métrica de ranking: pf_60d (mayor es mejor).

Uso directo
Bash:
python corazon_auto.py \
  --exchange binanceus \
  --fg_csv ./data/sentiment/fear_greed.csv \
  --funding_csv ./data/sentiment/funding_rates.csv \
  --max_bars 975 \
  --freeze_end "2025-08-05 00:00" \
  --compare_both \
  --adx_min 22 \
  --report_csv reports/corazon_auto_daily.csv

Con helpers
Bsh:
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
Bash :
crontab -e

Añade (ejecuta a las 09:30 todos los días; ajusta ruta/horario):
Cron :
SHELL=/bin/zsh
30 9 * * * cd /Users/rafaelcifuentes/PycharmProjects/Bot_BTC && \
  source .venv/bin/activate && \
  source scripts/corazon_cmds.zsh && \
  runC_auto_daily >> logs/corazon_auto.log 2>&1

Opción B — launchd (más "mac-style")
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

Bash :
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

Plan
	•	Mantener snapshot semanal (FREEZE) sin cambios.
	•	Criterios PASS (PF dentro ±5–10% de base, MDD/σ ≤ base, exposición razonable).

Do
	•	Correr runC_shadow_daily (diario) y runC_log_weekly_freeze (lunes).
	•	Registrar en corazon/daily_xi.csv (ya lo hace el flujo) y guardar KPIs/summary.

Check
	•	Vigilar:
	•	PF_overlay < 0.90× PF_base → alerta.
	•	|MDD_overlay| > |MDD_base| o σ_overlay > σ_base → alerta.
	•	Exposición muy capada (p.ej., w̄ << 0.6) sin razón → revisar.

Act
	•	Si perdemos impulso (FAIL): no escalamos; analizamos señales/umbrales del YAML (ADX/EMA/ATR, histéresis, maxΔ/barra, corr-gate) y ajustamos en laboratorio; rehacemos FREEZE y re-medimos.
	•	Si mantenemos PASS 1–2 semanas: entonces sí proponemos aplicar ξ* en Cerebro (paso controlado, por ejemplo 50% del salto primero) manteniendo límites de riesgo.

