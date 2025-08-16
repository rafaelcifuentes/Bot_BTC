# Plan operativo (6 semanas) — Bot BTC

**Premisa:** mantener **V2 intocable** durante 4–6 semanas mientras comparamos semanalmente contra dos benchmarks (**Perla Negra** y **Sentiment EXP**) y preparamos un arranque de **paper trading** condicionado.

---

## 1) Objetivo

- Validar **estabilidad** y **competitividad** de la cartera de señales actual sin alterar V2.
- Medir desempeño semanal con **PF**, **MDD**, **Score** y **Sortino**.
- Decidir (semana 4–6) si promovemos algún runner o activamos ajustes en **autolock**.

## 2) Principios

- **No tocar V2** en 4–6 semanas.
- Comparar **manzanas con manzanas**: usar corrida **congelada** (reproducible) para tuning y **en vivo** para feeling de mercado.
- **Trazabilidad**: cada corrida guarda JSON/CSV y se registra en un resumen semanal.

## 3) Cronograma y hitos

**Semana 0 (hoy) — estado: ✅ completado**
- Parche JSON (Timestamp) ✅  
- Sortino en el reporte ✅  
- Validación rápida con “Selected” ✅  
- Gate **ADX1D** integrado (resample, len=14, umbral configurable) ✅

**Semanas 1–2**
- V2 (sin cambios): 1 corrida semanal (congelada + en vivo) → registrar métricas.
- Perla Negra (benchmark): misma cadencia.
- Sentiment EXP (benchmark): misma cadencia.
- Paper trading: dejar config/credenciales listas ahora; arrancar solo si 1–2 semanas confirman (**PF ≥ 1.10**, **MDD ≤ baseline**, **Sortino ≥ 0.3**).

**Semanas 3–4**
- Mantener rutina semanal y actualizar semáforo.
- Si se habilita paper: operar **riesgo reducido** (p.ej., `risk_perc = 0.75%`) + **kill-switch** por MDD intrames.

**Semanas 4–6 — evaluación de promoción**
- Regla de racha (2–3 runs): **Score ↑** vs baseline y **MDD ≤ baseline** (+ tolerancia 10–20%).
- Umbrales sugeridos: **PF ≥ 1.15**, **Sortino ≥ 0.50**, **MDD ≤ 1.1× baseline**.
- Post–evaluación: si procede, activar tolerancia de **Net** en autolock (±30%) y/o usar **Sortino** como desempate.

## 4) Métricas y semáforo

- Métricas: **Net**, **PF (Profit Factor)**, **Win%**, **MDD**, **Score**, **Sortino**.
- Semáforo semanal:
  - 🟢 **Verde**: PF ≥ 1.15 y Sortino ≥ 0.50 y MDD ≤ baseline×1.1 → Candidato a promoción / iniciar paper.
  - 🟡 **Amarillo**: 1 de 3 falla levemente → seguir observando.
  - 🔴 **Rojo**: PF < 1.0 o MDD > baseline×1.2 → no promover / pausar paper.

## 5) Procedimiento semanal (paso a paso)

1. Correr **congelada** (misma ventana) con `--freeze_end` + `--repro_lock`.  
2. Correr **en vivo** (sin freeze) para sentir mercado actual.  
3. Guardar artefactos (JSON/CSV) en `./reports/` (auto).  
4. Actualizar **registro semanal (CSV)** con PF, MDD, Score, Sortino, notas.  
5. **Semáforo**: marcar verde/amarillo/rojo para cada runner.  
6. Si **2 semanas** consistentes y semáforo **🟢** → arrancar **paper trading**.

## 6) Configs sugeridas (Sentiment EXP)

**Freeze (comparables):**
```bash
: ${FREEZE:=2025-08-10}
python runner_profileA_RF_sentiment_EXP.py \
  --primary_only --freeze_end "$FREEZE" --repro_lock \
  --use_sentiment --fg_csv ./data/sentiment/fear_greed.csv \
  --funding_csv ./data/sentiment/funding_rates.csv \
  --fg_long_min -0.18 --fg_short_max 0.18 --funding_bias 0.01 \
  --threshold 0.59 \
  --pt_mode pct --sl_pct 0.02 --tp_pct 0.10 --trail_pct 0.015 \
  --adx_daily_source resample --adx1d_len 14 --adx1d_min 30 \
  --adx4_min 18

Live:
python runner_profileA_RF_sentiment_EXP.py \
  --primary_only \
  --use_sentiment --fg_csv ./data/sentiment/fear_greed.csv \
  --funding_csv ./data/sentiment/funding_rates.csv \
  --fg_long_min -0.18 --fg_short_max 0.18 --funding_bias 0.01 \
  --threshold 0.59 \
  --pt_mode pct --sl_pct 0.02 --tp_pct 0.10 --trail_pct 0.015 \
  --adx_daily_source resample --adx1d_len 14 --adx1d_min 30 \
  --adx4_min 18
  
7) Helpers (zsh)

Pégalos en tu ~/.zshrc o en utilidades del repo.
# Defaults y flags seguros (zsh arrays)
typeset -a FG=( --fg_long_min -0.18 --fg_short_max 0.18 )
: ${FREEZE:=2025-08-10}

runEXP_freeze() {
  python runner_profileA_RF_sentiment_EXP.py \
    --primary_only --freeze_end "$FREEZE" --repro_lock \
    --use_sentiment --fg_csv ./data/sentiment/fear_greed.csv \
    --funding_csv ./data/sentiment/funding_rates.csv \
    "${FG[@]}" --funding_bias "${1:-0.01}" \
    --threshold "${2:-0.59}" \
    --pt_mode pct --sl_pct 0.02 --tp_pct 0.10 --trail_pct 0.015 \
    --adx_daily_source resample --adx1d_len 14 --adx1d_min "${3:-30}" \
    --adx4_min "${4:-18}"
}

runEXP_live() {
  python runner_profileA_RF_sentiment_EXP.py \
    --primary_only \
    --use_sentiment --fg_csv ./data/sentiment/fear_greed.csv \
    --funding_csv ./data/sentiment/funding_rates.csv \
    "${FG[@]}" --funding_bias "${1:-0.01}" \
    --threshold "${2:-0.59}" \
    --pt_mode pct --sl_pct 0.02 --tp_pct 0.10 --trail_pct 0.015 \
    --adx_daily_source resample --adx1d_len 14 --adx1d_min "${3:-30}" \
    --adx4_min "${4:-18}"
}

runEXP_both() { runEXP_freeze "$@"; runEXP_live "$@"; }

# Ejemplos:
# runEXP_freeze 0.01 0.59 30 18
# runEXP_live   0.01 0.59 30 18
# runEXP_both   0.01 0.59 30 18

Nota: corregido el typo en el ejemplo (0.01).

8) Semáforo de entorno (CLI)

Actívalo con --traffic_light para ver estado ADX1D/ADX4h/Sentimiento (último valor y % últimos 30 días).
Regla práctica: si el semáforo tiende a 🟡/🔴 y los picks del barrido tienen net180 ≤ 0 o holdout flojo, reduce exposición o sube --threshold.

9) Barridos y resúmenes (Makefile)
	•	Solo barrido actual (INDEX):
	•	make sweep-sentexp (o sweep-sentexp-freeze)
	•	make summarize-sweep-index
	•	make open-sweep-index
	•	Histórico completo (PATTERN):
	•	make summarize-sweep-pattern
	•	make open-sweep-pattern

10) Paper trading — checklist

Listo ahora (sin ejecutar aún):
	•	.env/credenciales de exchange (paper/sandbox).
	•	risk_perc inicial 0.75% (ajustable 0.5–1.0% según ADX1D).
	•	Kill-switch: pausa si MDD semanal > 1.2× baseline o 2 pérdidas seguidas con PF < 0.8.
	•	Logging: ./logs/paper/.
	•	Para encender: 2 semanas 🟢 en al menos 2 de 3 runners y sin errores de datos.

11) Reglas de cambio y promoción
	•	Ningún cambio directo a V2 durante el período.
	•	Ajustes se prueban primero en Sentiment EXP (o Perla Negra) con corrida congelada.
	•	Si se cumplen umbrales y racha 2–3 runs → congelar nueva config como V2.1.
	•	Autolock (post-promoción): tolerancia de Net (±30%) + Sortino como desempate si PF≈.

12) Artefactos y registro
	•	Reportes automáticos: ./reports/summary_*.json, walkforward_*.csv, competitiveness_*.json.
	•	Registro semanal (CSV): ./reports/weekly/weekly_summary.csv (1 fila por runner/semana).
	•	Campos sugeridos: fecha, runner, modo (congelada/en_vivo), Net, PF, Win%, MDD, Score, Sortino, ADX1D_min, notas, semáforo.

13) Interpretación del barrido actual

Si el pick del barrido (p.ej. 0.57 / 25 / 16) muestra net180 negativo y holdout flojo, el entorno no es favorable.
Acciones sugeridas:
	•	Subir --threshold (menos señales, mayor selectividad), o
	•	Reducir exposición temporalmente (menor risk_perc, o pausar si semáforo 🔴).

14) Validación rápida (sanity check)

# Resumen INDEX (solo barrido actual)
make sweep-sentexp
make summarize-sweep-index
make open-sweep-index

# Resumen PATTERN (todo el histórico)
make summarize-sweep-pattern
make open-sweep-pattern

# Conteo de filas (deberías ver 16 combos en INDEX)
wc -l reports/sweep_index.csv

# Ver el “mejor” seleccionado
cat reports/sweep_index_out/sweep_best.json 2>/dev/null || true
cat reports/sweep_pattern_out/sweep_best.json 2>/dev/null || true

