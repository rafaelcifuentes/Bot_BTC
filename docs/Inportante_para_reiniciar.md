Confirmación de alcance

Sí: con lo que ya tenemos (pipeline de señales tzsafe, backtester y diagnósticos) podemos llevar a término Diamante → Perla → Corazón y luego el Allocator (Cerebro). El cuello de botella de Q4 (Trades que no activan gates) lo estamos tratando con micro-grid + filtros de tendencia "suaves" y, si aún así Q4 no entra, pasamos a validación rolling/walk-forward (estándar en trading) sin sobre-optimizar un único trimestre. Esto sigue las mejores prácticas de validación temporal (rolling origin) y walk-forward, evitando data-snooping con tests de robustez como White/SPA y métricas como Deflated Sharpe (para no "comprarnos" falsas victorias de backtest).  ￼ ￼ ￼

⸻

Plantilla diaria (para copiar/pegar al cierre del día)

1) Objetivo del día (según plan semanal)
	•	Ej.: "Día 4 — Micro-grid Q4 (threshold/partial/horizon) + validación rolling en Fold A/B".

2) Pasos ejecutados (checklist mínimo)
	•	Correr micro-grid param X (cmd exacto abajo)
	•	Validación rolling (TimeSeriesSplit / walk-forward)
	•	Export de resultados (KPIs + logs)
	•	Snapshot de entorno/datos (ver §3)

3) Snapshot de reinicio (obligatorio para reiniciar el proyecto sin perder contexto)
	•	Git: branch + git rev-parse --short HEAD
	•	Entorno: conda env export > environment.yml y pip freeze > requirements.txt (garantiza reproducibilidad).  ￼ ￼
	•	Datos: marca/commit/tag de datos (si usas DVC: dvc add + dvc push o referencia del dataset usado).  ￼
	•	Semilla RNG (si aplica) y flags deterministas.
	•	Ventanas/periodos exactos evaluados (ej.: 2023Q4: 2023-10-01 → 2023-12-31).
	•	Comandos ejecutados (tal cual en consola).
	•	Artefactos: paths a reports/kpis_grid.csv, reports/top_configs.csv, y a logs/*.log.
	•	Top configs del día: (asset, horizon, threshold, partial, filtros, KPIs).
	•	Decisión de salida (ver §4 criterios).

4) Criterios de salida / fin de etapa (con "go/no-go")
	•	Micro-grid por fold (Día 3–4)
	•	GO si existe ≥1 config que cumple gates en Q4 y pasa rolling split (WR y PF no caen >X%).
	•	NEAR-MISS si Q4 da Trades≥30 con WR≈0.58–0.59 → aplicar "poda suave" (↑threshold +0.02 y/o ↑REARM_MIN).
	•	NO-GO si no hay Trades≥30 en Q4 tras micro-grid + poda suave → pasamos a ajustar únicamente filtros de tendencia (frecuencia/span/banda) manteniendo TP/SL/partials fijos.
	•	Sustenta el enfoque con walk-forward y TimeSeriesSplit de sklearn (o equivalente) para no leakear información futura.  ￼

⸻

Inventario de scripts esenciales (para tener "a mano" y versionados)

Del repo actual (confirmados por tus logs):
	•	scripts/run_grid_tzsafe.py — Lanza grids de backtest con señales + filtros.
	•	scripts/quick_kpi_diagnosis.py — Lee reports/kpis_grid.csv y reporta cobertura/near-miss/top configs.
	•	scripts/backtest_grid.py — Backend de grillas (CLI base que aparece en los errores si faltan args).

Convención que mantendremos (evita improvisar y facilita reinicio):
	•	Diamante (tendencia + probabilidad): usamos los 3 de arriba (no cambiamos nombres).
	•	Perla (mean-reversion) — a crear cuando toque: scripts/run_grid_perla.py (misma interfaz CLI).
	•	Corazón (breakout intradía) — a crear cuando toque: scripts/run_grid_corazon.py (misma interfaz CLI).
	•	Allocator (Cerebro) — esqueleto propuesto: scripts/allocator_cerebro.py que:
	1.	lee reports/top_configs.csv por régimen/ventana,
	2.	aplica reglas de asignación (p.ej., normalización por riesgo),
	3.	escupe reports/allocationsYYYYMMDD.json/csv.

Nota: que Perla/Corazón/Allocator compartan la misma interfaz CLI reduce fricción y riesgos de reproducibilidad. Para tracking de experimentos, si deseas algo más robusto, podemos anotar corridas en MLflow/DVC (experimentos/datos).  ￼ ￼

⸻

Lista "mínima de misión" (lo que debes guardar cada día)
	1.	environment.yml y requirements.txt actualizados (post-cambios).  ￼ ￼
	2.	Hash Git, branch, y mensaje corto del objetivo del día.
	3.	Comandos ejecutados (texto literal).
	4.	Artefactos: reports/kpis_grid.csv, reports/top_configs.csv, logs/*.log.
	5.	Resumen "top-3 configs" por fold con sus KPIs.
	6.	Checklist completado y decisión GO/NO-GO según criterios.
	7.	Si hubo cambios de datos: commit/tag de DVC o checksum + origen.  ￼
	8.	Próximos pasos y comandos listos para el día siguiente.

⸻

Recordatorio metodológico (por qué este flujo es "la vía segura")
	•	Validación rolling/walk-forward para series temporales (no shuffle).  ￼
	•	Evitar data-snooping/overfitting: tests White's Reality Check y SPA cuando elijamos "el mejor" sobre muchas combinaciones.  ￼ ￼
	•	Reportar rendimiento con correcciones como Deflated Sharpe Ratio y considerar la probabilidad de backtest overfitting.  ￼ ￼

⸻

¿Qué te entrego al final de cada día?
	•	1 doc con la plantilla completada (secciones 1–4)
	•	1 paquete de reinicio (environment.yml, requirements.txt, comandos, hashes, artefactos, próximas corridas)
	•	1 inventario de scripts esenciales (actualizado)
	•	1 carpeta de logs con timestamp