Plan operativo (6+2 semanas) — Diamante primero, Corazón después

Actualización Semanas 3–6 (OOS rolling, guardrails y Decisión)

Objetivo (en cristiano)
	•	Diamante = “francotirador” swing 4h con gestión por ATR y parcial. Lo que añadimos (loader de diamante_selected.yaml, gate de correlación contra Perla) no cambia la lógica ni tus resultados; son herramientas para automatizar selección y evitar que la cartera quede “pegada” a otra pata.
	•	Perla (V2) queda intocable por ahora; solo definimos cómo leer sus series y, cuando toque, medir la correlación para filtrar combinaciones demasiado similares. Eso es coherente con validación temporal (walk-forward / forward-chaining) y con buenas prácticas OOS.  ￼ 
        Premisa: Perla no se toca (solo lectura de su CSV cuando exista). Diamante sigue siendo el “francotirador” y mantiene independencia: el gate de correlación sirve para evitar que se pegue a Perla durante la evaluación.

Piezas nuevas y su propósito
	1.	Loader configs/diamante_selected.yaml
	•	Objetivo: que todas las corridas semanales usen la última selección (p. ej., H=90, TH=0.66) sin reescribir comandos.
	•	Efecto: estandariza y reduce errores. Los scripts cargan la selección y la imprimen al inicio del run.
Ejemplo de YAML
yaml
# configs/diamante_selected.yaml
horizon: 90
threshold: 0.66
partial: "50_50"
fee_bps: 6
slip_bps: 6
rearm_min: 6
hysteresis_pp: 0.04

	2.	Gate de correlación vs Perla: --perla_csv + --max_corr
	•	Objetivo: medir y limitar la correlación efectiva entre la exposición/posiciones de Diamante y la serie binaria/posiciones de Perla en la ventana OOS.
	•	Efecto: si corr_perla > --max_corr, descarta esa config (evita “duplicar” riesgo).
	•	Salida: los CSV (val_* o winners_*) incluyen una columna corr_perla.
Especificación mínima del CSV de Perla (--perla_csv)
	•	Debe cubrir la misma ventana temporal de validación.
	•	Columnas aceptadas (al menos una):
	•	ts (ISO 8601 o epoch), más una de:
	•	pos en {−1, 0, +1} o {0, 1}
	•	signal en {0, 1}
	•	El script alinea por timestamp y calcula Pearson sobre las intersecciones válidas.

⸻

Semanas 3–4 — OOS rolling & guardrails (con integración de piezas)

Objetivo: validar out-of-sample encadenado, con guardrails de costes/régimen y, si ya hay serie de Perla disponible, aplicar gate de correlación para filtrar configs “pegadas”.

Checklist semanal (Semanas 3–4)

A. Preparación (cada lunes):
	•	Actualiza FREEZE/ventanas OOS (mes encadenado; forward-chaining).
	•	Verifica que existe configs/diamante_selected.yaml.
	•	(Opcional) Si ya tienes Perla semanal: confirma ruta --perla_csv.

B. Corridas base (con loader activo):
	•	Ejecuta validación OOS con los costes acordados (slippage y fees de Día 4).
	•	Sin cambiar flags manuales de H/TH: el loader inyecta tu elección.
	•	Guarda KPIs y confirma impresión de selección cargada en logs.

Ejemplo (placeholders de ventana OOS):
bash:
# Con YAML loader (no pasas H/TH a mano)
PYTHONPATH="$(pwd):$(pwd)/scripts" \
python backtest_grid.py \
  --windows "2025M01:2025-01-01:2025-01-31" \
  --signals_root reports/windows_fixed --ohlc_root data/ohlc/1m \
  --fee_bps 6 --slip_bps 6 --partial 50_50 --breakeven_after_tp1 \
  --risk_total_pct 0.75 --weights BTC-USD=1.0 \
  --gate_pf 1.6 --gate_wr 0.60 --gate_trades 30 \
  --out_csv reports/wf_2025M01.csv --out_top reports/wf_2025M01_top.csv

C. Gate de correlación (activar sólo si ya hay Perla):
	•	Ejecuta la misma validación añadiendo --perla_csv ... --max_corr 0.35 (valor sugerido 0.30–0.40).
	•	Revisa que el CSV de salida tenga corr_perla y que respete el gate (descartes si > max_corr).

bash:
PYTHONPATH="$(pwd):$(pwd)/scripts" \
python backtest_grid.py \
  --windows "2025M01:2025-01-01:2025-01-31" \
  --signals_root reports/windows_fixed --ohlc_root data/ohlc/1m \
  --fee_bps 6 --slip_bps 6 --partial 50_50 --breakeven_after_tp1 \
  --risk_total_pct 0.75 --weights BTC-USD=1.0 \
  --gate_pf 1.6 --gate_wr 0.60 --gate_trades 30 \
  --perla_csv reports/perla_weekly_positions.csv \
  --max_corr 0.35 \
  --out_csv reports/wf_2025M01_corr.csv --out_top reports/wf_2025M01_corr_top.csv

D. Guardrails y stress:
	•	Régimen: mantén los filtros ya definidos (trend/ADX si aplica).
	•	Costes/latencia: repite con slip x2 y/o fee in/out mayores y confirma que PF OOS se mantiene aceptable (umbral interno ≥1.3 para stress).

E. Entregables Semanas 3–4:
	•	reports/wf_*.csv (con corr_perla si procede).
	•	reports/winners_wf.csv (finalistas tras gates + correlación).
	•	Bitácora semanal de parámetros cargados por loader (log).

⸻

Semanas 4–6 — Decisión (paper readiness)

Semáforo de decisión (con costes):
🟢 PF ≥ 1.15, Sortino ≥ 0.50, MDD ≤ 1.1× baseline
(Se evalúa OOS acumulado de Semanas 3–6; la correlación se usa como guardrail de independencia: preferir finalistas con corr_perla ≤ 0.35 si Perla está disponible).

Checklist semanal (Semanas 4–6)

A. Consolidación OOS (rolling):
	•	Repite forward-chaining con loader activo (misma receta de arriba).
	•	Si Perla está disponible, mantén --perla_csv/--max_corr para todas las ventanas OOS.

B. Scorecard de decisión:
	•	Une los KPIs OOS (PF, WR, MDD, Sortino) de todas las semanas.
	•	Verifica que los finalistas también cumplen corr_perla ≤ 0.35 (si aplica).
	•	Marca Semáforo (🟢/🟡/🔴).

C. Paper (si 🟢):
	•	Congela configs/diamante_selected.yaml y exporta perfil de riesgo.
	•	Genera “playbook” de ejecución y alertas (bitácora, límites de MDD semanal, kill-switch).
	•	(Perla queda fuera; sólo se usa su CSV como referencia de independencia).

Entregables Semanas 4–6:
	•	reports/wf_rollup.csv (todas las ventanas OOS concatenadas).
	•	reports/winners_final.csv (con corr_perla si procede).
	•	reports/decision_scorecard.md (semáforo + justificación).
	•	configs/diamante_selected.yaml (congelado para paper).

⸻
Qué hacen exactamente las piezas nuevas (para cuando toque)
	•	Loader diamante_selected.yaml: auto-inyecta tu última selección (p. ej. H=90, TH=0.66) sin tocar comandos.
	•	--perla_csv + --max_corr: (apagado por defecto) leen la serie de posición o señal binaria de Perla y calculan la correlación efectiva con la exposición de Diamante en la ventana OOS; si supera --max_corr, se descarta esa config. Esto es una forma práctica de forzar diversificación (evitar config “pegadas”).  ￼

⸻

Notas operativas rápidas
	•	Si no hay Perla aún, deja fuera --perla_csv/--max_corr. El plan sigue igual; la correlación se activa apenas esté la serie disponible (misma ventana OOS).
	•	La correlación efectiva se calcula sobre la exposición de Diamante (o señal binaria) y la pos/senal de Perla, alineadas por timestamp en la ventana. Se ignoran NaN tras alinear.
	•	corr_perla aparece en cada fila de los CSV de salida; si ejecutas rejillas, se aplica por cada combinación candidata.

⸻

Resumen de “cuándo usar qué”
	•	Loader diamante_selected.yaml:
➜ Desde Semana 3 en adelante en todas las corridas OOS.
Beneficio: consistencia y menos errores manuales.
	•	Gate --perla_csv / --max_corr:
➜ Semanas 3–6 solo si ya tienes la serie de Perla.
Beneficio: independencia de patas; evita configs demasiado correlacionadas (>0.35 sugerido).

Notas adicionales:
	•	El gate de correlación es para filtrar configs excesivamente alineadas con Perla (para que Diamante conserve su rol de “francotirador” independiente en la cartera). Esto está alineado con prácticas de forward-chaining y diversificación de cartera (minimizando covarianza entre legs).
	•	Si luego quieres hacer hard-gate (descartar filas con |corr|>max_corr) dentro del propio script, dilo y te dejo el filtro aplicado antes de guardar el CSV (ahora lo dejamos solo reportado en la columna corr_perla para inspección).

¿Quieres que también te deje un helper zsh para correr con --use_selected + --perla_csv ya cableado y un agregador que ordene por bajo corr_perla manteniendo PF/WR gates?

Blindaje anti-regresiones (Diamante) — Semanas 3–6

Objetivo: no degradar la versión actual de Diamante mientras buscamos mejores parámetros.
Archivo canónico: configs/diamante_selected.yaml (la única fuente de verdad).
Dónde guardamos: snapshots en reports/baseline/, experimentos en reports/experiments/.

Reglas:
	1.	Gate + No-regresión (obligatorio) antes de promover cualquier candidato:

	•	Gate por ventana OOS (walk-forward): PF ≥ 1.15, Sortino ≥ 0.50, MDD ≤ 1.1× baseline.
	•	Costes reales aplicados (slippage y comisión ida+vuelta).
	•	Igual lector/ENV y mismas ventanas que baseline.

	2.	Si falla gate: mantener familia de señales y aplicar poda suave (subir threshold +0.01–0.02 o REARM_MIN +2) y revalidar en las mismas ventanas.
	3.	Promoción controlada: solo se copia sobre configs/diamante_selected.yaml cuando:

	•	pasa gate en todas las ventanas del período, y
	•	pasa check de no-regresión contra el baseline congelado de la semana, y
	•	hay confirmación manual.

	4.	Opcional (Semanas 3–4): filtro de independencia vs Perla:

	•	--perla_csv <ruta> + --max_corr <umbral> para descartar configs de Diamante demasiado pegadas a Perla (correlación efectiva por exposición).
	•	Diamante se mantiene “francotirador”; Perla sirve de referencia, no se toca (benchmarks).

Entregables de control (por semana):
	•	reports/baseline/diamante_week0.json (snap inicial de KPIs).
	•	reports/experiments/*.csv (candidatos).
	•	reports/check_noregr_last.txt + tabla de comparación.
	•	Si se promueve: copia fechada configs/diamante_selected_YYYYMMDD_HHMM.yaml.

⸻

Uso del gate de correlación (Semanas 3–4, opcional)
	•	Banderas nuevas: --perla_csv, --max_corr
	•	Salida: añade columna corr_perla al CSV de resultados; si corr_perla > max_corr, se descarta la config.
	•	Idea: mantener independencia de señales; Perla como capturadora de ondas largas, Diamante como disparos selectivos.

⸻

Día 4 — Costes realistas (hoy)
	1.	Repetir rejilla corta con 2× slippage (ej. --slip 0.0002) y fee ida+vuelta (ej. --cost 0.0004), mismas ventanas/--freeze_end.
	2.	Gate semanal: PF ≥ 1.5 y WR ≥ 60% con costes → ✅ Semana 1.
	3.	Si no pasa: poda suave (mismos lectores/ventanas) y revalidar.

Modelar costes evita sobreestimar; walk-forward evita mirar el futuro.

⸻

Criterios de salida (Semanas 4–6)
	•	🟢 Paper (riesgo reducido) si durante 4–6 semanas el candidato (o el baseline vigente) mantiene:
	•	PF ≥ 1.15, Sortino ≥ 0.50, MDD ≤ 1.1× baseline, y
	•	pasa no-regresión frente al baseline congelado, con costes.
	•	Si no: mantener baseline actual; registrar aprendizajes para fine-tuning posterior.
