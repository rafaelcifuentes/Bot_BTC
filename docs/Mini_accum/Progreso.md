Progreso al 2025-09-09

DONE
	•	Núcleo v0.1 implementado y congelado (macro D1 EMA200, 21/55 4h, salida activa confirmada, pasiva por cruce, dwell, costes).
	•	Normalización de tiempo (timestamp/ts UTC, orden y deduplicación) para evitar KeyError: 'timestamp' y duplicate keys.
	•	Runner y artefactos:
	•	CLI mini-accum-backtest operativo (start/end) + renombrado con sufijo vía REPORT_SUFFIX y/o script rename_last_reports.py.
	•	Diagnóstico de presupuesto semanal (BUY/semana, cap por semana, violaciones = 0).
	•	Presupuesto dinámico por ATR (2-verde / 1-resto): activo y validado (sin violaciones del cap).
	•	Buffer de cruce anti-microcruces signals.cross_buffer_bps (probado 0/10/15/25; preset actual xbuf25).
	•	Trazabilidad: experiments_log.csv + freeze de entorno (env/requirements-YYYYMMDD.txt) + checksums OHLC.
	•	OOS ejecutado 2022–2023 (sin violaciones de cap).
	•	Tag local v0.1-prudente-xbuf25 creado.

NOTAS DE DESEMPEÑO (últimos runs)
	•	Variantes xbuf (dinATR + dwell6): net_btc_ratio ≈ 0.59–0.61, mdd_model ≈ 0.232–0.243, flips/año ≈ 58–69.
	•	Cumple MDD vs HODL (≈0.75–0.80 ≤ 0.85).
	•	No cumple aún: net_btc_ratio ≥ 1.05 ni flips/año ≤ 26 (objetivos del plan).

TODO (por prioridad)
	1.	OOS formal por ventanas del plan (guardar KPIs por ventana):
	•	2022H2, 2023Q4, 2024H1 → tabla con net_btc_ratio, mdd_model, mdd_vs_hodl, flips/año.
	2.	Reducir turnover manteniendo MDD:
	•	Ablations rápidas: dwell 4 vs 6 (actual) y xbuf 25/35.
	•	Probar confirmación de salida más estricta (ej. confirm_bars=2) y/o macro persist (N días > EMA200).
	•	Enforzar hard 26/año en CLI (ya está en core sim; exponer flips_blocked_hard en summary).
	3.	Módulos opt-in (ablation con KPIs OOS):
	•	ATR “pausa amarilla” (slim): debe bajar flips ≥10% o MDD ≥10% con Net_BTC_ratio ≈.
	•	Grace TTL: cooldown suave tras flip; objetivo: turnover −10% con ratio ≈.
	•	Hibernación por chop (≥2 cruces 21/55 en 40 barras).
	4.	Documentar preset “prudente-xbuf25” en el plan (snippet YAML) y dejar BASE separado.
	5.	Integración final del sufijo en CLI (--suffix) y remover duplicado de _rename_last_reports en el runner.
	6.	CI mínima (lint + test de humo) y tests de I/O/EMA/merge D1→4h.
	7.	Git remoto y push del tag (o crear …-r1 si re-anclas).
	8.	Resumen de KPIs en markdown: incluir flips_blocked_hard y deltas vs baseline.

Presets
	•	Preset actual (prudente-xbuf25): dinATR (2/1), dwell=6, cross_buffer_bps=25, yb=5, p=40.
Objetivo: bajar aún más flips/año sin romper MDD; mejorar net_btc_ratio hacia 1.05.

⸻

Resumen ejecutivo

✅ DONE
	•	Core v0.1 congelado y replicable.
	•	Sufijo de reportes automatizado + diagnóstico de cap semanal.
	•	Din-ATR (2/1) funcionando, sin violaciones.
	•	Anti-microcruces (xbuf25) incorporado.
	•	Logging, freeze, checksums; OOS 2022–2023 corrido.

🔜 TO-DO (acción inmediata)
	1.	Correr OOS por ventanas del plan y tabular KPIs.
	2.	Ablations para bajar flips: dwell 6→4/8 y xbuf 25→35.
	3.	Probar macro_persist ligero (ej. 1–2 días > EMA200) y/o confirm_bars=2.
	4.	Exponer flips_blocked_hard en el summary y consolidar --suffix en CLI.
	5.	Pushear remoto + tag.


# TODO (próxima sesión)
- GitHub (SSH): terminar alta de clave y cambiar remoto a SSH; luego `git push` y `git push --tags`.
- OOS formal: correr ventanas 2022H2 / 2023Q4 / 2024H1 con preset prudente xbuf25 y registrar KPIs.
- CLI: integrar `--suffix` directo en `mini_accum/cli.py` (ahora lo cubre rename_last_reports.py).
- Tests: smoke de weekly cap (BUY<=cap) y de cross_buffer_bps.
- Docs: reflejar cross_buffer_bps en plan y YAML (xbuf25) y resultados de ablation xbuf0/10/15/25.

¿Cómo vamos?
	•	Infra/packaging & reproducibilidad: ~85%
Paquete instalable, CLI funcionando, runner con sufijo (rename), logging de experimentos, freeze de entorno, comprobaciones de datos, weekly cap dinámico por ATR y cross_buffer_bps activos.
	•	Core v0.1 (reglas congeladas): ~90%
Macro D-1, 21/55 4h, salida activa confirmada, dwell, costes, presupuesto hard.
	•	Validación cuantitativa (ablation + OOS): ~30–40%
Corridas 2024–2025 y 2022–2023 hechas; falta batería OOS formal (2022H2 / 2023Q4 / 2024H1), consolidar KPIs y anotar en el log.
	•	Docs/CI/tests: ~40%
Plan y progreso empezados; faltan tests unitarios (cap semanal, cross buffer, integridad datos) y CI simple.

Progreso global aproximado: ~60% del proyecto v0.1 “prudente”.

⸻

¿Rinde? (honesto y directo)
	•	Con el preset prudente (dyn-ATR + dwell=6 + xbuf=25) los últimos KPIs que mostraste están en net_btc_ratio ≈ 0.60–0.61 y MDD_model ≈ 0.23 vs HODL 0.306 (≈**-24%** de MDD frente a HODL).
	•	Qué pasa: Mejoramos el drawdown (bien), pero no superamos HODL y el turnover anual sigue por encima del soft/hard (≈57–70/año vs 26 objetivo).
	•	Conclusión hoy: 1/3 de umbrales pasa (MDD ✔️), pero Net_BTC_ratio y flips/año no. Aún no es un bot “rentable vs HODL” según el criterio del plan.

No te vendo humo: no puedo estimar probabilidades ni ROI futuro con responsabilidad hasta tener OOS sólido pasando umbrales. Ahora mismo, con la evidencia mostrada, la probabilidad de batir HODL con este preset es baja salvo que ajustemos.

⸻

Qué falta para “candidato a promoción”
	1.	OOS formal (2022H2, 2023Q4, 2024H1) con el preset prudente y 1–2 variantes (p/yellow-band, dwell=8, xbuf=15/25).
	2.	Reducir flips sin perder MDD: probar grace_TTL o endurecer cross_buffer_bps y/o pausa amarilla (p40±yb3→5).
	3.	Sanity de costes y sensibilidad (±5–10 bps) para ver robustez.
	4.	Tests mínimos (cap semanal ≤, microcruces, datos sin duplicados).

⸻

¿Cuántos días?

No te voy a dar plazos: no doy estimaciones de tiempo. Lo que sí puedo decir es que el bloque crítico es la batería OOS + ajustes de flips; cuando eso pase umbrales, el resto (docs/tests/CI) es ejecución.

⸻

Recomendación práctica (mañana)
	•	Correr OOS con el preset actual y guardar KPIs en experiments_log.csv.
	•	Probar yb=5 (amarillo más ancho) y dwell=8 (o xbuf=15) para intentar –10–20% flips manteniendo MDD ≈.
	•	Registrar todo (rename con sufijo) y actualizar docs/Mini_accum/Progreso.md.

Descansa con la tranquilidad de que la base está sólida; ahora toca “afinar para ganar vs HODL” manteniendo el MDD bajo y los flips dentro del presupuesto.