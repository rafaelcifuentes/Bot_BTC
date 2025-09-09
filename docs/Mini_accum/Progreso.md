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
