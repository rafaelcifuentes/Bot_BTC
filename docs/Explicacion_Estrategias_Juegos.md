La historia de los dos juegos y los "papelitos"

Imagina que queremos adivinar si la próxima vela sube o baja.
	•	Juego A (justo/SAFE):
El árbitro te da un papelito de ayer (los indicadores calculados con la vela anterior). Con eso decides hoy. No puedes mirar nada de hoy hasta que ya tomaste la decisión.
👉 En código: shift(1) en las features. No hay filtración de datos.
	•	Juego B (tramposo/LEAKY):
El árbitro sin querer te deja ver un papelito de hoy antes de decidir. A veces parece que aciertas más… pero ¡porque viste lo que no debías!
👉 En código: sin shift(1). Hay data leakage.

Cuando entrenas con los dos juegos:
	•	En pruebas cortas, el juego B a veces "brilla" (porque hizo trampa).
	•	En un torneo de verdad (varias rondas con datos que nunca vio, walk-forward), el juego B se desinfla. El juego A no gana todas, pero es el único que podemos confiar.

⸻

Paralelo con lo que hicimos (la versión para mayores)
	•	Papelito de ayer = features con shift(1) → nuestro modo SAFE.
	•	Papelito de hoy = features "en crudo" → modo LEAKY (solo para comparar).
	•	Torneo = walk-forward por tramos (entrenar en pasado A, probar en futuro B; avanzar ventanas).
	•	Marcadores = PF, Win-Rate, Net, MDD…

⸻

Balance de resultados hasta hoy

1) En ventanas recientes "congeladas" (post-features 950–975 velas)
	•	Con SAFE y threshold en 0.58–0.59 (nuestro "sweet spot"):
	•	PF 60–90d ≈ 2.1–3.2 (puntualmente vimos 5.2 en un sweep, pero con menos trades).
	•	Win-Rate 70–79%, MDD ~ −1% a −2.7%, trades 30–40 aprox.
	•	Conclusión: edge claro en esta ventana, coste de riesgo bajo.
	•	SAFE vs LEAKY lado a lado: el SAFE no quedó peor; de hecho, varias corridas SAFE > LEAKY. Bien.

2) En walk-forward (WF) honesto
	•	Seguro (SAFE): resultados mixtos según periodo.
	•	En un WF multi-años (binanceus 2021-2022) vimos PF ponderado ~2.06, Win-Rate ~69.9%, MDD_min ~ −3.8% ⇒ prometedor.
	•	En el WF sobre el tramo reciente (varios folds 2025) el PF_w ~1.1–1.8 y Net_sum flojo en 60–90d, con algún fold duro (MDD hasta ~ −8%).
	•	Lectura: dependemos del régimen (compresión/trending) y del umbral.
	•	Tramposo (LEAKY): se ve "mejor" en WF… justo lo que esperarías si filtra información. Lo descartamos para decisiones.

3) Herramientas y disciplina que ya tenemos
	•	Flags de reproducibilidad: --freeze_end, --max_bars, --skip_yf.
	•	SAFE vs LEAKY en un mismo run + comparativa.
	•	Walk-forward sin errores de índices.
	•	Barrido de --threshold (grid o rango) con CSV de salida.
	•	CSV principal y extendido (ROI% sobre base 10k) guardándose en reports/.
	•	Función _fetch_ccxt estable (binanceus) y yfinance bloqueado por defecto.

⸻

¿Qué nos dice todo esto?
	•	El juego A (SAFE) funciona y puede ser muy bueno cuando el entorno acompaña (PF>2, MDD bajo).
	•	Para generalizar (WF) necesitamos dos llaves:
	1.	Filtro de régimen (evitar operar en compresión o subir umbral con ADX bajo/slope plano).
	2.	Validación multi-activo (ETH, SOL, y si luego metemos funding bias para perp BTC).

⸻

Pronóstico (qué espero al final si seguimos el plan)

Sin promesas, pero siendo realistas con lo visto:
	•	Con filtro de régimen + umbral 0.58–0.59, espero en WF:
	•	PF 60–90d ≈ 1.5–1.8, Win-Rate 62–70%, MDD ≤ 6–8%.
	•	Net estimado anual sobre base 10k (regla de dedo: 60d Net ~500–800 USD → ×6 períodos ≈ 3k–4.8k al año = 30–48% bruto, sin componer, asumiendo condiciones similares y costes realistas).
	•	En ventanas muy favorables puede superar eso; en compresión bajará.
	•	Con allocator futuro (Diamante-Perla) y validación multi-activo, el objetivo es subir la estabilidad más que perseguir el máximo porcentaje.

En resumen:
	•	Nuestro Juego A (Diamante SAFE) ya es sólido en ventanas recientes y honesto metodológicamente.
	•	En torneo largo (WF) aún flaquea en algunos tramos: toca ponerle casco de régimen y validar en ETH/SOL.
	•	Si hacemos esas dos cosas, veo factible cerrar el tramo de 4–6 semanas con un sistema PF~1.6–1.8 en WF, MDD contenido, y ROI anualizable 30–40% en condiciones medias (con picos mejores cuando el mercado trendéa).

¿Siguiente paso mañana? Día 1–2: Validación multi-activo + repetir sweep de threshold (0.56–0.62) y un gate simple de tendencia (subir threshold +0.02 cuando ADX1D < 18 o slope EMA≈0). Con eso ya empezamos a transformar un buen "juego A" en un campeón de torneo.