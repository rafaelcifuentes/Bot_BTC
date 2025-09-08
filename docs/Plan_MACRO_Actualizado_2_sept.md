Plan Macro (actualizado 2 sept 2025)

0) Foto actual
	•	Allocator (CEREBRO): ✅ congelado con el perfil estable que verificaste (tests_overlay_check alinea NET al céntimo).
	•	Corazón (semáforo): ✅ "saneado" y listo para operar en modo sombra o entrar cuando Perla quede totalmente validada OOS.
	•	Perla (ojo de mercado): ✅ tiene edge con el grid reciente (p. ej. Donchian 40/10 en longflat). Uplift claro del NET overlay ≈ +16.3% en el tramo 4h analizado, con costes de 12 bps (fees+slip) ya descontados y turnover razonable.
	•	Diamante (ojo de mercado): ⚠️ prioridad de auditoría/mejora. El raw audit fue flojo; hay que rediseñar/afinar señales.

⸻

1) Qué cerrar antes del Allocator (contratos & gates)

1.1 Dónde brilla cada uno (validado empíricamente por régimen)
	•	Diamante (4h, breakout–swing)
	•	Mejor: tendencia/expansión (ADX≳20, |slope(EMA50)| alto, ATR% medio/alto).
	•	Peor: rango/chop de baja vol (ADX≲15–20) → fakeouts + costes.
	•	Perla (semanal/4h longflat, contracíclica/estable)
	•	Mejor: rangos, transiciones y tramos ruidosos/bajistas; corr baja vs Diamante en esos regímenes.
	•	Peor: tendencias limpias y fuertes (cede terreno frente a rompimientos sostenidos).
	•	Corazón (semáforo suave)
	•	Pesa de forma continua (histeresis + dwell 6–8 barras + maxΔ peso) en vez de gates binarios.
	•	Debe mejorar MDD/vol de la mezcla vs 50/50 sin hundir PF.

1.2 Gates mínimos (OOS, con costes) — reafirmados
	•	Diamante: (aspiracional y revisable tras el rediseño)
PF ≥ 1.6, WR ≥ 60%, ≥ 30 trades por fold OOS 60–90d, MDD ≤ 8% (en BTC).
	•	Perla: PF ≥ 1.15–1.25, MDD ≤ 15%, corr(D,P) ≤ 0.35–0.40 en neutro/rango, NET OOS > 0.
	•	Corazón overlay (modo sombra): bajar MDD ≥ 15% o Vol ≥ 10% sin degradar PF > 5% ni subir turnover > 20%.

1.3 Contratos de interfaz (evitar peleas con timestamps)
	•	Zona horaria: todo UTC.
	•	Archivos (4h, ffill donde aplique):
	•	signals/diamante.csv → timestamp, sD∈{-1,1}, w_diamante_raw∈[0,1], retD_btc
	•	signals/perla.csv → timestamp, sP∈{-1,1} (o 1 en longflat), w_perla_raw∈[0,1], retP_btc
	•	corazon/weights.csv → timestamp, w_diamante, w_perla (∈[0,1], suma≈1)
	•	corazon/lq.csv → timestamp, lq_flag∈{HIGH_RISK,NORMAL} (histeresis 2 velas)
	•	Freshness: TTL=4h (señal vieja → peso 0).
	•	Costes operativos: fees 6 bps + slip 6 bps (12 bps totales).

⸻

2) Riesgo / operación (perfil congelado + tweaks opcionales)
	•	Objetivo de vol anual: mantener el del perfil congelado que te dio NET +16.3% (no mover ahora).
	•	Clamp & cap: clamp razonable (p. ej. 0.5–1.2) y w_cap_total 1.0 (sin cap binding en las últimas corridas).
	•	xi*: cap ≈ 1.65; freeze semanal + circuit breaker (vol diaria p98 o DD día ≤ −6% → xi*=1.0).
	•	Throttle profit-aware: activado, sin look-ahead (usa contribución t–1).
	•	Vol estimator: EWM span estable; vol floor activo (evita "scale infinito" en vol baja).
	•	Opcional (para rascar costes, probar en sandbox):
	•	exec.round_step: 0.15 (de 0.10)
	•	exec.max_delta_weight_bar: 0.15 (de 0.20)
Valida con tests_overlay_check.py que turnover/costes ↓ y NET ≈ (o mejor).

⸻

3) Semáforo (traducción de regímenes a pesos)
	•	Verde (tendencia clara: ADX≥20, |slope(EMA50)| alto, ATR% medio/alto):
w_diamante = 0.7–0.9, w_perla = 0.1–0.3.
	•	Rojo (rango/compression: ADX<15–20, ATR% bajo):
w_perla = 0.7–0.9, w_diamante = 0.1–0.3.
	•	Amarillo (transición: ADX sube de <15→>20 y EMA50 cambia de signo):
w_diamante ≈ w_perla ≈ 0.5 por unos días, con suavizado.
	•	Nota: en tus datos, los filtros duros por ADX/EMA separan poco → pesos suaves > "ON/OFF".

⸻

4) Cómo lo integramos (con lo nuevo)

Ahora mismo
	1.	Perla ya aporta edge; úsala para alimentar el Allocator (como vienes haciendo).
	2.	Allocator se queda congelado (perfil que validaste con tests_overlay_check.py).
	3.	Corazón corre en sombra: genera pesos y métricas (corr rolling, gate de correlación, dwell), sin afectar producción.

Cuando Perla esté validada OOS (W3–W4)
	•	Activamos Corazón real: mezcla por pesos suaves + gate de correlación
(ventana 60–90d; si corr(D,P) > 0.35–0.40, penaliza hasta 30% a la más débil por performance reciente).
	•	Criterio de go/no-go del blend (con costes):
	•	MDD cartera ↓ ≥ 15% vs sin blend,
	•	Vol ↓ ≥ 10%,
	•	PF no cae > 5–10%,
	•	NET ≥ baseline.

⸻

5) Checklists operativos

Diario/por corrida
	•	tests_overlay_check.py clavado con la curva (Diff ≈ 0).
	•	Turnover total y Cost share D/P razonables (sin picos raros).
	•	vol_est_ann p50, scale@max, cap binding en rangos normales (sin pegue a límites).

Semanal (IS→OOS)
	•	Perla: grid IS, selección por OOS (oos_net/oos_pf), escribir signals/perla.csv.
	•	Diamante: auditoría + rediseño; validar OOS 60–90d con costes; exige tus gates.
	•	Corazón: report de semáforo (pesos, dwell, corr rolling) y "what-if" del blend.

⸻

6) Hitos & métricas de éxito
	•	Corto (ya logrado): NET overlay > +12% con Perla + Allocator congelado; alineación de costes/NET (script vs curva) ✅.
	•	Próximo: con Corazón activo (Perla + Diamante afinado), apuntar a:
	•	MDD ↓ ≥ 15–25% vs mezcla fija,
	•	Vol ↓ ≥ 10–20%,
	•	NET ≥ baseline (ideal ↑),
	•	corr(D,P) controlada (≤ 0.35–0.40 en neutro/rango).

⸻

7) Qué sigue (NOW / NEXT / LATER)
	•	NOW
	•	Mantener Allocator congelado.
	•	Iterar Perla OOS (ampliar grid, longflat probado; intentar variaciones suaves de canales/confirmaciones).
	•	Auditar/rediseñar Diamante (objetivo: PF/WR OOS que pasen tus gates).
	•	NEXT
	•	Corazón en sombra con reportes semanales (pesos, corr gate, dwell).
	•	Si Perla está lista → blend real (w_diamante/w_perla) + gate de correlación.
	•	LATER
	•	Fine-tuning de costes (round_step/max_delta) sólo si no afecta NET.
	•	Añadir una tercera capa si aparece una señal no correlacionada.

⸻

Resumen ejecutivo (literal en "cristiano")

Metiste una Perla con ventaja real. El cerebro (Allocator) ya estaba sano; al alimentarlo con buenas señales, el sistema genera NET positivo y robusto. Corazón está listo para repartir pesos por régimen y bajar el riesgo sin matar el PF, pero lo encendemos "de verdad" cuando Perla quede blindada OOS y Diamante re-aprobado. Con eso, deberíamos subir el NET y bajar MDD/Vol de la cartera, manteniendo costes bajo control.