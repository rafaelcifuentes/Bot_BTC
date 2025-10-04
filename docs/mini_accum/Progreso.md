> Histórico completo (cuando se recupere): ver [`docs/mini_accum/Progreso_HISTORICO.md`](Progreso_HISTORICO.md).
> Atajos: [Ver histórico V1.0](Progreso_HISTORICO.md#v10--histórico-recuperado) · [Ver histórico V2.0](Progreso_HISTORICO.md#v20--histórico-recuperado)

**Actualizado:** 2025-10-04

---

# Progreso al 2025-09-09 — V1.0 (prudente‑xbuf25) y plan V2.0

**DONE**
- Núcleo v0.1 implementado y congelado (macro D1 EMA200, 21/55 4h, salida activa confirmada, pasiva por cruce, dwell, costes).
- Normalización de tiempo (timestamp/ts UTC, orden y deduplicación) para evitar `KeyError: 'timestamp'` y claves duplicadas.
- Runner y artefactos:
  - CLI `mini-accum-backtest` operativo (start/end) + renombrado con sufijo vía `REPORT_SUFFIX` y/o script `rename_last_reports.py`.
  - Diagnóstico de presupuesto semanal (BUY/semana, cap por semana, violaciones = 0).
- Presupuesto dinámico por ATR (2‑verde / 1‑resto): activo y validado (sin violaciones del cap).
- Buffer de cruce anti‑microcruces `signals.cross_buffer_bps` (probado 0/10/15/25; preset actual **xbuf25**).
- Trazabilidad: `experiments_log.csv` + freeze de entorno (`env/requirements‑YYYYMMDD.txt`) + checksums OHLC.
- OOS ejecutado 2022–2023 (sin violaciones de cap).
- Tag local `v0.1-prudente-xbuf25` creado.

**NOTAS DE DESEMPEÑO (últimos runs)**
- Variantes xbuf (dinATR + dwell6): `net_btc_ratio ≈ 0.59–0.61`, `mdd_model ≈ 0.232–0.243`, `flips/año ≈ 58–69`.
- Cumple MDD vs HODL (`≈0.75–0.80 ≤ 0.85`).
- Aún **no cumple**: `net_btc_ratio ≥ 1.05` ni `flips/año ≤ 26` (objetivos del plan).

**TODO (por prioridad)**
1) **OOS formal** por ventanas del plan (guardar KPIs por ventana):
   - 2022H2, 2023Q4, 2024H1 → tabla con `net_btc_ratio`, `mdd_model`, `mdd_vs_hodl`, `flips/año`.
2) **Reducir turnover** manteniendo MDD:
   - Ablations rápidas: dwell 4 vs 6 (actual) y xbuf 25/35.
   - Probar confirmación de salida más estricta (p. ej. `confirm_bars=2`) y/o *macro_persist* (N días > EMA200).
   - Enforzar **hard 26/año** en CLI (ya está en core sim; exponer `flips_blocked_hard` en summary).
3) **Módulos opt‑in** (ablation con KPIs OOS):
   - ATR “pausa amarilla” (slim): debe bajar flips ≥10% o MDD ≥10% con `net_btc_ratio ≈`.
   - Grace TTL: cooldown suave tras flip; objetivo: turnover −10% con ratio ≈.
   - Hibernación por *chop* (≥2 cruces 21/55 en 40 barras).
4) Documentar preset “prudente‑xbuf25” en el plan (snippet YAML) y dejar BASE separado.
5) Integración final del sufijo en CLI (`--suffix`) y remover duplicado de `_rename_last_reports` en el runner.
6) CI mínima (lint + test de humo) y tests de I/O/EMA/merge D1→4h.
7) Git remoto y push del tag (o crear …‑r1 si re‑anclas).
8) Resumen de KPIs en markdown: incluir `flips_blocked_hard` y deltas vs baseline.

**Presets**
- Preset actual (**prudente‑xbuf25**): dinATR (2/1), dwell=6, `cross_buffer_bps=25`, `yb=5`, `p=40`.  
  *Objetivo:* bajar aún más flips/año **sin romper MDD**; mejorar `net_btc_ratio` hacia **1.05**.

---

## Resumen ejecutivo (V1.0)

✅ **DONE**
- Core v0.1 congelado y replicable.
- Sufijo de reportes automatizado + diagnóstico de cap semanal.
- Din‑ATR (2/1) funcionando, sin violaciones.
- Anti‑microcruces (xbuf25) incorporado.
- Logging, freeze, checksums; OOS 2022–2023 corrido.

🔜 **TO‑DO (acción inmediata)**
1) Correr OOS por ventanas del plan y tabular KPIs.
2) Ablations para bajar flips: dwell 6→4/8 y xbuf 25→35.
3) Probar *macro_persist* ligero (ej. 1–2 días > EMA200) y/o `confirm_bars=2`.
4) Exponer `flips_blocked_hard` en el summary y consolidar `--suffix` en CLI.
5) Push remoto + tag.

### TODO (próxima sesión)
- GitHub (SSH): terminar alta de clave y cambiar remoto a SSH; luego `git push` y `git push --tags`.
- OOS formal: correr ventanas 2022H2 / 2023Q4 / 2024H1 con preset prudente xbuf25 y registrar KPIs.
- CLI: integrar `--suffix` directo en `mini_accum/cli.py` (ahora lo cubre `rename_last_reports.py`).
- Tests: *smoke* de weekly cap (BUY≤cap) y de `cross_buffer_bps`.
- Docs: reflejar `cross_buffer_bps` en plan y YAML (xbuf25) y resultados de ablation xbuf0/10/15/25.

**¿Cómo vamos?**
- Infra/packaging & reproducibilidad: **~85%**  
  Paquete instalable, CLI funcionando, runner con sufijo (rename), logging de experimentos, freeze de entorno, comprobaciones de datos, weekly cap dinámico por ATR y `cross_buffer_bps` activos.
- Core v0.1 (reglas congeladas): **~90%**  
  Macro D‑1, 21/55 4h, salida activa confirmada, dwell, costes, presupuesto hard.
- Validación cuantitativa (ablation + OOS): **~30–40%**  
  Corridas 2024–2025 y 2022–2023 hechas; falta batería OOS formal (2022H2 / 2023Q4 / 2024H1), consolidar KPIs y anotar en el log.
- Docs/CI/tests: **~40%**  
  Plan y progreso empezados; faltan tests unitarios (cap semanal, cross buffer, integridad datos) y CI simple.

**Progreso global aproximado:** ~**60%** del proyecto v0.1 “prudente”.

---

## ¿Rinde? (honesto y directo)
- Con el preset prudente (dyn‑ATR + dwell=6 + xbuf=25) los últimos KPIs que mostraste están en `net_btc_ratio ≈ 0.60–0.61` y `MDD_model ≈ 0.23` vs HODL `0.306` (≈ **−24%** de MDD frente a HODL).
- Qué pasa: Mejoramos el drawdown (bien), pero no superamos HODL y el *turnover* anual sigue por encima del soft/hard (`≈57–70/año` vs **26** objetivo).
- **Conclusión hoy:** 1/3 de umbrales pasa (MDD ✔️), pero `Net_BTC_ratio` y `flips/año` no. Aún no es un bot “rentable vs HODL” según el criterio del plan.

**No doy plazos.** El bloque crítico es la batería OOS + ajustes de flips; cuando eso pase umbrales, el resto (docs/tests/CI) es ejecución.

---

## Recomendación práctica (mañana)
- Correr OOS con el preset actual y guardar KPIs en `experiments_log.csv`.
- Probar `yb=5` (amarillo más ancho) y `dwell=8` (o `xbuf=15`) para intentar −10–20% flips manteniendo MDD ≈.
- Registrar todo (rename con sufijo) y actualizar `docs/mini_accum/Progreso.md`.

---

> ℹ️ **Nota**: La siguiente sección (V1.1 y posteriores) ya existía. Se mantiene intacta y continúa debajo.
---

# 2025-10-03 Mini-Accum V1.1 — SL/TP defensivo (ATR) · Promoción a canario (opt-in)

**Resumen**
- Método: ATR(14). SL=2×ATR, TP=3×ATR, TTL reentry configurable.
- Costes: CORE_2025.
- Ventanas evaluadas: 2025-Q3 (neutro), 2023-08 mini-crash (se activa).

**Resultados clave**
- 2025-Q3: Δmult=+0.0000, ΔROI_anual=0.00%, ΔFPY=0 → sin impacto (OK).
- 2023-08: Δmult≈–0.0068 (PASS), ΔROI_anual≈–3.41% (FAIL SLO estricto),
  ΔMDD ↓ (mejora), ΔFPY ≈ +6/año (**FAIL** vs +2/año).

**SLO**
- Pérdida acotada: Δmult ≥ –0.010 ✅ ; ΔROI_anual ≥ –0.03 ❌ (–0.0341)
- Riesgo: ΔMDD ≤ 0 ✅ ; ΔFPY ≤ +2 ❌ (≈ +6/año)
- SPA/Reality-Check: pendiente (criterio PASS ≥ 0.60)

**Decisión**
- Promover a **canario opt-in**, con guardrails:
  - ΔMDD(30d) > 0 → rollback
  - ΔFPY(30d) > +2 → rollback
  - ΔROI_anual(30d) < –4.0% → rollback
  - Mantener CORE sin cambios hasta SPA/RC ≥ 0.60 y 1–2 semanas sin violar guardrails.

### Tabla SLO por ventana

| Ventana (BTC-USD 4h) | Config CAND | Δmult | ΔROI_anual | ΔMDD | ΔFPY | SLO |
|---|---|---:|---:|---:|---:|---|
| 2025-07-01 → 2025-09-13 (Q3-2025) | CORE_2025 + ATR(14) ×{2.5,3.0,3.5,4.0} | +0.0000 | +0.00% | ≈ 0 | 0 | **PASS** |
| 2023-08-01 → 2023-09-30 (mini-crash) | CORE_2025 + **SL=2×ATR**, TP=3×ATR (post) | −0.0068 | −3.41% | ↓ (mejora) | ≈ **+6/año** | **MIXED** (falla ΔROI_anual y ΔFPY) |

> Nota: ΔMDD exacto puede verificarse con el snippet de MDD de más abajo usando las rutas de “Datos usados”.

### Datos usados (reproducibilidad)

**Q3-2025 (2025-07-01 → 2025-09-13)**
- Equity base (CORE): `reports/mini_accum/base_v0_1_20251004_0650_equity__CORE_2025.csv`
- Equity CAND (ATR×3.0): `reports/mini_accum/base_v0_1_20251004_0650_equity__CORE_2025_ATR14x3_0.csv`
- Flips base: `reports/mini_accum/base_v0_1_20251004_0650_flips__CORE_2025.csv`
- Flips CAND: `reports/mini_accum/base_v0_1_20251004_0650_flips__CORE_2025_ATR14x3_0.csv`

**Ago-2023 (2023-08-01 → 2023-09-30)**
- Equity base (CORE): `reports/mini_accum/base_v0_1_20251004_0731_equity__CORE_2025.csv`
- Equity post (SL=2×ATR, TP=3×ATR): `reports/mini_accum/post_20251004_032956_equity____ATR2x3_post.csv`
- Flips base: `reports/mini_accum/base_v0_1_20251004_0729_flips__CORE_2025.csv`
- Flips post: `reports/mini_accum/post_20251004_032956_flips____ATR2x3_post.csv`
- Overlay usado: `configs/mini_accum/presets/_kt_tmp/SLTP_overlay_ATR14_SL2_TP3.yaml`

### Snippet KISS para capturar rutas automáticamente

```zsh
setopt extendedglob nullglob
# Q3-2025
BASE_EQ_Q3=(reports/mini_accum/*_equity__CORE_2025.csv(NOm[1]))
CAND_EQ_Q3=(reports/mini_accum/*_equity__CORE_2025_ATR14x3_0.csv(NOm[1]))
BASE_FLIPS_Q3=(reports/mini_accum/*_flips__CORE_2025.csv(NOm[1]))
CAND_FLIPS_Q3=(reports/mini_accum/*_flips__CORE_2025_ATR14x3_0.csv(NOm[1]))

# Ago-2023 (post)
BASE_EQ_AUG=(reports/mini_accum/*_equity__CORE_2025.csv(NOm[1]))
POST_EQ_AUG=(reports/mini_accum/post_*_equity__*__ATR2x3_post.csv(NOm[1]))
BASE_FLIPS_AUG=(reports/mini_accum/*_flips__CORE_2025.csv(NOm[1]))
POST_FLIPS_AUG=(reports/mini_accum/post_*_flips__*__ATR2x3_post.csv(NOm[1]))

print -r -- "$BASE_EQ_Q3"; print -r -- "$CAND_EQ_Q3"
print -r -- "$BASE_EQ_AUG"; print -r -- "$POST_EQ_AUG"
[[ -s "$BASE_EQ_Q3" && -s "$CAND_EQ_Q3" && -s "$BASE_EQ_AUG" && -s "$POST_EQ_AUG" ]] || echo "[ERR] faltan equities"
```

**Acciones recomendadas**
- Probar ATR SL=**2.5×** (TP=3×) **o** `reentry_ttl=8–12` velas para reducir ΔFPY sin degradar MDD.
- Mantener los guardrails canario; promover a CORE solo si SPA/RC ≥ 0.60 y ΔFPY(30d) ≤ +2.

**Repro**
- `sltp_post.py` y comandos de métricas incluidos en el PR.

- Chequeo rápido de MDD (robusto a rutas vacías):
```python3 - "$BASE_EQ" "$POST_EQ" 2023-08-01 2023-09-30 <<'PY'
import sys, os, pandas as pd
def s(p):
    if not p or not os.path.exists(p): 
        print(f"[ERR] no existe: {p!r}", file=sys.stderr); sys.exit(2)
    df=pd.read_csv(p); ts=pd.to_datetime(df.get('timestamp',df.get('ts')), utc=True)
    eq=df.get('equity_btc', df.get('equity')); return pd.Series(eq.values, index=ts).dropna()
b=s(sys.argv[1]).loc[sys.argv[3]:sys.argv[4]]
p=s(sys.argv[2]).loc[sys.argv[3]:sys.argv[4]]
f=lambda x: (x/x.cummax()-1).min()
print(f"MDD_base={f(b):.2%} | MDD_post={f(p):.2%} | ΔMDD={f(p)-f(b):+.2%}")
PY
```

### Checklist exprés para cerrar V1.1
- Re-ejecuciones **limpias** de **Q3-2024** y **Q2-2025** con **sufijos únicos por ventana** (evita NaN/mismatch).
- **SPA/Reality-Check** en **Ago-2023** y **Q3-2024** (criterio PASS ≥ 0.60).
- **PR final**: overlays congelados, guardrails canario, flag `V1_1_SLTP_DEFENSIVE`, tag `mini-accum-v1.1-canary` y changelog breve.

---

## V1.0 — Resumen y aprendizajes (restaurar/pegar aquí)
**Histórico:** [Ver histórico V1.0](Progreso_HISTORICO.md#v10--histórico-recuperado)

> ⚠️ Este bloque consolida el progreso de **V1.0**. Si ya recuperaste el histórico, pégalo aquí o en `docs/mini_accum/Progreso_HISTORICO.md` y enlázalo.

**Objetivo / alcance (V1.0)**
- [Pendiente de restaurar → ver histórico V1.0](Progreso_HISTORICO.md#v10--histórico-recuperado)

**KPIs (baseline V1.0)**
- [Δmult, ROI anual, MDD, FPY, ventanas evaluadas].
- Fuente: ver histórico V1.0 (equities/flips)

**Decisiones cerradas**
- [lista desde histórico]

**Lecciones y pendientes para V2.0**
- [lista desde histórico]


## V2.0 — Estado y gaps
**Histórico:** [Ver histórico V2.0](Progreso_HISTORICO.md#v20--histórico-recuperado)

**Meta V2.0 (tentativa)**
- [definir aquí el objetivo de V2.0 con una o dos frases]

**Gaps vs V1.1**
- [añade puntos concretos que faltan respecto a V1.1]

**Experimentos planificados**
- [ ] SPA/RC multisets (Ago-2023, Q3-2024) — criterio PASS ≥ 0.60
- [ ] Afinado SL ATR **2.5×** (TP=3×) y/o `reentry_ttl=8–12` — meta: ΔFPY ≤ +2/año sin empeorar MDD
- [ ] Barridos adicionales de ventanas con chop
- [ ] Seguimiento canario 30d con guardrails (ΔMDD ≤ 0, ΔFPY ≤ +2, ΔROI_anual ≥ −4%)


## Plan de trabajo comparativo (V1.0 → V1.1 → V2.0)

| Área | V1.0 | V1.1 (actual) | Target V2.0 | Estado / próximas acciones |
|---|---|---|---|---|
| SL/TP | [histórico V1.0](Progreso_HISTORICO.md#v10--histórico-recuperado) | ATR(14), **SL=2×**, **TP=3×**, `fix_on_entry` | Evaluar **SL=2.5×**; `reentry_ttl=8–12` | En canario; medir ΔFPY y MDD 30d |
| Consistencia (SPA/RC) | [histórico V1.0](Progreso_HISTORICO.md#v10--histórico-recuperado) | **Pendiente** (PASS ≥ 0.60) | PASS multisets | Correr SPA/RC en Ago-2023 y Q3-2024 |
| Rotación (FPY) | [histórico V1.0](Progreso_HISTORICO.md#v10--histórico-recuperado) | ≤ baseline en Q3-2025; ~**+6/año** en Ago-2023 | ≤ baseline **+2/año** | Test TTL y SL 2.5× |
| Drawdown (MDD) | [histórico V1.0](Progreso_HISTORICO.md#v10--histórico-recuperado) | ↓ (mejora) en Ago-2023 | ≤ baseline | Seguimiento canario 30d |
| Operativa / módulos | [histórico V1.0](Progreso_HISTORICO.md#v10--histórico-recuperado) | Sin impacto en Q3-2025 | Estabilidad en chop | Barridos adicionales |

> Cuando recuperes el histórico, reemplaza los campos `[restaurar]` con datos exactos y añade enlaces a los reportes/artefactos correspondientes.

### Histórico anterior (restaurar)
> Este archivo fue sobrescrito con el bloque de V1.1. Para **recuperar TODO lo que había antes** y mantenerlo junto con lo nuevo, usa cualquiera de estos métodos y pega aquí el contenido recuperado (o guárdalo como `docs/mini_accum/Progreso_HISTORICO.md` y enlázalo).

#### Opción A — Git (recomendado)
```bash
# mostrar la versión previa (un commit antes de HEAD)
git show HEAD^:docs/mini_accum/Progreso.md > /tmp/Progreso_old.md

# si quieres una versión de una fecha o commit específico:
git log -- docs/mini_accum/Progreso.md
git show &lt;commit&gt;:docs/mini_accum/Progreso.md > docs/mini_accum/Progreso_HISTORICO.md

# abrir diff para merge manual
git difftool --no-prompt HEAD -- docs/mini_accum/Progreso.md
```

#### Opción B — PyCharm Local History
1. Click derecho sobre `docs/mini_accum/Progreso.md` → **Local History** → **Show History**.  
2. Selecciona la versión anterior y **Revert/Copy** su contenido.
3. Péga el histórico en esta sección o crea `Progreso_HISTORICO.md`.

> Una vez pegado, conserva estructura:
> - `## Histórico hasta YYYY-MM-DD` (contenido previo)
> - `## 2025-10-03 Mini-Accum V1.1 …` (bloque nuevo)

---
