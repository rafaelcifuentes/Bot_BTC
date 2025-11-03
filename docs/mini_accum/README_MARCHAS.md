# Marchas del BOT todo-terreno

Este documento resume **cuándo usar cada “marcha” (setup)** según el régimen de mercado (semáforo) y define los **gates** de adopción. Está pensado para linkearse desde el índice principal de `docs/`.

---

> 🔗 **Referencias rápidas:**  
> • [PRESETS.md](./PRESETS.md) — lista completa de presets/overlays y cómo correrlos (runners, gates, freeze).  
> • `scripts/mini_accum/run_with_overlay.zsh` — fusiona preset base + overlay M3 para pruebas A/B.  
> • `scripts/mini_accum/ab_m3_check.zsh` — gate neto (borrow/funding) + semáforo en bull.

---

## Marchas y uso recomendado
El bot opera con tres “marchas” según régimen:

- **M1 — Terreno bear/shock (Año+2)** → **`E1_Y2`**  
  Para correcciones fuertes o fase post-cima. Prioriza protección y drawdown bajo.

- **M2 — Rango / Neutral (2023–2024)** → **`CORE_2025`**  
  Base estable para laterales/compresiones. Control de flips y MDD contenido.

- **M3 — Bull claro (2025)** → **`bull_hold_ext`** (freeze H1-2025)  
  Mantener largos con guardrails en bull marcado.  
  **Freeze canónico (H1-2025):** `sats_mult ≈ 1.3554494`, `mdd_vs_hodl ≈ 0.360052`, `flips = 0` ✅  
  **Regla:** usar **M3 solo cuando** el semáforo esté **verde** (D1 > EMA200 **y** ADX ≥ 20).

  **Situación H2-2025 (a la fecha nov, 2025):** régimen plano. El overlay neto **no** pasó el gate (el coste se lo comió). Operamos con **M2/M1** hasta que el régimen cambie. Nota: nov-dic tienden a ser meses alcistas ⇒ dejamos **rutina A/B** para promover M3 si aparece bull real.

---

## Semáforo → Marcha sugerida

| Régimen / Semáforo                 | Condiciones (D1/ADX)                                   | Marcha | Preset/Modo              |
|------------------------------------|--------------------------------------------------------|--------|--------------------------|
| **Bear / Shock**                   | `close ≤ EMA200` **o** (`close < EMA200` con ADX<20)   | **M1** | `E1_Y2`                  |
| **Rango / Neutral**                | `close > EMA200` **y** `ADX14` moderado (**&lt; 20**)  | **M2** | `CORE_2025`              |
| **Bull claro**                     | `close > EMA200` **y** `ADX14 ≥ 20`                    | **M3** | `bull_hold_ext` (freeze) |

## Presets/Modos – Resumen ejecutivo

### M1 — `E1_Y2` (Año+2; defensivo en bear/shock)
- **Objetivo:** proteger satoshis en caídas fuertes y fases post‑cima; reducir MDD vs HODL.
- **Cuándo usar:** D1 ≤ EMA200 **o** pérdida de EMA200 con ADX bajo/moderado.
- **Lógica base (KISS):** EMA21/EMA55 4h con filtro macro D1 (EMA200). Salidas **tempranas** en rojo; sin bull‑bias.
- **Parámetros típicos:** DD≈15%, RB=1, H=30 barras; costes 2/1 bps. Presupuesto de flips **≤26/año**.
- **Fortalezas:** MDD contenido, disciplina de salida, robusto en shocks.
- **Riesgos:** puede perder parte de rallies de transición; sensible a whipsaw en rebotes cortos.
- **Gobernanza:** FREEZE semanal; pasar a **M2** cuando D1&gt;EMA200 y el ADX comience a subir de forma sostenida.

### M2 — `CORE_2025` (neutral/rango; baseline estable)
- **Objetivo:** navegar laterales y compresiones manteniendo estabilidad y reproducibilidad.
- **Lógica base (KISS):** cruces EMA21/EMA55 4h + macro D1 (EMA200); sin bull‑bias ni leverage.
- **Preset canónico:** `DD15_RB1_H30_G200_BULL0` (costes 2/1 bps).
- **Fortalezas:** comportamiento estable, control de flips/MDD, fácil de auditar (freeze/csv/docs).
- **Riesgos:** bajo‑captura en tendencias muy fuertes; reacciona con cierto retraso a cambios bruscos.
- **Gobernanza:** continuar como **marcha por defecto** salvo que el semáforo active M1 o M3; FREEZE semanal y A/B cuando corresponda.

### M3 — `bull_hold_ext` (solo bull claro; **freeze H1‑2025**)
- **Objetivo:** **mantener largos** mientras el bull está confirmado, minimizando decisiones.
- **Gate de activación:** D1&gt;EMA200 **y** ADX≥20; si pierde condiciones → volver a M2/M1.
- **Freeze canónico H1‑2025:** `sats_mult≈1.3554494`, `mdd_vs_hodl≈0.360052`, `flips=0` ✅.
- **Coste neto (ejemplo 180d, borrow 10%):** divisor≈1.048 ⇒ net≈**1.29** (~30%).
- **Política:** **inmutable**; no usar scripts de “repair” sobre estos CSV; usar A/B para candidatos nuevos.

---

## Gates y gobernanza

- **Metric gate M3 (A/B neto):**
  - Neto ≥ **1.05** (ajustado por `borrow_apr`/`funding_apr`)
  - `bull_pct` (porcentaje de barras D1 en bull) ≥ **0.90**
- **Freeze M3 (H1-2025)** es **fuente de verdad inmutable**. No se re-escribe ni se “repara”.
- Si **falla A/B** (ej. H2-2025 plano, coste se come el edge) ⇒ **mantener M2/M1** hasta que el régimen cambie.

---

## Chequeo A/B rápido (no destructivo)

```zsh
# Ejemplo ya probado (H2-2025):
ROOT="$HOME/PycharmProjects/Bot_BTC"
EQ_H2="$ROOT/reports/mini_accum/base_v0_1_20251102_0148_equity__OOS_2025H2_m3_try.csv"

# A/B M3 neto con gates:
"$ROOT/scripts/mini_accum/ab_m3_check.zsh" "$EQ_H2" 0.10 0.00 --min-net 1.05 --min-bull-pct 0.90
echo "[RC]=$?"   # 0=PASS (promover), 1=FAIL (seguir M2/M1)
```
→ Resultado H2-2025 (a la fecha): net < 1.05 ⇒ **FAIL**. Mantener M2/M1 hasta cambio de régimen.

---

## Anexos

**Freeze canónico M3 (H1-2025):**
- KPIs: `reports/mini_accum/base_20251101_060922_kpis__OOS_2025H1_bullhold_ext.csv`
- Equity: `reports/mini_accum/base_20251101_060922_equity__OOS_2025H1_bullhold_ext.csv`
- Freeze dir (manifest + hashes): `reports/mini_accum/_freezes/M3_2025H1_bullhold_ext_20251101_133227/`
- **Métrica sellada:** `sats_mult≈1.3554494`, `mdd_vs_hodl≈0.360052`, `flips=0`.
- **CSV origen:**
  - `reports/mini_accum/base_20251101_060922_equity__OOS_2025H1_bullhold_ext.csv`
  - `reports/mini_accum/base_20251101_060922_kpis__OOS_2025H1_bullhold_ext.csv`
- **Preset/Overlay usados (base documentaria):**
  - Preset base: `configs/mini_accum/presets/CORE_2025.yaml`
  - Overlay M3: `configs/mini_accum/overlays/bull_hold_levered.yaml`
- **Freeze (manifest & hashes):**  
  `reports/mini_accum/_freezes/M3_2025H1_bullhold_ext_20251101_133227/`
**Backlink:** Ver también [PRESETS.md](./PRESETS.md) para runners, gates A/B y lista de presets.
Ejemplo H1-2025 (≈180 días, borrow_apr=0.10, funding=0.00): divisor ≈ 1.048 ⇒ neto ≈ **1.29** (~30%).

> **Política:** el freeze es **inmutable** (no ejecutar reparaciones sobre sus CSV).

---

### Cálculo neto (coste de borrow/funding)
Si el equity gross no incluye carry (≈ *days* días):
sats_mult_neto ≈ sats_mult_gross / ((1+borrow_apr)^(days/365) * (1+funding_apr)^(days/365))
Ejemplo H1-2025 (≈180 días, borrow_apr=0.10, funding=0.00): divisor ≈ 1.048 ⇒ neto ≈ **1.29** (~30%).

> **Política:** el freeze es **inmutable** (no ejecutar reparaciones sobre sus CSV).

---

## Resumen 2022–2025 (sats_mult por período)

> Base 1 BTC; valores a la fecha **2025‑10‑31** para H2‑2025.

| Período   | Semáforo            | Marcha | Preset/Modo             | sats_mult |
|-----------|---------------------|--------|-------------------------|----------:|
| 2022      | Bear / Shock        | M1     | `E1_Y2`                 | 2.921250 |
| 2023      | Rango / Neutral     | M2     | `CORE_2025`             | 2.641397 |
| 2024      | Rango / Neutral     | M2     | `CORE_2025`             | 1.613240 |
| 2025 H1   | Bull claro          | M3     | `bull_hold_ext` (freeze)| 1.355449 |
| 2025 H2   | Rango (plano)       | M2     | `CORE_2025`             | 1.027582 |

> Nota: los acumulados y detalle por ventanas están en **PRESETS.md**.

---

## Evidencia reciente (H2-2025)

- **CORE_2025 (M2):** `sats_mult ≈ 1.0276`, `mdd_vs_hodl ≈ 0.5804`, `flips=6`.
- **E1_Y2 (M1):** `sats_mult ≈ 1.0021`, `mdd_vs_hodl ≈ 0.6116`, `flips=17`.
- **M3 candidato (overlay levered, ventana H2):**  
  A/B: `gross ≈ 1.0283` → **net ≈ 0.9986** (112.3 días, borrow 10%) · bull_pct=1.00 → **FAIL** (net < 1.05).

**Decisión:** mantener **M2/M1** en H2; promover **M3** solo si el semáforo gira a bull y el A/B neto pasa el gate.

---

## Procedimiento A/B para M3

1) **Correr overlay candidato** (no toca el freeze canónico):
```zsh
ROOT="$HOME/PycharmProjects/Bot_BTC"
PRESET="$ROOT/configs/mini_accum/presets/CORE_2025.yaml"
OVER="$ROOT/configs/mini_accum/overlays/bull_hold_levered.yaml"
SUF="OOS_2025H2_m3_try"
"$ROOT/scripts/mini_accum/run_with_overlay.zsh" "$PRESET" "$OVER" 2025-07-01 2025-12-31 "$SUF"
```

2) **Chequear gate neto y semáforo (A/B):**
```zsh
EQ=$(ls -1t "$ROOT"/reports/mini_accum/*_equity__${SUF}.csv | head -n1)
"$ROOT/scripts/mini_accum/ab_m3_check.zsh" "$EQ" 0.10 0.00 --min-net 1.05 --min-bull-pct 0.90
echo "[RC]=$?"   # 0=PASS, 1=FAIL
```

3) **Decisión:**
- Si **PASS** → congelar candidato (nuevo freeze en `reports/mini_accum/_freezes/`) y registrar decisión en `reports/mini_accum/_decisions/`.
- Si **FAIL** → mantener **M2/M1** y reintentar solo si cambia el régimen (semáforo).

---

## Reglas de gobernanza
	•	No mutar el freeze M3 H1-2025. Es fuente de verdad y auditoría.
	•	repair_kpi_from_equity.zsh se usa solo para nuevos equity CSV (para derivar su KPI). Nunca sobre el freeze canónico.
	•	Mantener FREEZE semanal y decisiones A/B en:
	•	reports/mini_accum/_freezes/
	•	reports/mini_accum/_decisions/

**Nota:** El freeze es inmutable (no ejecutar reparaciones sobre estos CSV).

---

TL;DR
	•	M1: E1_Y2 (defensiva) • M2: CORE_2025 (neutral) • M3: bull_hold_ext (solo bull real).
	•	Freeze M3 H1-2025 = 1.355 / 0.360 / 0 flips (sellado).
	•	H2-2025: overlay neto FAIL → seguimos con M2/M1.
	•	Monitorear semáforo y correr A/B: promover M3 si net ≥ 1.05 con bull_pct alto.
	•	Docs relacionadas: **README_MARCHAS.md** ⇄ **PRESETS.md** (navegación cruzada).

### Anexo — Resultados todos los años por marcha (GROSS y NET)

> Base 1 BTC. **NET = GROSS** salvo cuando hay coste de carry (borrow/funding) — p.ej. en M3 (bull_hold_ext).

| Período | Semáforo        | Marcha | Preset/Modo                  | sats_mult (GROSS) | sats_mult (NET)* | mdd_vs_hodl | flips | KPI / Evidencia                                                                 |
|---------|------------------|--------|------------------------------|------------------:|-----------------:|------------:|------:|----------------------------------------------------------------------------------|
| 2022    | Bear / Shock     | M1     | `E1_Y2`                      | **2.921250**      | 2.921250         | 0.104540    | 8     | `reports/mini_accum/base_v0_1_20251013_0231_kpis__OOS_2022_E1.csv`               |
| 2023    | Rango / Neutral  | M2     | `CORE_2025`                  | **2.641397**      | 2.641397         | 0.936073    | 7     | `reports/mini_accum/base_v0_1_20251014_1509_kpis__OOS_2023_REGIME.csv`           |
| 2024    | Rango / Neutral  | M2     | `CORE_2025`                  | **1.613240**      | 1.613240         | 0.768424    | 6     | `reports/mini_accum/base_v0_1_20251014_1509_kpis__OOS_2024_REGIME.csv`           |
| 2025 H1 | Bull claro       | M3     | `bull_hold_ext` (freeze)     | **1.355449**      | **~1.293214**    | 0.360052    | 0     | `reports/mini_accum/base_20251101_060922_kpis__OOS_2025H1_bullhold_ext.csv`      |
| 2025 H2 | Rango (plano)    | M2     | `CORE_2025` (OOS ventana H2) | **1.027582**      | 1.027582         | 0.580359    | 6     | `reports/mini_accum/base_v0_1_20251102_0227_kpis__OOS_2025H2_core.csv`           |

\* **NET** solo difiere de **GROSS** cuando hay coste de carry (p.ej., M3 con leverage/borrow). Para H1-2025 (≈180 días, `borrow_apr=0.10`, `funding_apr=0.00`) se usó:  
`net ≈ gross / ((1+borrow_apr)^(days/365) * (1+funding_apr)^(days/365))` ⇒ **~1.293214**.

> 📦 **Inventario de artefactos (BOM):** ver [ARTIFACTS.md](./ARTIFACTS.md) para presets, overlays, runners, freezes y datos (con SHA256 y última modificación).

> **2025-H2 (auto-gate):** M3 **FAIL** (net 0.9986 < 1.05; bull_pct 1.00 PASS).  
> Marcha activa: **M2/CORE_2025**. Se revisa semanalmente; promover M3 sólo si net ≥ 1.05.

> **Chequeo H2-2025 (bull-hold puro):** ejecutado con `overlays/bull_hold_puro.yaml`.  
> Gate: net ≥ 1.05, bull_pct ≥ 0.90.  
> Decisión arriba en `reports/mini_accum/_decisions/`.

### Estado M3 en H2-2025
- M3 (bull-hold puro) **FAIL**: net < 1.05 aun con bull_pct=1.00 ⇒ **no se promueve**.
- Marcha activa: **M2 (CORE_2025.yaml)** hasta nuevo aviso.
- Re-test semanal automatizado: sólo promover si **net ≥ 1.05** y **bull_pct ≥ 0.90**.

### Estado Q4-2025
- M3 (bull_hold_ext): **PAUSADO** en H2-2025; gates net≥1.05 no superados (último net=1.02835 puro / 0.99862 con coste).
- Marcha por defecto: **M2 (CORE_2025)**; en shock: **M1 (E1_Y2)**.
