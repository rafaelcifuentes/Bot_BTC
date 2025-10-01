# Mejoras de Alto ROI — Priorización y Detalle

> **Prioridad 0 (global):** alinear TODO el sistema a **caza de satoshis** (numeraire **BTC**).  
> KPIs estándar: **NetBTC / sats_mult**, **CAGR_sats**, **MDD_vs_HODL**. Los KPIs en USD quedan de apoyo durante la transición.

---

## 0) P0 — Migración a numeraire BTC (KPIs NetBTC)
**Objetivo:** que Corazón, Diamante y Perla reporten en BTC (y HODL BTC como benchmark).

**Tareas operativas (orden sugerido):**
- [ ] Agregar columnas de retorno en BTC (p. ej. `ret_*_btc`) en artefactos clave y normalizar nombres.
- [ ] En reportes comparativos (p. ej. `report_heart_vs_baseline`), incluir: **sats_mult**, **CAGR_sats**, **MDD_vs_HODL** y **FPY**.
- [ ] En dashboards/semana, mostrar **HODL BTC** y ratios vs HODL.
- [ ] Mantener PF/MDD en USD como telemetría secundaria durante 2 semanas de transición.

**Criterio de cierre P0:** Todos los resúmenes semanales y comparativas principales muestran métricas en BTC.

  
### Anexo — KPIs en BTC (definiciones rápidas)
  
- **sats_mult (NetBTC):** \(\prod_{t}(1 + r^{BTC}_t)\).
- **CAGR_sats:** \((\text{sats\_mult})^{1/\text{años}} - 1\), con \(\text{años} = \frac{N_{\text{barras}} \times \text{horas\_barra}}{8766}\).
- **MDD_vs_HODL:** en numeraire BTC, HODL≡1 ⇒ coincide con **MDD** de la serie en BTC. *(Si quisieras comparar en USD: usar \(\text{MDD\_estrat} / \text{MDD\_HODL\_USD}\)).*
- **FPY:** flips por año (o tasa de cambio de señales); mantenerlo estable o menor.
  

  
#### Snippet pandas (cálculo de KPIs NetBTC — solo lectura)
  
```python
import pandas as pd
import numpy as np

def kpis_btc(df: pd.DataFrame, ret_col: str = "retP_btc", bar_hours: float = 4.0):
    """
    Calcula sats_mult, CAGR_sats, MDD (en BTC) y FPY (flips/año).
    - ret_col: columna de retornos en BTC (ej.: 'retP_btc', 'ret_4h_btc', 'ret_btc').
    - bar_hours: duración de la barra (4h por defecto).
    """
    s = pd.to_numeric(df[ret_col], errors="coerce").fillna(0.0)

    # sats_mult (NetBTC)
    sats_mult = float((1.0 + s).prod())

    # Años efectivos en base a la resolución temporal
    years = len(s) * (bar_hours / 8766.0)
    cagr_sats = (sats_mult ** (1.0 / years) - 1.0) if years > 0 else np.nan

    # MDD en BTC (HODL BTC ≡ 1, por eso MDD_vs_HODL = MDD)
    eq = (1.0 + s).cumprod()
    dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min())  # negativo; usar abs(mdd) para porcentaje

    # FPY (flips por año): preferir columna de posición si existe; si no, aproximar con signo del retorno
    pos = None
    for c in ("w", "pos", "position"):
        if c in df.columns:
            pos = pd.to_numeric(df[c], errors="coerce")
            break
    if pos is None:
        pos = np.sign(s)
    flips = (np.sign(pos).diff().fillna(0) != 0).sum()
    fpy = float(flips) / years if years > 0 else np.nan

    return {
        "sats_mult": sats_mult,
        "CAGR_sats": cagr_sats,
        "MDD": mdd,
        "FPY": fpy,
    }

# Ejemplo de uso
# df = pd.read_csv("reports/Allocator/perla_for_allocator.csv", parse_dates=["timestamp"])
# # Si tu columna se llama distinto:
# # RET_PREF = ["retP_btc", "ret_4h_btc", "ret_btc"]
# # ret_col = next(c for c in RET_PREF if c in df.columns)
# out = kpis_btc(df, ret_col="retP_btc", bar_hours=4.0)
# print(out)
```
  
**Notas rápidas:**
- En BTC-numeraire, **MDD_vs_HODL = MDD** (HODL BTC ≡ 1).
- Si trabajas con otra resolución, ajusta `bar_hours` (ej.: 1h, 24h).
- Para FPY “real” usa una columna de posición/peso (p. ej. `w` o `pos`). La aproximación por signo de retorno es conservadora.
  

---

## A) Acciones sin cambiar código (solo parámetros / toggles)

| # | Idea                                        | Estado | Beneficio | Esfuerzo | Tiempo | ¿Código? | ¿Solo parámetros? | KPI de aceptación |
|---|---------------------------------------------|:------:|-----------|----------|--------|----------|-------------------|-------------------|
| 1 | **Corazón slim (EMA200+ATR%)** [mini_accum: (2) Regímenes]              | ✅ Prod (ATR_MAX=0.07; grid sugiere 0.08) | Alto | 2/10 | 2/10 | No | Sí | **MDD_overlay ≤ 0.85×** base y **ΔPF ≥ −0.05** *(migrará a NetBTC en P0)* |
| 2 | Grace TTL (dwell/TTL de estado) [mini_accum: (3) Anti-whipsaw]             | ❌ Rechazado (no aportó) | Medio-alto | 2/10 | 2/10 | No | Sí | flips <24–48h ↓ ≥20% **sin perder PF** |
| 3 | **Nudge por alineación** (régimen=señal) [mini_accum: (1) Señales/salidas]    | ✅ Aceptado (sin cambios relevantes) | Medio | 3/10 | 2/10 | No | Sí | ↑ Net BTC con ΔMDD ≈ 0; Δturnover ≤ +5% |
| 5 | **Perla — filtro EMA200 (soft)** (0.990/1) [mini_accum: (2) Regímenes]  | ✅ Toggle base/ema | Medio-alto | 3/10 | 3/10 | No* | Sí* | PF ≈ estable; actividad razonable; NetBTC ≥ base |
| A4| **ATR_MAX mini‑grid** (0.07–0.09) [mini_accum: (2) Regímenes]           | ✅ Hecho (recom. 0.08) | Medio | 2/10 | 2/10 | No | Sí | Mantener PF y bajar MDD **o** ↑actividad útil |

---

## B) Requieren algo de código (pequeño a medio)

| # | Idea                                 | Beneficio | Esfuerzo | Tiempo | ¿Código? | ¿Solo parámetros? | KPI de aceptación |
|---|--------------------------------------|-----------|----------|--------|----------|-------------------|-------------------|
| 4 | Exec adaptativo 2-niveles (agresivo/pasivo por vol) [mini_accum: (4) Costes/Ejecución] | Medio | 5/10 | 5/10 | Menor–Medio | Parcial | ↓ coste/slippage **sin ↑ rejects** |
| 6 | Beta-cap suave (cap por beta a BTC) [mini_accum: (Cerebro/Allocator)]  | Medio     | 5/10     | 5/10   | Medio    | Parcial            | MDD ↓ ~10% con ΔPF ≥ −0.05 |
| 7 | Turnover budget semanal [mini_accum: (3) Anti-whipsaw]              | Medio     | 5/10     | 4/10   | Menor–Medio | Parcial         | Turnover ↓ ≥15% con Net BTC ≥ baseline |
| 8 | Perla ensemble chico (3–5 configs) [mini_accum: (5) Robustez]   | Medio     | 6/10     | 6/10   | Medio    | Parcial            | ↓ var PnL y MDD, PF ≈ estable |
| 9 | Funding tilt en extremos [mini_accum: (9) Moonshot]             | Variable  | 6/10     | 6/10   | Medio    | Parcial            | Mejora en clusters extremos sin dañar régimen normal |

---

## C) Ideas más ambiciosas (planear, no urgentes)
- Corazón **advanced** (LQ + correlación + guardarraíles) como perfil macro v0.2.
- **Cerebro/Allocator v0.2** (vol targeting, Kelly-cap, correlación activa).
- Meta-pesos por régimen (pequeño meta-modelo) tras recolectar logs.

---

## D) Proceso / Claridad (gobernanza rápida)
- ✅ Runner semanal: `heart_monday` (zsh) + `run_heart_slim_pipeline.sh`.
- ✅ Decision log + snapshot tras cada freeze.
- ✅ Toggle Perla por **symlink** (`base` ↔ `ema`).
- ✅ ATR watchdog D1 (porcentaje de días sobre umbrales 0.07–0.09).
- 🔄 Migración de reportes a **NetBTC** (ver P0).

---

## Operativo — Corazón (shadows activos, semana actual)
**Shadows activos:**
- `slim_ema200_atrpct_20250908_cb0965_shadow` (circuit breakers más protectores: `vol_daily_pctl=0.965`, `dd_day=-0.05`).
- `slim_ema200_atrpct_20250908_wfloor045_shadow` (floor `w_floor_on_signal=0.45`).

**KPI de aceptación (fijos):** Aceptar si **|MDD_overlay| / |MDD_base| ≤ 0.85** **y** **ΔPF ≥ −0.05**. *(Post‑P0 pasará a NetBTC)*

**Lunes (00:00 UTC) — lanzamiento semanal**
```zsh
FREEZE="YYYY-MM-DD 00:00"   # lunes 00:00 UTC
ATR="0.08"                  # sugerido (fallback 0.07)

for S in cb0965_shadow wfloor045_shadow; do
  heart_monday "$FREEZE" "$ATR" "$S"
done
```

**Fin de semana — elección de ganador**
1) Abrir `*_vs_base.md` de ambos shadows.  
2) Elegir al que cumpla KPI y tenga mejor combinación (menor MDD_ratio, mayor PF; actividad si empatan).  
3) Promover config en `configs/heart_rules.yaml` (param‑only) y archivar el otro como `shadow/`.

---

## Tabla consolidada de la "última lista" (#1–#9, P1)

| #  | Idea                                                    | Beneficio  | Esfuerzo | Tiempo | ¿Código?     | ¿Solo parámetros? | KPI de aceptación |
|----|---------------------------------------------------------|------------|----------|--------|--------------|-------------------|-------------------|
| 1  | Corazón slim (EMA200+ATR%) [mini_accum: (2) Regímenes]                              | Alto       | 2/10     | 2/10   | No           | Sí                | MDD_overlay ≤ 0.85× base y ΔPF ≥ −0.05 |
| 2  | Grace TTL (dwell/TTL de estado) [mini_accum: (3) Anti-whipsaw]                         | Medio-alto | 2/10     | 2/10   | No           | Sí                | −20% flips <24–48h sin perder PF |
| 3  | Nudge por alineación (sesgo cuando régimen=señal) [mini_accum: (1) Señales/salidas]       | Medio      | 3/10     | 2/10   | No           | Sí                | ↑Net BTC con ΔMDD ≈ 0; Δturnover ≤ +5% |
| 4  | Exec adaptativo 2-niveles (agresivo/pasivo por vol) [mini_accum: (4) Costes/Ejecución]     | Medio      | 5/10     | 5/10   | Menor–Medio  | Parcial           | ↓ coste/slippage sin ↑ rejects |
| 5  | Perla con filtro EMA200 [mini_accum: (2) Regímenes]                                 | Medio-alto | 3/10     | 3/10   | No*          | Sí*               | oos_pf ↑ (o estable) y turnover ↓; oos_net>0 |
| 6  | Beta-cap suave (cap por beta a BTC) [mini_accum: (Cerebro/Allocator)]                     | Medio      | 5/10     | 5/10   | Medio        | Parcial           | MDD ↓ ~10% con ΔPF ≥ −0.05 |
| 7  | Turnover budget semanal [mini_accum: (3) Anti-whipsaw]                                 | Medio      | 5/10     | 4/10   | Menor–Medio  | Parcial           | Turnover ↓ ≥15% con Net BTC ≥ baseline |
| 8  | Perla ensemble chico (3–5 configs) [mini_accum: (5) Robustez]                      | Medio      | 6/10     | 6/10   | Medio        | Parcial           | Vol PnL ↓ y MDD ↓ con PF ≈ estable |
| 9  | Funding tilt en extremos [mini_accum: (9) Moonshot]                                | Variable   | 6/10     | 6/10   | Medio        | Parcial           | Mejora en extremos sin dañar régimen normal |
| P1 | Mini-BOT Acumulación (BTC/Stable, sin short) (hilo aparte) | Medio-alto | 4/10  | 4/10   | Medio        | —                 | — |

---

### Notas
- **Beneficio:** Alto / Medio / Bajo / Variable
- **Esfuerzo / Tiempo (1–10):** menor es más fácil/rápido
- **¿Código?:** No (solo param), Menor (toques chicos), Medio (funciones pequeñas), Alto (módulos nuevos)
- **KPI de aceptación (ejemplos):** mantener PF (ΔPF ≥ −0.05), bajar MDD (≤ −10/15%), bajar turnover, etc.
