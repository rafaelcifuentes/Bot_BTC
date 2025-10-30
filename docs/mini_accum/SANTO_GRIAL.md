# Estrategias SANTO GRIAL 🏆 — Mini-Accum KISS v1 (TOP)
**Base canónica (sellada):** DD15 • RB1 • H30 • G200 • BULL0  
**Preset:** `configs/mini_accum/presets/CORE_2025.yaml`  
**Freeze:** `KISSv1_BASE_20250915_1642_final`  
**North Star:** Sats primero. KISS. Trazable. Sin rodeos.

> Si hubiéramos usado solo V1.0 desde 2022, 1 BTC ⇒ **≈5 BTC** a la fecha (WF 22–24 × OOS 2025H1).

---

## One-Pager — OOS 2025H1 · KISS v1 (PROD)
- **Producto WF 2022–2024:** **4.340727**  
- **Composición con OOS 2025H1 (6m):** **4.941751** *(indicativo)*

| Periodo       | Modo | sats_mult | BTC desde 1 BTC | mdd_vs_hodl | flips_total | Source (archivo) |
|---|---|---:|---:|---:|---:|---|
| 2022 (WF)     | WF   | **1.018661** | 1.018661 | 0.000000 | 0 | reports/mini_accum/kiss_v1/WF_2022_kpis__v1_2.csv |
| 2023 (WF)     | WF   | **2.641397** | 2.690688 | 0.936073 | 7 | reports/mini_accum/kiss_v1/WF_2023_kpis__v1_2.csv |
| 2024 (WF)     | WF   | **1.613240** | 4.340726 | 0.768424 | 6 | reports/mini_accum/kiss_v1/WF_2024_kpis__v1_2.csv |
| **Acum. WF**  | —    | **×4.340727** | 4.340726 | — | — | producto 22–24 |
| 2025H1 (OOS)  | OOS  | **1.138462** | 4.941751 | 0.741494 | 2 | reports/mini_accum/base_v0_1_20251011_0320_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv |
| **Acum. total** | —  | **×4.941751** | 4.941751 | — | — | WF × OOS 2025H1 |

### Gate & Decisión (KISS v1 → PROD)
- **NetBTC_OOS > 0** ✔︎  
- **Riesgo consistente** (mdd_vs_hodl ≈ 0.74) ✔︎  
- **Baja rotación** (flips_total = 2) ✔︎  
- **Overlay SL/TP (12×24)**: 0% lift, mismo riesgo ⇒ **se mantiene en experimento (OFF en PROD)**.  
**Conclusión:** **Promovida KISS v1 a `PROD_KISSv1_2025H1` sin SL/TP.**  
**Siguiente:** micro-barridos H31/H32 (RB1; RB2 solo referencia) y repetir gate.

---

## Definiciones rápidas
- **sats_mult**: multiplicador de satoshis del periodo. ROI% ≈ (sats_mult−1)×100.  
- **BTC desde 1 BTC**: capital en BTC si comenzaras con 1 BTC.  
- **mdd_vs_hodl**: drawdown del bot vs HODL ( <1 mejor; =1 igual; >1 peor ).  
- **flips_total**: cambios BUY↔SELL; KISS prefiere **pocos y buenos**.

**Cálculo acumulado**  
2022: 1.018661  
2023: 1.018661 × 2.641397 = 2.690688  
2024: 2.690688 × 1.613240 = 4.340726  
2025H1: 4.340726 × 1.138462 = 4.941751

---

## Parámetros de la base canónica (TOP)
- **Modo:** Price-Trend 1D (EMA21/55) con macro **SMA200** ON  
- **Riesgo:** DD15 (hard_dd_pct = 15%)  
- **Rebalanceo:** **RB1** (1% por flip)  
- **Horizonte (Trend Eligibility):** **H30** (~5 días)  
- **Ejecución:** gamma_bps=200, bull_bias_bps=0 (**BULL0**)  
- **Costes (baseline):** medidos por separado en stress; baseline sellada sin fricción.

---

## FAQ
**¿Por qué 2022 (WF) aporta ~+1.9%?**  
KISS v1 entra con **confirmación** macro/tendencia y evita cazar techos: en pico/corrección prioriza **no ceder sats**. Es consistente con su diseño long/flat y defensa de MDD.

---

## Contrato (referencia operativa)
- **Superset**: v2+ no elimina palancas ganadoras de v1.  
- **Mejor por año** (ε=+0.10%): 2022=1.018661, 2023=2.641397, 2024=1.613240, 2025H1=1.138462.  
- **Lift OOS ≥ +5% vs BASE** y **MDD no peor**.  
- **Anti-NaN** en KPIs. **Trazabilidad**: sufijos, docs, tag/rollback.  
- **Nuevas versiones**: opt-in.

**Mantra:** *Sats primero. Sin apalancamiento. Sin magia. KISS y trazabilidad a rajatabla.*
