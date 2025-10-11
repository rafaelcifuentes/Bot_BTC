## Freeze — KISSv1_BASE_20251010_freeze (one-pager)

**Candidato base:** `DD15_RB1_H30_G200_BULL0`  · **Decisión:** ✅

### NetBTC por ventana
| window  | sats_mult | mdd_vs_hodl | flips_total | fuente KPI |
|:--------|----------:|------------:|------------:|:-----------|
| WF_2022 | 1.018661  | -0.000000   | 0           | `reports/mini_accum/kiss_v1/WF_2022_kpis__v1_2.csv` |
| WF_2023 | 2.641397  | 0.936073    | 7           | `reports/mini_accum/kiss_v1/WF_2023_kpis__v1_2.csv` |
| WF_2024 | 1.613240  | 0.768424    | 6           | `reports/mini_accum/kiss_v1/WF_2024_kpis__v1_2.csv` |

**NetBTC total (producto):** **4.340726883639296**

**Robustez a costes:** el ranking por config es invariante (Spearman=1) en fricción Δbps ∈ [-20, +20].  
**Anti-overfitting (CSCV):** PBO p̂ = 0.285 (n_samples=2048, draws=2048).  
**Tag git:** `KISSv1_BASE_20251010_freeze_NETBTC_4p340727`  
**Snapshot:** `reports/mini_accum/kiss_v1/_snapshots/20251010_171051__NETBTC_4p340727__DD15_RB1_H30_G200_BULL0/`

**Reglas de promoción (v1.0):**
- Promover si **NetBTC_OOS ≥ 1.30** **y** **mdd_vs_hodl ≥ 0.70** en **≥2 ventanas OOS**.
- Robustez a costes: **Spearman ≥ 0.90** del ranking bajo fricción **±10 bps** (mediana `sats_mult` por Δbps).
- Anti-overfitting: **PBO (CSCV) ≤ 0.35** con ≥2048 permutaciones.
- Sanidad operativa: **flips_total ≈ 6–8/año** y **fail_rate ≤ 5%** (si aplica).
- Empaquetar solo si **alias legacy** (`*_v1_2.csv`) y **manifest.json** están en snapshot + tag.

### Reproducibilidad & Rollback
Para rehidratar alias legacy, verificar integridad y, si fuera necesario, volver al tag dorado, ver: `docs/mini_accum/PLAYBOOK_V1_KISS.md`.