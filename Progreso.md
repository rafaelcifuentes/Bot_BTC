
### 2025-10-30
- Normalizado OHLC 4h → `timestamp`.
- Arreglado TTL/dwell en `sim.py` (estable) y removal de fillna(method).
- OOS 2025H1 sanity con DWELL=96 → flips ~7 (razonable).
- H31/H32 bajo DWELL=96: **FAIL** por Gate (+5% & MDD≤base) y cláusula D.7.
- KPI Guard OK; Canario DRYRUN=1 GREEN (último log con `[PAPER] flip`).
- Git tag: `MINIACCUM_DWELL_FIX_AND_H31_H32_FAIL_20251030_0020`.

### WF/OOS actualizados

| Periodo | sats_mult | mdd_vs_hodl | FPY | Source |
|---|---:|---:|---:|---|
| 2023 (WF) | 0.491058 | 1.643146565999477 | 20.817357512953368 | base_v0_1_20251030_0511_kpis__WF_2023_CORE.csv |
| 2025H1 (OOS) | 0.779017 | 1.0085263198153618 | 19.667307692307695 | base_v0_1_20251030_0511_kpis__OOS_2025H1_CORE.csv |

**Producto WF 22–24 (+OOS si aplica):** ×0.382542

<!-- 2025-10-30 — B4 sombra y checklist A→B -->
### 2025-10-30 — B4 en sombra: PASS
- Storm guard validado en sombra: `scripts/mini_accum/check_storm.zsh` → **≤1 ejecución/hora** en 24 h.
- Canario: **7/7 GREEN**; KPI Guard: OK; evidencia diaria empaquetada.

**Checklist A → B (estado actual)**
- [x] 7/7 días GREEN (1/h a :07, sin tormentas)
- [x] 7/7 ATTEST OK (≥1 por día)
- [x] gates_pilot_live OK (sin errores tras write_status)
- [x] REPORT.md diario presente
- [x] canary_pack diario no vacío
- [x] Cero DRYRUN=0 con placed/filled en mainnet

> Siguiente acción: preparar *Pilot Live armado* (Etapa B) con ejecutor en modo **ARMED** (no-envía) y evidencia diaria.
