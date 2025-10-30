
### 2025-10-30
- Normalizado OHLC 4h → `timestamp`.
- Arreglado TTL/dwell en `sim.py` (estable) y removal de fillna(method).
- OOS 2025H1 sanity con DWELL=96 → flips ~7 (razonable).
- H31/H32 bajo DWELL=96: **FAIL** por Gate (+5% & MDD≤base) y cláusula D.7.
- KPI Guard OK; Canario DRYRUN=1 GREEN (último log con `[PAPER] flip`).
- Git tag: `MINIACCUM_DWELL_FIX_AND_H31_H32_FAIL_20251030_0020`.
