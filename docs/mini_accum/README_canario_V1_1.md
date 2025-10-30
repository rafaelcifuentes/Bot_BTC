# Mini-Accum KISS v1.1 — Canario

**Qué cambia**:
- SL/TP defensivo: SL=ATR14*2.0, TP=ATR14*3.5 (conservador).
- Sin leverage, sin short. KISS preservado.

**Guardarraíles**:
- `attest=OK`, señal fresca (no stale), kill-switch si FAIL.
- DRYRUN obligado antes de LIVE.

**Criterio GO (canario 10–20%)**:
- attest == OK
- Señal fresca (latest.json no stale)
- Canario DRYRUN con `ready (signal fresh)` + `[PAPER] flip`.
[
**Rollback**:
- `override_mode: PAUSE`, reemisión señal, restore desde snapshot.

**A/B**:
- Promoción sólo si Δ median(sats_mult) ≥ +0.02 **sin empeorar** median_mdd_vs_hodl ni median_fpy.
]()