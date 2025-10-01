# sl_tp_defensivo (opt-in) — Runbook
**Estado:** módulo opt-in (OFF por defecto; no baseline) · **Última actualización:** 2025-10-01

**Objetivo:** Añadir **red de protección** en eventos adversos (flash-crash) y tomar beneficios cuando el movimiento se agota, **sin** romper la lógica KISS.

---

## 1) Regla operativa (KISS)
- **Stop-Loss defensivo:** `SL = k_sl × ATR(14)` (por defecto **2.5×**), desde el precio de entrada o último rebalance.  
- **Take-Profit opcional:** `TP = k_tp × ATR(14)` (por defecto **4.0×**) con **breakeven** al tocar `TP1` (opcional).

**Compatibilidad con la base:**
- SL/TP **no generan señales**; solo **gestionan la posición** existente.
- Nunca deben forzar re-entrada inmediata (respetar TTL/reentry_buffer si existe).

---

## 2) Guardarraíles
- **No aumentar FPY** (si lo hace, revisar k y/o desactivar TP).
- **ATR smoothing**: usar **EMA ATR(14)** para evitar cierres por mecha rápida.
- **Suspender TP** en bull fuerte (si `bull_hold=ON`), dejando correr la tendencia.

---

## 3) Parámetros sugeridos (overlay YAML)
```yaml
modules:
  sl_tp_defensivo:
    enabled: false
    atr_period: 14
    k_sl: 2.5            # múltiplos de ATR para SL
    k_tp: 4.0            # múltiplos de ATR para TP
    use_breakeven_after_tp1: true
    smooth_atr_with_ema: true
    respect_ttl_after_exit: true
    disable_tp_if_bull_hold: true
```

---

## 4) KPIs mínimos / Gates
- **MDD ↓** o = baseline, **FPY** estable.
- **ΔNetBTC ≥ +0.02** *o* mejora sustancial de drawdown.
- **SPA/RC** no rechaza; **DSR** positivo.

---

## 5) Protocolo de pruebas
- Ventanas OOS canónicas + FREEZE.
- A/B contra baseline.
- Sensibilidad de `k_sl`, `k_tp` (grid pequeña) bajo **misma** fricción.

---

## 6) Observabilidad
- `sl_active`, `tp_active`, `atr`, `sl_level`, `tp_level`, `last_hit`, `reason_code` en `live_kpis.csv`.
- Log: `logs/mini_accum/sl_tp.log`.

---

## 7) Riesgos
- **Salidas prematuras** por ATR bajo → suavizado EMA + histeresis.
- **Sobreoperar** por TP frecuente → desactivar TP y dejar solo SL.
- **Interferencia** con `bull_hold` → regla `disable_tp_if_bull_hold`.

---

## 8) Estado / Roadmap
- **Estado:** OFF (opt-in).  
- **Roadmap:** `docs/mini_accum/roadmap.md` (v1.1 → v2).
