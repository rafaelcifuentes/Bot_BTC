cat > docs/mini_accum/cooldown_after_loss.md <<'MD'
# cooldown_after_loss (opt-in) — Runbook
**Estado:** opt-in (OFF por defecto) · **Objetivo:** tras una pérdida, imponer un enfriamiento temporal para evitar reentradas impulsivas, reducir whipsaw y FPY, y mejorar la estabilidad.

## 1) Regla operativa (KISS)
- Al cerrarse una operación con **pérdida significativa** (≥ `min_loss_frac_to_trigger`), activar un **cooldown** de `cooldown_bars_4h` velas 4h durante el cual:
  - **No** se permiten **nuevas entradas**.
  - Sí se permiten **salidas/TP/SL** si hay posición abierta.

**Extensión adaptativa (opcional):**
- Si hay ≥ `recent_losses_threshold` pérdidas en la ventana `recent_losses_window` (trades), el cooldown se **extiende** por `extend_factor`×.
- Se puede exigir **macro reconfirmación** (`require_macro_flag`) para terminar el cooldown (p.ej., D1>EMA200).

## 2) Guardarraíles
- **No aumentar FPY**; si sube, rollback.
- **Ignorar micro-pérdidas** (< fricción total).
- **Compatibilidad TTL**: el cooldown **se suma** al TTL base (no lo reduce).
- **No bloquea salidas**: solo bloquea **entradas**.

## 3) Parámetros sugeridos (overlay YAML)
```yaml
modules:
  cooldown_after_loss:
    enabled: false
    cooldown_bars_4h: 12
    min_loss_frac_to_trigger: 0.001
    recent_losses_window: 3
    recent_losses_threshold: 2
    extend_factor: 2.0
    require_macro_flag: false
    publish_reason_codes: true