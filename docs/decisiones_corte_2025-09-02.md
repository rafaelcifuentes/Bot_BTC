# decisiones.md — corte 2025-09-02 (UTC)
**Generado:** 2025-09-03 01:54 UTC

## Decisiones tomadas
1. **Allocator congelado.** Se fija el YAML estable; toda mezcla pasa por este perfil.  
2. **Runner+ oficial.** Usar `allocator_sombra_runner_plus.py` para leer `fee_bps/slip_bps` desde el YAML y emitir breakdown de costes/NET en cada corrida.
3. **Perla → única pata activa de mezcla (Plan B)** hasta que Diamante pase gates. `signals/perla.csv` alimenta al Allocator (rejilla 4h; `longflat`).
4. **Corazón en sombra.** Mantener semáforo, LQ y corr‑gate sólo como diagnóstico; activación real cuando Perla esté OOS‑ready y Diamante re‑aprobado.
5. **Diamante en auditoría.** Se posterga cualquier cambio del “selected” hasta cumplir gates OOS con fricción (PF≥1.6, WR≥60%, ≥30 trades/fold, MDD≤8%).

## Hechos/KPIs recientes
- **Plan B (Perla sola, fees+slip=12 bps):**  
  NET overlay **0.020884**; Gross 0.054750; Costes 0.032640; Turnover **27.2**; Cost share D/P = 0/100.  
  *Validado por* `tests_overlay_check.py` (Diff≈−2.22e−16).
- **Opción A (histórico, referencia):** en otra ventana/config, NET overlay **≈+0.163** con costes; no se usa como baseline actual.
- **Reporter**: KPIs generados en `reports/allocator/sombra_kpis.md`; curvas en `reports/allocator/curvas_equity/`.

## Cambios de configuración (últimos)
- `configs/allocator_sombra.yaml`: mantener **perfil congelado**. Ejemplo recomendado actual:  
  - `rebalance_freq: 4h`, `timezone: UTC`  
  - `costs: fee_bps: 6, slip_bps: 6`  
  - `exec.round_step: 0.20`, `exec.max_delta_weight_bar: 0.10` (si difiere, mantener el congelado que validaste)  
  - `alloc_base: diamante: 0.0, perla: 1.0` (Plan B)  
  - `risk.vol_target_ann: 0.12`, `w_cap_total: 1.20`, clamps y throttle según YAML estable  
  - `corr_gate: enabled: true, lookback_bars: 72, threshold: 0.35, max_penalty: 0.30`
  - `xi_star.cap: 1.65` y **freeze semanal**

## Qué estamos haciendo esta semana
- **Miércoles:** pruebas semanales **Diamante** y **Perla** (grid OOS; actualizar señales de Perla).  
- **Lunes:** snapshot **Corazón** (modo sombra) y refresh de `xi_star.txt`.  
- **Diario (cada 4h):** Allocator en sombra con Perla; verificación con `tests_overlay_check.py`.

## Próximos pasos (NOW / NEXT / LATER)
- **NOW:** mantener Plan B; seguir ampliando grid OOS de Perla y registrar KPIs.  
- **NEXT:** revalidar Diamante (gates + stress costos); si pasa, activar Corazón para mezclar D+P con corr‑gate.  
- **LATER:** micro‑tweaks de coste (round_step/maxΔ) solo tras confirmar que **NET no cae**; evaluar tercera pata si aparece señal no correlacionada.

## Rutas/artefactos de referencia
- Señales: `signals/perla.csv`, `signals/diamante.csv`  
- Corazón: `corazon/weights.csv`, `corazon/lq.csv`, `reports/heart/xi_star.txt`  
- Allocator: `reports/allocator/sombra_kpis.md`, `reports/allocator/curvas_equity/*.csv`

## Repro rápido (último pipeline)
```bash
# Generar/actualizar señales de Perla (ejemplo)
python3 scripts/perla_grid_oos.py \
  --ohlc btc_4h.csv \  --freeze_end 2024-06-30 \  --mode longflat \  --select_by oos_net \  --write_best_signals

# Correr Allocator y breakdown con fees del YAML
python3 scripts/allocator_sombra_runner_plus.py --config configs/allocator_sombra.yaml
# o sólo el breakdown si ya corriste recién:
python3 scripts/allocator_sombra_runner_plus.py --config configs/allocator_sombra.yaml --skip-runner
```
