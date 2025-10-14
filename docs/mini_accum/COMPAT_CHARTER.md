mkdir -p docs/mini_accum
cat > docs/mini_accum/COMPAT_CHARTER.md <<'MD'
# COMPAT CHARTER — Mini-Accum KISS

## Principios
- **Superset por defecto:** toda versión ≥ v1 mantiene las palancas/lógica que ya acumularon SATS.
- **Defaults bloqueados al último freeze:** presets `*_compat` replican alias estable (v1_2).
- **Alias/artefactos estables:** KPIs/Equities `reports/mini_accum/kiss_v1/WF_*.csv` y snapshots con `manifest.json` + SHA256.
- **Reproducibilidad:** `docs/mini_accum/PLAYBOOK_V1_KISS.md` (rehidratación, hashes y rollback).
- **Gates anti-regresión:** `compat_guard_v1(cur, snap, tol)`; stress de costes (Spearman) y PBO/CSCV documentados en cada freeze.
- **Breaking changes:** RFC → feature flag OFF → A/B → SPA/RC → freeze + tag → migración de presets.

## Operativa
1. **Baseline:** define `BASELINE_SNAP` (snapshot/tag del freeze vigente).
2. **Tolerancia:** `NETBTC_TOL_PCT` (p.ej. 0.02 = −2% permitido).
3. **Guard:** el pipeline invoca `compat_guard_v1` y aborta si `cur_netbtc < base*(1−tol)`.
4. **Freeze:** snapshot + manifest con hashes, tag anotado y One-Pager actualizado.
5. **Promoción:** sólo si pasa OOS, costes y anti-overfitting del Charter.

## Checklist de FREEZE (6 líneas)
```bash
export FREEZE_VERSION="KISSv1_BASE_$(TZ=America/New_York date +%Y%m%d)_freeze"
export CANDIDATE="DD15_RB1_H30_G200_BULL0"
export NETBTC_TOL_PCT=0.02
export BASELINE_SNAP="reports/mini_accum/kiss_v1/_snapshots/20251010_171051__NETBTC_4p340727__DD15_RB1_H30_G200_BULL0"
bash scripts/mini_accum/kiss_v1_wf_pipeline.sh
git tag -n | grep "$FREEZE_VERSION"