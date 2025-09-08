## 📅 Weekly Control — Runbook & KPIs

### ✅ Checklist (semanal)
1. **Actualizar datos** (OHLC/retornos) → todo en **UTC**.
2. **Perla (grid OOS)**  
   - `python3 scripts/perla_mvp_grid.py --ohlc data/BTC-USD_4h.csv --write_best_signals`
   - Verificar `reports/heart/perla_grid_results.csv` y `signals/perla.csv`.
   - Mantén Plan B activo y corre semanalmente el perla_grid_oos.py con un --freeze_end dentro del rango de tu CSV para renovar OOS.
3. **Frozen Allocator (perfil estable)**  
   - `python3 scripts/allocator_sombra_runner.py --config configs/allocator_sombra.yaml`
4. **Verificación independiente**  
   - `python3 tests_overlay_check.py` (Gross, costes, NET, turnover).
   - Diff(calc-curve) ≈ 0 → ✔️
5. **Corazón (modo sombra)**  
   - Generar `corazon/weights.csv` (si aplica) y dejar registros en `reports/heart/`.
6. **Reporte semanal**  
   - `python3 scripts/weekly_kpi_report.py` → `reports/weekly/summary.md`

### 🎯 KPIs a registrar
- **NET base / overlay** (desde curvas)
- **ΔNET** (overlay − base)
- **Vol anual overlay** (mediana de `vol_est_ann`)
- **MDD base / overlay**
- **PF / WR** (overlay) — vía retornos de la curva
- **Turnover total** y **Costes totales** (12 bps por defecto: 6 fees + 6 slip)
- **Cost share D/P**
- **scale@max frac** y **cap binding frac**
- **corr(D,P) por contribución** (eD·retD vs eP·retP)

### ✅ Criterios de aceptación (go/no-go semanal)
- `ΔNET ≥ 0` (con costes)
- **Vol** overlay ↓ **≥10%** vs base **o** **MDD** ↓ **≥15%**
- **PF** overlay no cae más de **5–10%** vs base
- **Diff(calc-curve)** ≈ **0** (consistencia)
- **cap binding** ~ **0%**, **scale@max** razonable (no pegado >70%)

### 🚩 Flags
- Diff(calc-curve) alto → revisar timestamps/UTC o ffill.
- cap binding alto → revisar `w_cap_total`/clamp/xi*.
- scale@max >70% → el piso/estimador de vol quizá muy bajo.
- corr(D,P) > 0.40 sostenido → activar penalización de la pierna débil.