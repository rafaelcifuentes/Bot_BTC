# Plan Operativo (v2025-09-02) — "Playbook"
**Actualizado:** 2025-09-03 01:54 UTC  
**Ámbito:** semana a semana (6+2), con tracks Perla, Diamante, Corazón y Allocator.

---

## 1) Objetivos operativos
- **Perla:** mantener edge OOS estable; producir `signals/perla.csv` listo para mezcla.
- **Diamante:** recuperar edge con gates exigentes y no‑regresión.
- **Corazón:** correr en **modo sombra** (pesos suaves + LQ + corr‑gate), con bitácora diaria.
- **Allocator:** perfil **congelado**; usar Perla (y luego Diamante) como inputs; reporter y checks de costes/NET.

---

## 2) Cadencia (NOW / NEXT / LATER)
**NOW**
- Perla: grid **IS/OOS** y selección por `oos_net`/`oos_pf`; escribir `signals/perla.csv` (4h).
- Allocator: correr con YAML congelado → `reports/allocator/*` + `tests_overlay_check.py`.
- Diamante: auditoría y rediseño; preparar loader `configs/diamante_selected.yaml`.
- Corazón: snapshot diario (ALL y LONG_V2) + ranking por `pf_60d`.

**NEXT**
- Activar blend real (Corazón) cuando Perla esté OOS‑ready y Diamante pase gates.
- Gate de correlación (ventana 60–90d; thr 0.35–0.40) al mezclar D/P.

**LATER**
- Fine‑tuning de costes sólo si no afecta NET (round_step/maxΔ). Añadir tercera capa si surge señal no correlacionada.

---

## 3) Track Perla — Operativa
### 3.1 Grid IS/OOS y señales
Ejemplo (OHLC 4h local `btc_4h.csv`):
```bash
python3 scripts/perla_grid_oos.py   --ohlc btc_4h.csv   --freeze_end 2024-06-30   --mode longflat   --select_by oos_net   --write_best_signals
```
Salida: `reports/heart/perla_grid_oos.csv` + `signals/perla.csv` (cols: `timestamp,sP,w_perla_raw,retP_btc`).

### 3.2 Preparar para Allocator
```bash
python3 scripts/fix_perla_ret.py --perla signals/perla.csv   --out reports/allocator/perla_for_allocator.csv   --spot signals/diamante.csv   --default_exposure 0 --prefer_default_exposure --align nearest --tolerance 3h
```

### 3.3 Gates Perla (OOS, con costes)
PF ≥ 1.15–1.25; MDD ≤ 15%; corr(D,P) ≤ 0.35–0.40 en neutro/rango; **NET OOS > 0**.

---

## 4) Track Diamante — Operativa (6+2 semanas)
### 4.1 Guardrails OOS (horizonte 60–90d)
- Baseline (W1·D4, costes duros): `MDD_base_60d ≈ 1.70%` (slip=0.0002, cost=0.0004).
- **Gate principal (mediana sobre freezes OOS):** PF_med ≥ 1.50; WR_med ≥ 60%; Trades_med ≥ 30.
- **Drawdown cap dinámico:** `MDD_med ≤ 1.10 × MDD_base_60d` (≈1.87% hoy).
- **Robustez de colas:** PF_min ≥ 1.30.
- **Stress fricción (slip↑, fee↑):** en ≥60% de casos, PF_stress/PF_base ≥0.90 y MDD_stress/MDD_base ≤1.20.

### 4.2 Loader y gate de correlación
- `configs/diamante_selected.yaml` (la única fuente de verdad).  
- Gate opcional: `--perla_csv ... --max_corr 0.35` para descartar configs "pegadas" a Perla.

### 4.3 Comandos de referencia
BTC (ejemplo multi‑horizonte):
```bash
EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf   --symbol BTC-USD --period 730d --horizons 30,60,90   --freeze_end "2025-08-05 00:00"   --out_csv reports/diamante_btc_weekX.csv
```
Backtest grid (con loader activo; placeholders de fechas):
```bash
PYTHONPATH="$(pwd):$(pwd)/scripts" python backtest_grid.py   --windows "2025M01:2025-01-01:2025-01-31"   --ohlc_root data/ohlc/1m   --fee_bps 6 --slip_bps 6   --gate_pf 1.6 --gate_wr 0.60 --gate_trades 30   --out_csv reports/wf_2025M01.csv --out_top reports/wf_2025M01_top.csv
```

---

## 5) Track Corazón — Operativa (4h, diagnóstico)
### 5.1 Pipeline diario
- ALL (sin gates) y LONG_V2 (gates: ADX/FG/Funding).  
- Umbral por defecto `0.60`; `max_bars 975`; FREEZE semanal (lunes).  
- Métrica de ranking: `pf_60d`.

Comandos:
```bash
# Helpers ZSH
source scripts/corazon_cmds.zsh
runC_all_freeze
runC_all_live
runC_long_freeze_v2
runC_auto_daily
runC_auto_status
```
Script directo:
```bash
python corazon_auto.py   --exchange binanceus   --fg_csv ./data/sentiment/fear_greed.csv   --funding_csv ./data/sentiment/funding_rates.csv   --max_bars 975   --freeze_end "YYYY-MM-DD 00:00"   --compare_both   --adx_min 22   --report_csv reports/corazon_auto_daily.csv
```

### 5.2 Programación (macOS cron)
```cron
SHELL=/bin/zsh
30 9 * * * cd ~/PycharmProjects/Bot_BTC &&   source .venv/bin/activate &&   source scripts/corazon_cmds.zsh &&   runC_auto_daily >> logs/corazon_auto.log 2>&1
```

---

## 6) Track Allocator — Operativa
### 6.1 Perfil congelado
Usar YAML estable; ejecutar y verificar:
```bash
python3 scripts/allocator_sombra_runner.py --config configs/allocator_sombra.yaml
python3 tests_overlay_check.py
```
**Runner+** (lee `fee_bps/slip_bps` del YAML y genera breakdown):
```bash
python3 scripts/allocator_sombra_runner_plus.py --config configs/allocator_sombra.yaml
# O, si ya corriste el allocator:
python3 scripts/allocator_sombra_runner_plus.py --config configs/allocator_sombra.yaml --skip-runner
```

### 6.2 Aceptación overlay (cuando se mezcle D+P vía Corazón)
- MDD cartera ↓ ≥ 15% vs sin blend.  
- Vol ↓ ≥ 10%.  
- PF no cae > 5–10%.  
- **NET ≥ baseline** (ideal ↑).

---

## 7) Costes & riesgo (defaults operativos)
- **Fees/slip:** 6 bps + 6 bps (12 bps) — se toma del YAML del allocator.
- **exec.round_step / max_delta_weight_bar:** mantener según YAML congelado (ajustar **solo** en sandbox).
- **ξ***: cap ≈ 1.65–1.70; **freeze semanal**; CB: vol día>p98 o DD día ≤−6% → ξ*=1.0.

---

## 8) Checklists
### Diario / por corrida
- `tests_overlay_check.py` con **Diff≈0**.
- Turnover/cost share razonables (sin picos anómalos).  
- Vol estimator OK; sin "scale infinito"; sin pegue a caps.

### Semanal (IS→OOS)
- **Perla:** grid IS, selección por OOS (`oos_net`/`oos_pf`), escribir `signals/perla.csv`.
- **Diamante:** auditoría/rediseño, OOS 60–90d con costes, **gates** arriba.  
- **Corazón:** reporte semáforo (pesos, dwell, corr rolling) y "what‑if" del blend.

---

## 9) Entregables mínimos
- `reports/heart/perla_grid_oos.csv`, `signals/perla.csv`.  
- `reports/allocator/sombra_kpis.md`, `reports/allocator/curvas_equity/*.csv`.  
- `reports/diamante/*` (OOS + snapshots).  
- `reports/corazon_auto_daily.csv` + métricas por variante.
