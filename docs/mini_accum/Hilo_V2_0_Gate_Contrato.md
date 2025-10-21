# Mini-Accum KISS v2.0 — Hilo Nuevo (OPT-IN)

**Mantra:** KISS, trazable, sin romper el core de v1. **Objetivo:** explorar V2.0 **sólo si** pasa el 🧪 **Gate & Contrato**.

## Reglas (Gate & Contrato)
1) **Superset v1** (no eliminar palancas que suman sats).  
2) **NetBTC por año ≥ v1** (ε=+0.10%).  
3) **Lift OOS ≥ +5%** vs BASE.  
4) **MDD no peor** que BASE.  
5) **Anti-NaN** en KPIs.  
6) (Estricto) **Spearman ≥0.95** y **PBO ≤0.30** cuando aplique.  
7) **Fricción**: si FPY_cand > 2× FPY_base y lift < +5% ⇒ **FAIL**.  
8) **Trazabilidad**: sufijos claros, docs y tag/rollback.  
9) **Nuevas versiones = OPT-IN** (v1 sigue en PROD).

## Pre-flight
```bash
# OHLC 4h apuntando al WF vigente
mkdir -p data/ohlc/4h
ln -sf ../../tmp_wf/BTC-USD_4h_WF_2025.csv data/ohlc/4h/BTC-USD.csv
head -2 data/ohlc/4h/BTC-USD.csv

# Instalar/actualizar paquete
pip install -e packages/mini_accum

# Sincronizar YAML v2 con TOP (DD15, RB1, H30, G200, BULL0, tag)
bash scripts/mini_accum/dev/sync_v2_top.sh
