# Plan semanal — Diamante (swing_4h_forward_diamond)

**Semana actual:** Semana 1 — Robustez (parte 1)  
**Estado general:** ✅ Semana 0 completada | ⚙️ Semana 1 en curso  
**Horizonte total:** 4–6 semanas out-of-sample antes de tocar V2 (Perla Negra) o el allocator.

Plan Operativo Semanal (Semana 1)
	•	Día 1–2: multi-activo (BTC/ETH/SOL) ✅ en curso.
	•	Día 3: micro-grid por fold ✅.
	•	Día 4: costes realistas (hoy).
	•	Día 5: cierre semanal (freeze del viernes), consolidación de KPIs.
---

## Panorama actual (Mapa mental)

**Objetivo:** Swing 4h BTC con **entrada probabilística** + **gestión ATR** (SL/TP1/TP2 y parcial).  
- **Señal:** RandomForest con features **EMA12/EMA26**, **RSI**, **ATR**.  
- **Gestión:** `SL = k · ATR`, `TP1/TP2 = k · ATR`, **parcial en TP1**, **coste/slippage modelado**.

**Datos & Conectividad**  
- Fuente primaria: **ccxt/binanceus** (`BTC/USD`).  
- Fallback **yfinance** bloqueado; **binance global 451** → usamos `EXCHANGE=binanceus`.  
- `_fetch_ccxt` **restaurada** a paginación con `limit=1500`.  
- Flags utilitarios: `--skip_yf`, `--freeze_end`, `--max_bars` (comparación manzana con manzana).

**Validación (últimos runs, 975 velas con freeze al 2025‑08‑05 00:00)**  
- **Net 60–90d:** ≈ **$1.08k–$1.16k** | **PF:** ≈ **2.2–2.4** | **WR:** ≈ **76–80%** | **MDD:** ≈ **2–3%**.  
- Congelando historia (jun/15 y jul/15) para evitar look‑ahead: mantiene **PF > 1.3** y MDD bajo; **edge vs B&H** depende del régimen.  
- Sensibilidad del umbral (**0.58–0.61**): performance similar; **sweet spot 0.60–0.61**.

**Riesgo**  
- **Drawdowns** controlados (< **3%**).  
- **Trades** razonables (**40–76** en 60–90d).

---

## Objetivo de la semana (Semana 1)
- **Validación multi‑activo** (BTC, **ETH**, **SOL** en binanceus spot).  
- **Micro‑grid** de hiperparámetros por fold (threshold / tp/sl / partial).  
- **Costes realistas**: duplicar SLIP y sumar comisión **entrada+salida**.

**Gate para pasar a Semana 2:** en **≥2 activos**, **PF > 1.6** y **WR > 60%** con costes duros (**PF > 1.5**).

---

## Plan detallado (Semana 1)

### Día 1–2 · Validación multi‑activo (Alta)
- Activos: `BTC-USD`, `ETH-USD`, `SOL-USD` (spot, `EXCHANGE=binanceus`, `--skip_yf`).  
- Criterios: **PF > 1.6**, **WR > 60%**, trades ≥ 30 por horizonte (30/60/90).

**Comandos ejemplo**
```bash
EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf   --symbol ETH-USD --period 730d --horizons 30,60,90   --freeze_end "2025-08-05 00:00" --out_csv reports/diamante_eth_week1.csv

EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf   --symbol SOL-USD --period 730d --horizons 30,60,90   --freeze_end "2025-08-05 00:00" --out_csv reports/diamante_sol_week1.csv
```

### Día 3 · Micro‑grid por fold (Media‑Alta)
- Rejilla corta alrededor de: `threshold`, `sl_atr_mul`, `tp1_atr_mul`, `tp2_atr_mul`, `partial_pct`.  
- Elegir por **validación**, no por test.
```bash
EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf   --walk_k 6 --horizons 30,60,90   --best_p_json '{"threshold":0.60,"sl_atr_mul":1.3,"tp1_atr_mul":0.8,"tp2_atr_mul":5.0,"partial_pct":0.70}'   --out_csv reports/diamante_microgrid_week1.csv
```

### Día 4 · Costes realistas (Alta)
- Duplicar **SLIP** y aplicar comisión **entrada+salida**; confirmar **PF > 1.5**.
```bash
EXCHANGE=binanceus python swing_4h_forward_diamond.py --skip_yf   --slip 0.0002 --cost 0.0004 --horizons 30,60,90   --out_csv reports/diamante_costes_week1.csv
----
## Pendientes / próximos pasos (alimentan Semana 2)
- **Walk‑Forward ampliado multi‑años** con mismas flags (ya expuesto).  
- **Curva de capital & volatilidad** del sistema vs **B&H**.  
- **KPI adicionales:** Sharpe/Sortino sobre PnL por trade, time‑in‑market, exposure.  
- **Stress tests:** spreads mayores, latencia, huecos, comisiones variables.  
- **Filtro de régimen** (p. ej., slope EMA/MACD) para evitar shorts en bull puro o subir el umbral.

---

## Entregables y rutas
- Dashboards y métricas: `reports/`  
  - `swing4h_metrics.csv`, `swing4h_1460d.csv`, `swing4h_1460d_wf_*.csv`, `diamante_*_week1.csv`…
- Planes y estado:  
  - `reports/plan_semana.md` (este documento)  
  - `reports/tareas_plan_semana.csv` (tabla de tareas)
