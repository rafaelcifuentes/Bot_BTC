Plan_operativo.md (actualizado)

Plan operativo (6+2 semanas) — Diamante primero, Corazón después
Estado: Diamante en curso · Corazón AGENDADO (no ejecutar en paralelo).
Premisa: Perla Negra (V2) permanece intocable; sólo se registran series históricas (posición/señal) para medir correlación con Diamante y filtrar solapamientos.

Alcance y orden
	1.	Fase A — Diamante (semanas 0–6)
Swing 4h con ATR (SL/TP1/TP2 + parcial), validación OOS forward-chaining (tipo TimeSeriesSplit) y costes realistas.  ￼
	2.	Fase B — Corazón (C1–C2, después de Diamante)
Intradía 4h por probabilidad + gates (ADX/FG/Funding), sin ATR. Se corre después.

Regla: Durante Fase A NO se ejecutan tareas de Corazón; Perla sólo aporta una serie auxiliar para control de correlación / independencia de señales (diversificación).  ￼

⸻

Novedades (lo que pediste integrar)

0) Mini-loader: el bot leerá configs/diamante_selected.yaml y activará la última selección (horizon, threshold, parciales, etc).

1) Especificación del CSV de Perla (benchmark silencioso)
Formato esperado por --perla_csv (timestamps en UTC):

ts,exposure,label
2024-10-01 00:00:00+00:00, 1, LONG
2024-10-01 04:00:00+00:00, 0, FLAT
...
	•	exposure ∈ {−1, 0, +1} (o {0,1} si sólo long).
	•	Puede aceptar signal∈[0,1] en vez de exposure; si existe ambas, usa exposure.

Gate de correlación (--max_corr)
	•	Calcula correlación efectiva entre exposición de Diamante y serie exposure (o signal) de Perla dentro de la ventana del fold.
	•	Por defecto usa Pearson; si el usuario pasa --corr_method spearman, usa Spearman.
	•	Si |corr| > max_corr ⇒ descarta la config para ese fold (se marca corr_gate=0 en el CSV y no se considera en “winners”).

2) Banderas nuevas (para run_grid_tzsafe.py y backtest_grid.py)
	•	--perla_csv PATH
	•	--max_corr FLOAT (p. ej., 0.65)
	•	--corr_method {pearson,spearman} (opcional, default pearson)

3) Snippet de correlación efectiva
	•	Construye exposición de Diamante (vector E_diamante en {0,1} o {−1,0,1}) a partir de fills si están disponibles, o de la señal (threshold+rearm+histeresis) como fallback.
	•	Alinea (reindex) con la serie de Perla E_perla en la misma rejilla temporal; relleno ffill y muestreo a 4h si procede.
	•	Requiere mínimo N_overlap ≥ 60 barras para reportar corr_perla; si no, NaN y no aplica gate.
	•	Escribe corr_perla y corr_gate columna en los CSV de salida (kpis/val).

4) Checklist operativo semanal + métricas
	•	Lunes: Congelar FREEZE, correr Diamante multi-activo (BTC/ETH/SOL) y generar diamante_*_weekX.csv.
	•	Miércoles: Micro-grid por fold (en Diamante) con el mismo ENV (walk-forward / rolling origin).  ￼
	•	Jueves: Costes “duros” (2× slip + fee IN/OUT).
	•	Viernes: Consolidar KPIs y exportar:
pf, wr, trades, mdd, sortino, net, corr_perla, corr_gate.
	•	Gate semanal (seguir igual que ya definiste): p.ej., PF>1.5 con costes duros.
	•	Nota: Mantener --perla_csv para filtrar configs demasiado correlacionadas (diversificación).  ￼

5) Patches exactos para: argparse + lectura Perla + exposición Diamante + corr_perla en CSV (abajo).

⸻

Referencias rápidas
	•	Forward-chaining (TimeSeriesSplit) recomendado para series temporales (sin mezcla y respetando causalidad).  ￼
	•	Backtest overfitting: considerar métricas “robustas” y walk-forward; literatura clásica (Bailey & López de Prado; DSR / sharp razor).  ￼ ￼
	•	Diversificación: el beneficio cae con alta correlación entre estrategias/activos; usar gates/umbrales de corr.  ￼

⸻

Patches exactos

Los bloques están listos para copiar/pegar. Si prefieres recibir los archivos completos en otro momento, dime y te los empaqueto; por ahora te doy los diff-style y funciones concretas.

A) Mini-loader YAML (activar última selección)

Archivo nuevo: scripts/selected_loader.py
Phyton :# scripts/selected_loader.py
from __future__ import annotations
import os, json, yaml
from pathlib import Path

DEFAULTS = {
    "asset": "BTC-USD",
    "horizon": 90,
    "threshold": 0.66,
    "partial": "50_50",
    "fee_bps": 6,
    "slip_bps": 6,
}

def load_diamante_selected(path: str | Path = "configs/diamante_selected.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return DEFAULTS.copy()
    with p.open("r") as f:
        data = yaml.safe_load(f) or {}
    cfg = DEFAULTS | data
    # Exporta a ENV (opcional)
    os.environ["DIAMANTE_SELECTED_JSON"] = json.dumps(cfg, ensure_ascii=False)
    return cfg

Uso (al inicio de tu runner / CLI):
Phyton :
from selected_loader import load_diamante_selected
SELECTED = load_diamante_selected()  # dict con horizon, threshold, etc.

B) run_grid_tzsafe.py — argparse + Perla + correlación

B.1 Añade argumentos

(En tu bloque argparse existente)
Phyton :
ap.add_argument("--perla_csv", type=str, default=None,
                help="CSV con series de Perla (ts, exposure[,signal]) para medir correlación.")
ap.add_argument("--max_corr", type=float, default=None,
                help="Máxima correlación permitida vs Perla (abs). Si se excede, se descarta la config.")
ap.add_argument("--corr_method", type=str, choices=["pearson","spearman"], default="pearson",
                help="Método de correlación efectiva (default: pearson).")

B.2 Utilidades (poner cerca de otras helpers)
Phyton
import pandas as pd
import numpy as np

def _read_perla_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # columnas mínimas
    if "ts" not in df.columns:
        raise ValueError("perla_csv debe contener columna 'ts'")
    if "exposure" not in df.columns and "signal" not in df.columns:
        raise ValueError("perla_csv debe contener 'exposure' o 'signal'")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").set_index("ts")
    # normaliza 'E' = exposición en [-1,0,1] o [0,1]
    if "exposure" in df.columns:
        E = df["exposure"].clip(-1,1)
    else:
        # si sólo hay señal [0,1], úsala como proxy de exposición (0/1)
        E = (df["signal"] > 0.5).astype(float)
    return pd.DataFrame({"E_perla": E})

def build_diamante_exposure(sig_w: pd.Series,
                            threshold: float,
                            rearm_min: int,
                            hysteresis_pp: float) -> pd.Series:
    """
    Expone 1 cuando hay 'estado activo' tras cruce al alza y hasta 'rearm' barras de enfriamiento
    (fallback si no existen fills). Si tu pipeline ya produce fills/positions, reemplaza por ellos.
    """
    s = sig_w.astype(float).copy()
    above = s >= threshold
    # Histeresis simple: requiere que caiga por debajo (th - hys) para 'soltar'
    hys = max(0.0, hysteresis_pp)
    active = np.zeros(len(s), dtype=int)
    armed = False
    cooldown = 0
    for i, val in enumerate(s.values):
        if cooldown > 0:
            cooldown -= 1
            active[i] = 0
            continue
        if not armed and val >= threshold:
            armed = True
            active[i] = 1
        elif armed and val >= (threshold - hys):
            active[i] = 1
        else:
            # se desarma y entra a cooldown
            if armed:
                cooldown = max(0, int(rearm_min))
            armed = False
            active[i] = 0
    out = pd.Series(active, index=s.index, name="E_diamante").astype(float)
    return out

def effective_correlation(E_diamante: pd.Series,
                          perla_df: pd.DataFrame,
                          start: pd.Timestamp,
                          end: pd.Timestamp,
                          method: str = "pearson") -> tuple[float, int]:
    """
    Reindexa a 4h si hace falta, alinea en [start,end], y calcula corr(E_diamante, E_perla).
    Devuelve (corr, N_overlap). Si N_overlap<60 devuelve (np.nan, N_overlap).
    """
    # recorta ventanas y alinea
    d = E_diamante.loc[(E_diamante.index>=start)&(E_diamante.index<=end)]
    p = perla_df.loc[(perla_df.index>=start)&(perla_df.index<=end)]

    # muestreo a 4h si tus señales están a 1m/h; ajusta si procede
    def maybe_downsample(x):
        if pd.infer_freq(x.index) in (None, "T", "min", "1T", "H", "1H"):
            return x.resample("4H").last().ffill()
        return x

    d = maybe_downsample(d)
    p = maybe_downsample(p["E_perla"])

    df = pd.concat([d, p], axis=1).dropna()
    N = len(df)
    if N < 60:
        return (float("nan"), N)
    corr = df["E_diamante"].corr(df["E_perla"], method=method)
    return (float(corr), N)

B.3 Integración en el loop principal

(Justo donde ya calculas métricas por asset/horizon/threshold, añade antes de escribir la fila:)

Phyton :
# si se pasó --perla_csv, calcula corr_perla
corr_perla = np.nan
corr_gate = 1
if args.perla_csv:
    if not hasattr(args, "_perla_cache"):
        args._perla_cache = _read_perla_csv(args.perla_csv)
    E_d = build_diamante_exposure(sig_w, float(th), rearm_min, hysteresis_pp)
    corr_perla, N_overlap = effective_correlation(
        E_d, args._perla_cache, W.start, W.end, method=args.corr_method
    )
    if args.max_corr is not None and not np.isnan(corr_perla):
        if abs(corr_perla) > float(args.max_corr):
            corr_gate = 0  # descartar por exceso de correlación
# ... al construir el dict de salida:
row = {
    "window": W.label, "asset": asset, "horizon": int(hor), "threshold": float(th),
    "trades": int(crosses), "pf": pf, "wr": wr, "mdd": mdd, "sortino": sortino, "roi": roi,
    "corr_perla": corr_perla, "corr_gate": corr_gate,
    "_file": sig_file,
}
# si corr_gate==0 y hay gate -> (opcional) no escribir la fila, o escribirla con flag
if args.max_corr is not None and corr_gate == 0:
    # opción A: escribe con flag y la excluirás aguas abajo
    pass

Nota: Si prefieres excluir directamente, haz if args.max_corr and corr_gate==0: continue.

⸻

C) backtest_grid.py — argparse + Perla + columnas extra

C.1 Añade argumentos (idénticos)

Phyton :
ap.add_argument("--perla_csv", type=str, default=None)
ap.add_argument("--max_corr", type=float, default=None)
ap.add_argument("--corr_method", type=str, choices=["pearson","spearman"], default="pearson")

C.2 Reutiliza helpers

Copia las funciones _read_perla_csv / build_diamante_exposure / effective_correlation (o impórtalas desde run_grid_tzsafe.py si lo prefieres).
En el punto donde escribes las filas al CSV, añade corr_perla y corr_gate igual que arriba.

⸻

D) Escritura de corr_perla en CSVs
	•	kpis/val CSVs: nuevas columnas al final:
corr_perla, corr_gate
	•	Top CSVs: añade corr_perla en la vista (y, si aplicas gate duro, filtra por corr_gate==1 antes de rankear).

⸻

E) Mini-loader (auto-activación de selección)

E.1 Llamada de conveniencia

En tus scripts de “día a día” puedes incluir:

bash :
python - << 'PY'
from selected_loader import load_diamante_selected
sel = load_diamante_selected("configs/diamante_selected.yaml")
print("Selected:", sel)
PY

E.2 Uso en un comando real
bash :
SEL_JSON=$(python - << 'PY'
from selected_loader import load_diamante_selected
import json; print(json.dumps(load_diamante_selected()))
PY
)
H=$(python -c "import json,os; print(int(json.loads(os.environ['DIAMANTE_SELECTED_JSON'])['horizon']))")
TH=$(python -c "import json,os; print(float(json.loads(os.environ['DIAMANTE_SELECTED_JSON'])['threshold']))")

python scripts/run_grid_tzsafe.py \
  --windows "2024H1:2024-01-01:2024-06-30" \
  --assets BTC-USD --horizons "$H" --thresholds "$TH" \
  --signals_root reports/windows_fixed --ohlc_root data/ohlc/1m \
  --fee_bps 6 --slip_bps 6 --partial 50_50 --breakeven_after_tp1 \
  --risk_total_pct 0.75 --weights BTC-USD=1.0 \
  --perla_csv reports/perla_weekly_positions.csv --max_corr 0.65 \
  --out_csv reports/val_2024H1_selected.csv \
  --out_top reports/val_2024H1_selected_top.csv

F) Checklist operativo (semana a semana)

Lunes
	•	Actualiza FREEZE.
	•	Corre Diamante (BTC/ETH/SOL).
	•	Guarda diamante_*_weekX.csv.
	•	(Si tienes la serie de Perla semanal) actualiza reports/perla_weekly_positions.csv.

Martes
	•	Revisión gate multi-activo: PF>1.6, WR>60%, Trades≥30.

Miércoles
	•	Micro-grid por fold (misma ENV y ventanas), con --perla_csv y --max_corr activo.

Jueves
	•	Costes duros (2× slip + comisión IN/OUT).

Viernes
	•	Consolidación. Exporta winners_BothFolds.csv y reporte con:
pf, wr, trades, mdd, sortino, net, corr_perla, corr_gate.

⸻

G) Comprobaciones rápidas
	1.	Sanity sin gate de Perla:
bash :
python scripts/run_grid_tzsafe.py --windows "2023Q4:2023-10-01:2023-12-31" \
  --assets BTC-USD --horizons 90 120 --thresholds 0.64 0.66 0.68 \
  --signals_root reports/windows_fixed --ohlc_root data/ohlc/1m \
  --fee_bps 6 --slip_bps 6 --partial 50_50 --breakeven_after_tp1 \
  --risk_total_pct 0.75 --weights BTC-USD=1.0 \
  --out_csv reports/val_Q4_p5050.csv --out_top reports/val_Q4_top_p5050.csv

Con gate Perla (descarga configs pegadas)
Phyton:
python scripts/run_grid_tzsafe.py --windows "2024H1:2024-01-01:2024-06-30" \
  --assets BTC-USD --horizons 90 120 --thresholds 0.64 0.66 0.68 \
  --signals_root reports/windows_fixed --ohlc_root data/ohlc/1m \
  --fee_bps 6 --slip_bps 6 --partial 50_50 --breakeven_after_tp1 \
  --risk_total_pct 0.75 --weights BTC-USD=1.0 \
  --perla_csv reports/perla_weekly_positions.csv --max_corr 0.65 \
  --out_csv reports/val_2024H1_corrgate.csv --out_top reports/val_2024H1_corrgate_top.csv

Si corr_perla > max_corr, verás corr_gate=0 en CSV (puedes filtrar antes de rankear).

⸻

H) Observaciones finales
	•	Mantener forward-chaining (estilo TimeSeriesSplit) para evitar fugas temporales.  ￼
	•	Usa el gate de correlación para reforzar independencia entre Diamante (francotirador) y Perla (captura de ondas), mejorando la diversificación del portafolio global.  ￼
	•	Para documentación de control de sobre-optimización y selección robusta, ten a mano las notas de Bailey/López de Prado (DSR y “serial killers” de backtests).  ￼ ￼
