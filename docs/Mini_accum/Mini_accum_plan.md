docs/mini_accum_plan.md

Mini‑BOT BTC (Acumulación) — Hilo de Arranque "a rajatabla"

Contexto

Seguimos a rajatabla el núcleo esencial v0.1 y la regla de activación de módulos solo si pasan ablation + KPIs OOS. Tomamos las lecciones de Perla, Diamante, Corazón y Cerebro: simplicidad, disciplina y trazabilidad.

Misión

Acumular satoshis y superar HODL en BTC a 6–12 meses con menor MDD y bajo turnover. Sin shorts. Rotación binaria BTC ↔ Stable (USDC).

⸻

Núcleo esencial v0.1 (congelado)
	•	Estados: Acumulativo (BTC 100%), Defensivo (Stable 100%), Pausa disponible (no activa por defecto).
	•	Reloj: evaluación cada 4h (UTC) al cierre; ejecución en el open de la siguiente vela.
	•	Macro filtro: Precio D1 > EMA200_D1 → BTC; si no, Stable.
	•	Entrada (tendencial): EMA21_4h > EMA55_4h y macro verde (encima EMA200_D1).
	•	Salidas (primero que ocurra):
	•	Activa con confirmación: cierra < EMA21_4h y la siguiente vela no recupera > EMA21_4h (ejecuta al open posterior a la confirmación).
	•	Pasiva: EMA21_4h < EMA55_4h.
	•	Anti‑whipsaw: dwell = 4 velas (mínimo entre flips), ttl = 1 (barra de confirmación en salida activa).
	•	Costes: 6 bps fee + 6 bps slip por lado (≈ 24 bps RT).
	•	Disciplina: flip‑budget hard 26/año, soft 2/mes.

KPIs de aceptación (congelados)
	•	Win vs HODL (BTC): Net_BTC_ratio ≥ 1.05 (con costes).
	•	MDD (USD): MDD_model ≤ 0.85 × MDD_HODL (comparado en USD para que HODL no sea 0 en BTC).
	•	Turnover: flips/año ≤ 26 (soft 2/mes reportado).
	•	Robustez: estabilidad en 2022H2, 2023Q4, 2024H1.

Módulos opt‑in (solo si aportan en OOS)
	1.	Regímenes ATR% (p40) → Verde/Rojo; Amarillo = Pausa.
	2.	Hibernación por "chop" (≥2 cruces 21/55 en 40 velas 4h).
	3.	Grace TTL (cooldown suave): exigir señal "fuerte" para revertir en 1 vela post‑flip.
	4.	ATR‑adaptativo (2 niveles) y presupuesto de turnover semanal.

Se activan solo si mejoran MDD/turnover sin romper Net_BTC_ratio ni el presupuesto de flips.

Datos y costes (backtest en sombra)
	•	D1 (EMA200) y 4h (señales/ejecución), UTC, sin huecos/duplicados.
	•	Costes por lado: 6 bps fee + 6 bps slip.

Entregables del hilo
	•	docs/mini_accum_plan.md (este plan).
	•	configs/mini_accum.yaml (parámetros v0.1).
	•	scripts/mini_accum/backtest.py (runner en sombra sobre OHLC existentes).
	•	reports/mini_accum/* (equity vs HODL, KPIs, diffs y gráficos).
	•	CLI `mini-accum-dictamen` (consolidación de KPIs/aceptación).
	•	`reports/mini_accum/dictamen_*.{tsv,csv}` (export de dictamen).

Dictamen (consolidación de KPIs)
	•	Imprime en pantalla:
```
mini-accum-dictamen --reports-dir reports/mini_accum
```
	•	Incluir sufijos de depuración:
```
mini-accum-dictamen --reports-dir reports/mini_accum --include-dbg
```
	•	Exportar (TSV/CSV) y opcionalmente silenciar la salida:
```
mini-accum-dictamen --reports-dir reports/mini_accum \
  --out reports/mini_accum/dictamen_$(date +%Y%m%d_%H%M).tsv \
  --format tsv --quiet
```
	•	En CI: exit 0 si hay al menos un PASS, 1 si no:
```
mini-accum-dictamen --reports-dir reports/mini_accum \
  --only-pass --strict-exit --quiet \
  || echo "NO PASS"
```

Estado actual (última corrida)
	•	Resultado del dictamen: **NO PASS** (ningún run cumple los tres umbrales congelados).
	•	Motivos más frecuentes vistos en OOS recientes:
		- `net_btc_ratio` < 1.05
		- `flips_per_year` > 26.0
		- (en algunos setups) `mdd_vs_hodl_ratio` > 0.85
	•	Objetivo inmediato: reducir turnover y/o mejorar MDD **sin** degradar `net_btc_ratio` en 2022H2, 2023Q4 y 2024H1.

Siguientes pasos (ablation opt‑in)
	1)	Activar y evaluar **por separado (OOS)** los módulos opt‑in:
		- Hibernación por *chop* (≥2 cruces 21/55 en 40 velas) → Pausa.
		- *Grace TTL* (cooldown 1 vela) más estricto para reversión inmediata.
		- Regímenes ATR% con zona Amarilla (pausa).
		- Presupuesto semanal de flips (p.ej., ≤2).
	2)	Mantener **congelado** el núcleo v0.1; cualquier ajuste debe entrar como módulo opt‑in.
	3)	Documentar cada ablation con sufijos claros y consolidar con `mini-accum-dictamen`. 

Checklist de arranque
	•	Congelar núcleo v0.1 y KPIs.
	•	Definir datos (D1/4h UTC) y costes.
	•	Preparar YAML y one‑pager.
	•	Correr backtest en sombra (Base v0.1) y validar contra KPIs OOS.
	•	Evaluar módulos opt‑in en ablation (activar solo si ganan).

⸻

configs/mini_accum.yaml

version: 0.1
profile: base
numeraire: BTC
asset: BTC-USD
stable: USDC-USD
timezone: UTC

clock:
  timeframe_4h: "4h"
  decision_at_close: true
  execute_next_open: true

macro_filter:
  use: true
  type: ema200
  timeframe: "1d"
  lookback: 200
  condition: "d_close > d_ema200"

signals:
  ema_fast: 21
  ema_slow: 55
  entry: "ema21 > ema55 and macro_green"
  exit_active:
    rule: "close < ema21 and next_close <= next_ema21"  # confirma en la barra siguiente
    confirm_bars: 1
  exit_passive:
    rule: "ema21 < ema55"

anti_whipsaw:
  dwell_bars_min_between_flips: 4
  ttl_confirm_bars: 1

costs:
  fee_bps_per_side: 6
  slip_bps_per_side: 6

flip_budget:
  enforce_hard_yearly: true
  hard_per_year: 26
  soft_per_month: 2

modules_opt_in:
  regimes_atr:
    enabled: false
    atr_percentile: 40
    yellow_pause: true
  chop_hibernation:
    enabled: false
    window_bars: 40
    min_crosses: 2
  grace_ttl:
    enabled: false
    strong_gap_pct: 0.15  # ejemplo, gap relativo entre EMAs
  atr_adaptive:
    enabled: false
    levels: [1, 2]
    weekly_turnover_budget: 2

data:
  ohlc_4h_csv: "data/ohlc/4h/BTC-USD.csv"
  ohlc_d1_csv:  "data/ohlc/1d/BTC-USD.csv"
  ts_col: timestamp
  price_col: close
  tz_input: UTC
  drop_gaps: true

backtest:
  start: null   # ej. "2021-01-01"
  end:   null   # ej. "2025-09-01"
  oos_windows:
    - ["2022-07-01", "2022-12-31"]  # 2022H2
    - ["2023-10-01", "2023-12-31"]  # 2023Q4
    - ["2024-01-01", "2024-06-30"]  # 2024H1
  reports_dir: "reports/mini_accum"
  seed_btc: 1.0

kpis:
  accept:
    net_btc_ratio_min: 1.05
    mdd_vs_hodl_ratio_max: 0.85
    flips_per_year_max: 26
    flips_per_month_soft: 2
  robustness_windows: ["2022H2", "2023Q4", "2024H1"]


⸻

scripts/mini_accum/backtest.py

#!/usr/bin/env python3
"""
Mini‑BOT BTC (Acumulación) — Runner en sombra v0.1
- Rotación binaria BTC↔USDC
- Decisión: cierre 4h; ejecución: open de la siguiente vela
- Macro filtro: D1 close > EMA200_D1
- Señal tendencial: EMA21_4h > EMA55_4h
- Salida activa confirmada (ttl=1) y pasiva por cruce EMAs
- Anti‑whipsaw: dwell mínimo 4 velas entre flips
- Costes realistas: fee+slip por lado
- KPIs: Net_BTC_ratio, MDD (USD) vs HODL, flips/año

Este script no envía órdenes: solo backtest en sombra con OHLC existentes.
"""
from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd
import yaml

# --------------------------- utilidades ------------------------------------

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def load_ohlc(csv_path: str, ts_col: str, tz_input: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if ts_col not in df.columns:
        # tomar primera columna como timestamp si no existe el nombre esperado
        ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
    if df[ts_col].dt.tz is None:
        # si no contiene TZ, asumimos tz_input y convertimos a UTC
        if tz_input and tz_input.upper() != 'UTC':
            df[ts_col] = df[ts_col].dt.tz_localize(tz_input).dt.tz_convert('UTC')
        else:
            df[ts_col] = df[ts_col].dt.tz_localize('UTC')
    df = df.sort_values(ts_col).dropna(subset=[ts_col]).reset_index(drop=True)
    df = df.rename(columns={ts_col: 'ts'})
    required = {'open', 'high', 'low', 'close'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {csv_path}: {sorted(missing)}")
    return df


def merge_daily_into_4h(df4: pd.DataFrame, df1d: pd.DataFrame) -> pd.DataFrame:
    d = df1d[['ts', 'close']].copy()
    d['d_ema200'] = ema(d['close'], 200)
    d = d.rename(columns={'close': 'd_close'})
    # hacemos merge_asof (último daily cierre disponible <= ts_4h)
    merged = pd.merge_asof(
        df4.sort_values('ts'),
        d.sort_values('ts'),
        left_on='ts', right_on='ts',
        direction='backward'
    )
    merged['macro_green'] = merged['d_close'] > merged['d_ema200']
    return merged


def max_drawdown(series: pd.Series) -> float:
    rollmax = series.cummax()
    dd = series / rollmax - 1.0
    return float(dd.min()) * -1.0 if len(series) else 0.0


@dataclass
class TradeCosts:
    fee_bps_per_side: float
    slip_bps_per_side: float

    @property
    def rate(self) -> float:
        return (self.fee_bps_per_side + self.slip_bps_per_side) / 10_000.0


# --------------------------- backtest core ---------------------------------

def simulate(cfg: dict, df4: pd.DataFrame, costs: TradeCosts) -> Tuple[pd.DataFrame, dict]:
    """Simula la rotación BTC↔USDC con reglas v0.1. Devuelve timeseries y KPIs."""
    df = df4.copy()
    # EMAs 4h
    df['ema21'] = ema(df['close'], int(cfg['signals']['ema_fast']))
    df['ema55'] = ema(df['close'], int(cfg['signals']['ema_slow']))
    df['trend_up'] = df['ema21'] > df['ema55']
    df['trend_dn'] = df['ema21'] < df['ema55']

    # estado de la cartera
    btc = float(cfg['backtest'].get('seed_btc', 1.0))
    usd = 0.0
    position = 'STABLE'  # empezamos en stable por disciplina

    dwell_min = int(cfg['anti_whipsaw']['dwell_bars_min_between_flips'])
    bars_since_flip = dwell_min  # para permitir flip inicial cuando toque

    hard_per_year = int(cfg['flip_budget']['hard_per_year'])
    enforce_hard = bool(cfg['flip_budget'].get('enforce_hard_yearly', True))

    flips_exec_ts: List[pd.Timestamp] = []
    flips_blocked = 0

    # colas de órdenes
    schedule_buy_i = None   # índice donde ejecutaremos compra BTC en open
    schedule_sell_i = None  # índice donde ejecutaremos venta a USDC en open

    # lógica de salida activa confirmada
    pending_exit_i = None   # índice donde vimos close < ema21

    # series de resultado
    out = []

    for i in range(len(df) - 2):  # dejamos margen para ejecuciones diferidas
        row = df.iloc[i]
        nxt = df.iloc[i + 1]
        nxt2 = df.iloc[i + 2]

        # ejecutar órdenes programadas al open correspondiente
        executed = None
        if schedule_buy_i is not None and i == schedule_buy_i:
            # comprar BTC con USD en el open de esta barra
            price = row['open']
            if usd > 0:
                btc_delta = (usd / price) * (1.0 - costs.rate)
                btc += btc_delta
                usd = 0.0
                position = 'BTC'
                flips_exec_ts.append(row['ts'])
                bars_since_flip = 0
                executed = 'BUY'
            schedule_buy_i = None

        if schedule_sell_i is not None and i == schedule_sell_i:
            # vender BTC a USD en el open de esta barra
            price = row['open']
            if btc > 0:
                usd_delta = btc * price * (1.0 - costs.rate)
                usd += usd_delta
                btc = 0.0
                position = 'STABLE'
                flips_exec_ts.append(row['ts'])
                bars_since_flip = 0
                executed = 'SELL'
            schedule_sell_i = None

        # señales (evaluadas al CIERRE de la barra i)
        macro_green = bool(row['macro_green'])
        trend_up = bool(row['trend_up'])
        trend_dn = bool(row['trend_dn'])

        # control de dwell (mínimo de barras entre flips)
        bars_since_flip += 1
        can_flip = bars_since_flip >= dwell_min

        # activa salida con confirmación: si en BTC y close < ema21 → marcar pendiente
        if position == 'BTC' and row['close'] < row['ema21'] and pending_exit_i is None:
            pending_exit_i = i  # detectada

        # confirmar salida un bar después: si la barra siguiente NO recupera > ema21
        if pending_exit_i is not None and i == pending_exit_i + 1:
            if row['close'] <= row['ema21'] and can_flip:
                # programar venta al open de la próxima barra
                # respetando presupuesto hard anual
                if enforce_hard:
                    one_year_ago = row['ts'] - pd.Timedelta(days=365)
                    flips_last_year = sum(ts > one_year_ago for ts in flips_exec_ts)
                    if flips_last_year >= hard_per_year:
                        flips_blocked += 1
                    else:
                        schedule_sell_i = i + 1
                else:
                    schedule_sell_i = i + 1
            # limpiar el pendiente (confirmado o cancelado por recuperación)
            pending_exit_i = None

        # salida pasiva por cruce EMAs (si en BTC)
        if position == 'BTC' and trend_dn and can_flip and schedule_sell_i is None:
            # programar venta al open de la próxima barra
            if enforce_hard:
                one_year_ago = row['ts'] - pd.Timedelta(days=365)
                flips_last_year = sum(ts > one_year_ago for ts in flips_exec_ts)
                if flips_last_year >= hard_per_year:
                    flips_blocked += 1
                else:
                    schedule_sell_i = i + 1
            else:
                schedule_sell_i = i + 1

        # entrada BTC (si en STABLE) con macro + tendencia
        if position == 'STABLE' and macro_green and trend_up and can_flip and schedule_buy_i is None:
            if enforce_hard:
                one_year_ago = row['ts'] - pd.Timedelta(days=365)
                flips_last_year = sum(ts > one_year_ago for ts in flips_exec_ts)
                if flips_last_year >= hard_per_year:
                    flips_blocked += 1
                else:
                    schedule_buy_i = i + 1
            else:
                schedule_buy_i = i + 1

        # equity y tracking
        price_now = row['close']
        equity_btc = btc + (usd / price_now if price_now > 0 else 0.0)
        equity_usd = btc * price_now + usd
        out.append({
            'ts': row['ts'],
            'close': price_now,
            'd_close': row['d_close'],
            'd_ema200': row['d_ema200'],
            'ema21': row['ema21'],
            'ema55': row['ema55'],
            'macro_green': macro_green,
            'trend_up': trend_up,
            'position': position,
            'btc': btc,
            'usd': usd,
            'equity_btc': equity_btc,
            'equity_usd': equity_usd,
            'executed': executed,
        })

    res = pd.DataFrame(out)

    # KPIs
    # HODL (USD): 1 BTC buy&hold
    hodl_usd = res['close'] * 1.0
    mdd_model_usd = max_drawdown(res['equity_usd'])
    mdd_hodl_usd = max_drawdown(hodl_usd)
    mdd_ratio = (mdd_model_usd / mdd_hodl_usd) if mdd_hodl_usd > 0 else np.nan

    total_days = (res['ts'].iloc[-1] - res['ts'].iloc[0]).days if len(res) else 0
    years = total_days / 365.25 if total_days > 0 else np.nan
    flips_total = len(flips_exec_ts)
    flips_per_year = flips_total / years if years and years > 0 else np.nan

    net_btc_ratio = res['equity_btc'].iloc[-1] / 1.0 if len(res) else np.nan

    kpis = {
        'net_btc_ratio': float(net_btc_ratio),
        'mdd_model_usd': float(mdd_model_usd),
        'mdd_hodl_usd': float(mdd_hodl_usd),
        'mdd_vs_hodl_ratio': float(mdd_ratio) if not np.isnan(mdd_ratio) else None,
        'flips_total': int(flips_total),
        'flips_blocked_hard': int(flips_blocked),
        'flips_per_year': float(flips_per_year) if flips_per_year == flips_per_year else None,
    }

    return res, kpis


# --------------------------- ejecución CLI ---------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/mini_accum.yaml')
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    rep_dir = cfg['backtest']['reports_dir']
    os.makedirs(rep_dir, exist_ok=True)

    # cargar datos
    df4 = load_ohlc(cfg['data']['ohlc_4h_csv'], cfg['data']['ts_col'], cfg['data']['tz_input'])
    d1 = load_ohlc(cfg['data']['ohlc_d1_csv'], cfg['data']['ts_col'], cfg['data']['tz_input'])

    # merge diario→4h y aplicar filtros temporales
    df = merge_daily_into_4h(df4, d1)

    if args.start:
        start = pd.Timestamp(args.start, tz='UTC')
        df = df[df['ts'] >= start]
    if args.end:
        end = pd.Timestamp(args.end, tz='UTC')
        df = df[df['ts'] <= end]

    # costes
    costs = TradeCosts(
        fee_bps_per_side=float(cfg['costs']['fee_bps_per_side']),
        slip_bps_per_side=float(cfg['costs']['slip_bps_per_side'])
    )

    # simular
    res, kpis = simulate(cfg, df, costs)

    # guardar
    run_id = pd.Timestamp.utcnow().strftime('base_v0_1_%Y%m%d_%H%M')
    eq_path = os.path.join(rep_dir, f"{run_id}_equity.csv")
    kpi_path = os.path.join(rep_dir, f"{run_id}_kpis.csv")
    md_path = os.path.join(rep_dir, f"{run_id}_summary.md")

    res.to_csv(eq_path, index=False)
    pd.DataFrame([kpis]).to_csv(kpi_path, index=False)

    # resumen MD + veredicto contra umbrales
    acc = cfg['kpis']['accept']
    ok_btc = (kpis['net_btc_ratio'] is not None) and (kpis['net_btc_ratio'] >= float(acc['net_btc_ratio_min']))
    ok_mdd = (kpis['mdd_vs_hodl_ratio'] is not None) and (kpis['mdd_vs_hodl_ratio'] <= float(acc['mdd_vs_hodl_ratio_max']))
    ok_flip = (kpis['flips_per_year'] is not None) and (kpis['flips_per_year'] <= float(acc['flips_per_year_max']))

    verdict = 'ACEPTAR' if (ok_btc and ok_mdd and ok_flip) else 'RECHAZAR'

    with open(md_path, 'w') as f:
        f.write(f"# Mini‑BOT BTC v0.1 — Resumen {run_id}\n\n")
        f.write("## KPIs\n")
        for k, v in kpis.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Umbrales\n")
        for k, v in acc.items():
            f.write(f"- {k}: {v}\n")
        f.write(f"\n## Veredicto: **{verdict}**\n")
        f.write("\nNotas:\n- MDD comparado en USD vs HODL USD.\n- `net_btc_ratio` mide acumulación de BTC final vs 1 BTC inicial.\n- Presupuesto *hard* anual aplicado en simulación; excedentes se contabilizan como bloqueados.\n")

    print("[OK] Equity →", eq_path)
    print("[OK] KPIs  →", kpi_path)
    print("[OK] MD    →", md_path)


if __name__ == '__main__':
    main()


⸻

Cómo usar (rápido)

# 1) Guarda el YAML en configs/mini_accum.yaml y el script en scripts/mini_accum/backtest.py (chmod +x)
# 2) Verifica rutas de OHLC (D1 y 4h) en el YAML
# 3) Ejecuta el runner en sombra (todo en UTC)
python scripts/mini_accum/backtest.py --config configs/mini_accum.yaml \
  --start 2021-01-01 --end 2025-09-01

# Salidas:
# reports/mini_accum/<RUN_ID>_equity.csv
# reports/mini_accum/<RUN_ID>_kpis.csv
# reports/mini_accum/<RUN_ID>_summary.md (veredicto ACEPTAR/RECHAZAR)

Mantra: simple, probado y auditable. Satoshi a satoshi, con disciplina de flips y respeto estricto del plan.

⸻

Monorepo (PyCharm) — estructura "paquete aislado"

Objetivo: mantener Mini‑BOT (mini_accum) como paquete extraíble y reusable dentro de un monorepo simple. Más adelante, si lo deseas, podrás añadir diamante/, perla_negra/, corazon/, cerebro/ como paquetes hermanos sin romper nada.

Árbol propuesto

Bot_BTC/                         # raíz del monorepo (proyecto PyCharm)
├─ README.md
├─ .gitignore
├─ .env.example
├─ configs/
│  └─ mini_accum.yaml            # parámetros v0.1 (ya creado)
├─ docs/
│  └─ mini_accum_plan.md         # plan (ya creado)
├─ data/
│  └─ ohlc/
│     ├─ 1d/BTC-USD.csv
│     └─ 4h/BTC-USD.csv
├─ reports/
│  └─ mini_accum/                # salidas del backtest
├─ scripts/
│  └─ mini_accum/
│     └─ run_backtest.sh         # envoltorio CLI (opcional)
├─ packages/                     # **cada estrategia como paquete aislado**
│  └─ mini_accum/
│     ├─ pyproject.toml          # metadatos del paquete y entrypoint CLI
│     ├─ README.md
│     ├─ src/
│     │  └─ mini_accum/
│     │     ├─ __init__.py
│     │     ├─ indicators.py     # EMA u otros indicadores simples
│     │     ├─ io.py             # carga/merge de OHLC y daily EMA200
│     │     ├─ sim.py            # simulador v0.1 (core)
│     │     └─ cli.py            # CLI: `mini-accum-backtest`
│     └─ tests/
│        └─ test_smoke.py        # prueba rápida de integridad
└─ .github/workflows/
   └─ ci.yml                     # (opcional) lint + tests

Script de bootstrap (crea carpetas y archivos base)

Copia/pega en tu terminal dentro de la carpeta que será el monorepo (p.ej. ~/PycharmProjects/Bot_BTC) y ejecútalo.

#!/usr/bin/env bash
set -euo pipefail

# Raíz
mkdir -p Bot_BTC && cd Bot_BTC

# Carpeta raíz
printf "# Bot_BTC Monorepo\n\nMini‑BOT BTC (mini_accum) en paquete aislado.\n" > README.md
printf "# Entorno ejemplo\nPYTHONPATH=\n" > .env.example

# Ignora artefactos
cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
reports/
.env
EOF

# Estructura común
mkdir -p configs docs data/ohlc/1d data/ohlc/4h reports/mini_accum scripts/mini_accum packages/mini_accum/src/mini_accum packages/mini_accum/tests .github/workflows

# Placeholders si no existen
: > data/ohlc/1d/BTC-USD.csv
: > data/ohlc/4h/BTC-USD.csv

# Wrapper opcional de ejecución
cat > scripts/mini_accum/run_backtest.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
# Ejecuta el CLI empaquetado leyendo configs/mini_accum.yaml
mini-accum-backtest --config "configs/mini_accum.yaml" "$@"
EOF
chmod +x scripts/mini_accum/run_backtest.sh

# PyProject del paquete aislado
cat > packages/mini_accum/pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mini-accum"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "pandas>=2.0",
  "numpy>=1.24",
  "pyyaml>=6.0"
]
readme = "README.md"
license = {text = "MIT"}
authors = [{name = "Rafael Cifuentes"}]

[project.scripts]
mini-accum-backtest = "mini_accum.cli:main"

[tool.pytest.ini_options]
minversion = "8.0"
addopts = "-q"

[tool.ruff]
line-length = 100
EOF

# README del paquete
cat > packages/mini_accum/README.md <<'EOF'
# mini-accum (paquete)
CLI: `mini-accum-backtest --config configs/mini_accum.yaml --start 2021-01-01 --end 2025-09-01`
EOF

# Código fuente básico (CLI + módulos vacíos)
cat > packages/mini_accum/src/mini_accum/__init__.py <<'EOF'
__all__ = ["main"]
EOF

cat > packages/mini_accum/src/mini_accum/indicators.py <<'EOF'
import pandas as pd

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()
EOF

cat > packages/mini_accum/src/mini_accum/io.py <<'EOF'
from __future__ import annotations
import pandas as pd
from .indicators import ema

def load_ohlc(csv_path: str, ts_col: str, tz_input: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if ts_col not in df.columns:
        ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
    if df[ts_col].dt.tz is None:
        if tz_input and tz_input.upper() != 'UTC':
            df[ts_col] = df[ts_col].dt.tz_localize(tz_input).dt.tz_convert('UTC')
        else:
            df[ts_col] = df[ts_col].dt.tz_localize('UTC')
    df = df.sort_values(ts_col).dropna(subset=[ts_col]).reset_index(drop=True)
    df = df.rename(columns={ts_col: 'ts'})
    required = {'open', 'high', 'low', 'close'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {csv_path}: {sorted(missing)}")
    return df


def merge_daily_into_4h(df4: pd.DataFrame, df1d: pd.DataFrame) -> pd.DataFrame:
    d = df1d[['ts', 'close']].copy()
    d['d_ema200'] = ema(d['close'], 200)
    d = d.rename(columns={'close': 'd_close'})
    merged = pd.merge_asof(
        df4.sort_values('ts'),
        d.sort_values('ts'),
        left_on='ts', right_on='ts',
        direction='backward'
    )
    merged['macro_green'] = merged['d_close'] > merged['d_ema200']
    return merged
EOF

cat > packages/mini_accum/src/mini_accum/sim.py <<'EOF'
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd

def max_drawdown(series: pd.Series) -> float:
    rollmax = series.cummax()
    dd = series / rollmax - 1.0
    return float(dd.min()) * -1.0 if len(series) else 0.0

@dataclass
class TradeCosts:
    fee_bps_per_side: float
    slip_bps_per_side: float
    @property
    def rate(self) -> float:
        return (self.fee_bps_per_side + self.slip_bps_per_side) / 10_000.0


def simulate(cfg: dict, df4: pd.DataFrame, costs: TradeCosts) -> Tuple[pd.DataFrame, dict]:
    ema21 = df4['close'].ewm(span=int(cfg['signals']['ema_fast']), adjust=False).mean()
    ema55 = df4['close'].ewm(span=int(cfg['signals']['ema_slow']), adjust=False).mean()
    df = df4.copy()
    df['ema21'] = ema21
    df['ema55'] = ema55
    df['trend_up'] = df['ema21'] > df['ema55']
    df['trend_dn'] = df['ema21'] < df['ema55']

    btc = float(cfg['backtest'].get('seed_btc', 1.0)); usd = 0.0
    position = 'STABLE'
    dwell_min = int(cfg['anti_whipsaw']['dwell_bars_min_between_flips'])
    bars_since_flip = dwell_min
    hard_per_year = int(cfg['flip_budget']['hard_per_year'])
    enforce_hard = bool(cfg['flip_budget'].get('enforce_hard_yearly', True))
    flips_exec_ts: List[pd.Timestamp] = []
    flips_blocked = 0
    schedule_buy_i = None; schedule_sell_i = None; pending_exit_i = None
    out = []

    for i in range(len(df) - 2):
        row = df.iloc[i]
        executed = None
        # ejecutar órdenes
        if schedule_buy_i is not None and i == schedule_buy_i:
            price = row['open']
            if usd > 0:
                btc += (usd / price) * (1.0 - costs.rate); usd = 0.0
                position = 'BTC'; flips_exec_ts.append(row['ts']); bars_since_flip = 0; executed = 'BUY'
            schedule_buy_i = None
        if schedule_sell_i is not None and i == schedule_sell_i:
            price = row['open']
            if btc > 0:
                usd += btc * price * (1.0 - costs.rate); btc = 0.0
                position = 'STABLE'; flips_exec_ts.append(row['ts']); bars_since_flip = 0; executed = 'SELL'
            schedule_sell_i = None

        macro_green = bool(row['macro_green']); trend_up = bool(row['trend_up']); trend_dn = bool(row['trend_dn'])
        bars_since_flip += 1; can_flip = bars_since_flip >= dwell_min

        if position == 'BTC' and row['close'] < row['ema21'] and pending_exit_i is None:
            pending_exit_i = i
        if pending_exit_i is not None and i == pending_exit_i + 1:
            if row['close'] <= row['ema21'] and can_flip:
                if enforce_hard:
                    one_year_ago = row['ts'] - pd.Timedelta(days=365)
                    if sum(ts > one_year_ago for ts in flips_exec_ts) >= hard_per_year:
                        flips_blocked += 1
                    else:
                        schedule_sell_i = i + 1
                else:
                    schedule_sell_i = i + 1
            pending_exit_i = None

        if position == 'BTC' and trend_dn and can_flip and schedule_sell_i is None:
            if enforce_hard:
                one_year_ago = row['ts'] - pd.Timedelta(days=365)
                if sum(ts > one_year_ago for ts in flips_exec_ts) >= hard_per_year:
                    flips_blocked += 1
                else:
                    schedule_sell_i = i + 1
            else:
                schedule_sell_i = i + 1

        if position == 'STABLE' and macro_green and trend_up and can_flip and schedule_buy_i is None:
            if enforce_hard:
                one_year_ago = row['ts'] - pd.Timedelta(days=365)
                if sum(ts > one_year_ago for ts in flips_exec_ts) >= hard_per_year:
                    flips_blocked += 1
                else:
                    schedule_buy_i = i + 1
            else:
                schedule_buy_i = i + 1

        price_now = row['close']
        equity_btc = btc + (usd / price_now if price_now > 0 else 0.0)
        equity_usd = btc * price_now + usd
        out.append({'ts': row['ts'], 'close': price_now, 'equity_btc': equity_btc,
                    'equity_usd': equity_usd, 'executed': executed})

    res = pd.DataFrame(out)
    hodl_usd = res['close'] * 1.0
    mdd_model_usd = max_drawdown(res['equity_usd'])
    mdd_hodl_usd = max_drawdown(hodl_usd)
    mdd_ratio = (mdd_model_usd / mdd_hodl_usd) if mdd_hodl_usd > 0 else np.nan
    total_days = (res['ts'].iloc[-1] - res['ts'].iloc[0]).days if len(res) else 0
    years = total_days / 365.25 if total_days > 0 else np.nan
    flips_total = len([x for x in res['executed'] if isinstance(x, str)])
    flips_per_year = flips_total / years if years and years > 0 else np.nan
    net_btc_ratio = res['equity_btc'].iloc[-1] / 1.0 if len(res) else np.nan

    kpis = {
        'net_btc_ratio': float(net_btc_ratio) if net_btc_ratio == net_btc_ratio else None,
        'mdd_model_usd': float(mdd_model_usd),
        'mdd_hodl_usd': float(mdd_hodl_usd),
        'mdd_vs_hodl_ratio': float(mdd_ratio) if mdd_ratio == mdd_ratio else None,
        'flips_total': int(flips_total),
        'flips_per_year': float(flips_per_year) if flips_per_year == flips_per_year else None,
    }
    return res, kpis
EOF

cat > packages/mini_accum/src/mini_accum/cli.py <<'EOF'
from __future__ import annotations
import argparse, os
import pandas as pd, yaml
from .io import load_ohlc, merge_daily_into_4h
from .sim import simulate, TradeCosts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/mini_accum.yaml')
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    rep_dir = cfg['backtest']['reports_dir']; os.makedirs(rep_dir, exist_ok=True)
    df4 = load_ohlc(cfg['data']['ohlc_4h_csv'], cfg['data']['ts_col'], cfg['data']['tz_input'])
    d1  = load_ohlc(cfg['data']['ohlc_d1_csv'],  cfg['data']['ts_col'], cfg['data']['tz_input'])
    df = merge_daily_into_4h(df4, d1)
    if args.start: df = df[df['ts'] >= pd.Timestamp(args.start, tz='UTC')]
    if args.end:   df = df[df['ts'] <= pd.Timestamp(args.end, tz='UTC')]

    costs = TradeCosts(float(cfg['costs']['fee_bps_per_side']), float(cfg['costs']['slip_bps_per_side']))
    res, kpis = simulate(cfg, df, costs)

    run_id = pd.Timestamp.utcnow().strftime('base_v0_1_%Y%m%d_%H%M')
    eq_path = os.path.join(rep_dir, f"{run_id}_equity.csv")
    kpi_path = os.path.join(rep_dir, f"{run_id}_kpis.csv")
    md_path  = os.path.join(rep_dir, f"{run_id}_summary.md")
    res.to_csv(eq_path, index=False)
    pd.DataFrame([kpis]).to_csv(kpi_path, index=False)

    acc = cfg['kpis']['accept']
    ok_btc = kpis.get('net_btc_ratio') is not None and kpis['net_btc_ratio'] >= float(acc['net_btc_ratio_min'])
    ok_mdd = kpis.get('mdd_vs_hodl_ratio') is not None and kpis['mdd_vs_hodl_ratio'] <= float(acc['mdd_vs_hodl_ratio_max'])
    ok_flip = kpis.get('flips_per_year') is not None and kpis['flips_per_year'] <= float(acc['flips_per_year_max'])
    verdict = 'ACEPTAR' if (ok_btc and ok_mdd and ok_flip) else 'RECHAZAR'

    with open(md_path, 'w') as f:
        f.write(f"# Mini‑BOT BTC v0.1 — Resumen {run_id}\n\n")
        for k, v in kpis.items(): f.write(f"- **{k}**: {v}\n")
        f.write(f"\n**Veredicto:** {verdict}\n")
    print("[OK]", eq_path); print("[OK]", kpi_path); print("[OK]", md_path)

if __name__ == '__main__':
    main()
EOF

# Test mínimo
cat > packages/mini_accum/tests/test_smoke.py <<'EOF'
from mini_accum.indicators import ema
import pandas as pd

def test_ema_runs():
    s = pd.Series([1,2,3,4,5])
    out = ema(s, span=3)
    assert len(out) == 5
EOF

# CI placeholder (opcional)
cat > .github/workflows/ci.yml <<'EOF'
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - name: Install package
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install -U pip
          pip install -e packages/mini_accum
          pip install pytest ruff
      - name: Lint
        run: |
          source .venv/bin/activate
          ruff check packages/mini_accum/src
      - name: Tests
        run: |
          source .venv/bin/activate
          pytest packages/mini_accum/tests -q
EOF

printf "\n[OK] Monorepo inicializado en ./Bot_BTC (paquete mini_accum listo).\n"

Cómo abrir en PyCharm
	1.	Open → selecciona la carpeta Bot_BTC/ (raíz del monorepo).
	2.	Marca packages/mini_accum/src como Sources Root (PyCharm suele detectarlo solo).
	3.	Crea e activa tu .venv (Python 3.11) y ejecuta:

pip install -e packages/mini_accum
# ahora tienes el CLI disponible en el entorno:
mini-accum-backtest --config configs/mini_accum.yaml --start 2021-01-01 --end 2025-09-01


	4.	(Opcional) configura una Run Configuration que llame al binario mini-accum-backtest o al script scripts/mini_accum/run_backtest.sh.

Con esta estructura, extraer mini_accum a un repo propio es trivial: copia packages/mini_accum/ y publica. El monorepo seguirá limpio y modular.### Baselines fijados (v0.1.1-mini_accum)
- **H1_FZ (H1 2024)**: `flip_budget.enforce_hard_yearly=true`, `hard_per_year=26`
- **Q4_E3 (Q4 2023)**: `pause_affects_sell=true`

Reproducción:
```bash
bash scripts/mini_accum/run_seasonal.sh 2024-01-01 2024-06-30 demoH1
bash scripts/mini_accum/run_seasonal.sh 2023-10-01 2023-12-31 demoQ4
```
