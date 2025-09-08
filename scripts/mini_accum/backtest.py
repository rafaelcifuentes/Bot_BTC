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