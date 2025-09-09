# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .indicators import ema


# =============================
# Costs
# =============================
@dataclass
class Costs:
    bps_per_side: float = 12.0  # fee+slip por lado (12 bps side ~ 24 bps RT)

    @property
    def frac_side(self) -> float:
        return self.bps_per_side / 1e4


def TradeCosts(fee_bps_per_side: float, slip_bps_per_side: float):
    """
    Back-compat para CLI antiguo: TradeCosts(fee_bps, slip_bps) -> Costs
    """
    total = float(fee_bps_per_side) + float(slip_bps_per_side)
    return Costs(bps_per_side=total)


# =============================
# Utilidades
# =============================
def _ensure_cols(df: pd.DataFrame, name: str = "df") -> pd.DataFrame:
    """Normaliza nombres y tipos mínimos requeridos para OHLCV."""
    if not hasattr(df, "columns"):
        raise TypeError(f"{name}: se esperaba DataFrame, pero llegó {type(df)}")

    out = df.copy()

    # normaliza nombre de timestamp a 'timestamp'
    if 'timestamp' not in out.columns:
        for c in ['ts', 'date', 'datetime', 'time', 'Date', 'Datetime', 'Time']:
            if c in out.columns:
                out = out.rename(columns={c: 'timestamp'})
                break
    if 'timestamp' not in out.columns:
        raise ValueError(f"{name}: falta columna 'timestamp'.")

    # asegura tipo datetime tz-aware UTC
    out['timestamp'] = pd.to_datetime(out['timestamp'], utc=True, errors='coerce')
    out = out.dropna(subset=['timestamp']).sort_values('timestamp')

    # check OHLCV
    for c in ['open', 'high', 'low', 'close']:
        if c not in out.columns:
            raise ValueError(f"{name}: falta columna '{c}' en DataFrame OHLC.")
        out[c] = pd.to_numeric(out[c], errors='coerce')

    if 'volume' in out.columns:
        out['volume'] = pd.to_numeric(out['volume'], errors='coerce')

    out = out.drop_duplicates(subset=['timestamp'], keep='last')
    return out


def _weekly_key(ts: pd.Timestamp) -> str:
    iso = ts.isocalendar()
    return f"{iso.year}-W{int(iso.week):02d}"


def _calc_mdd(series: pd.Series) -> float:
    """Máximo drawdown en fracción (0..1) de una curva en USD."""
    s = series.astype(float)
    roll_max = s.cummax()
    dd = (s - roll_max) / roll_max.replace(0, np.nan)
    dd = dd.fillna(0.0)
    return float(-dd.min()) if len(dd) else 0.0


def _maybe_atr_pause(df4h: pd.DataFrame, cfg: Dict, i: int) -> bool:
    """
    Pausa por ATR% (opcional). Si no está activado en YAML, devuelve False (no pausar).
      - ATR14%
      - Percentil p sobre todo el histórico.
      - Banda amarilla ±yellow_band_pct (puntos de percentil): pausa si cae en la banda.
    """
    mods = cfg.get('modules', {})
    atr_cfg = mods.get('atr_regime', {}) or {}
    if not atr_cfg.get('enabled', False):
        return False

    lookback = int(atr_cfg.get('lookback_bars', 14))
    p = float(atr_cfg.get('percentile_p', 40.0))
    yb = float(atr_cfg.get('yellow_band_pct', 3.0))

    c = df4h['close']
    h = df4h['high']
    l = df4h['low']
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=lookback, adjust=False).mean()
    atr_pct = (atr14 / c.replace(0, np.nan)) * 100.0

    thr_y_low = np.nanpercentile(atr_pct, max(0.0, p - yb))
    thr_y_high = np.nanpercentile(atr_pct, min(100.0, p + yb))

    val = float(atr_pct.iloc[i])
    if math.isnan(val):
        return False

    return (val >= thr_y_low) and (val <= thr_y_high)


# =============================
# Preparación de datos
# =============================
def _prepare_macro_d1(d1: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Construye macro diario con confirmación D-1 (sin look-ahead).
    Devuelve columnas: ['d_date','d_close','d_ema200','macro_ok_d1']
    donde d_date es int YYYYMMDD.
    """
    d1 = _ensure_cols(d1, name='d1').sort_values('timestamp').set_index('timestamp')

    # Señal de D-1 (shift para no ver futuro)
    sig = pd.DataFrame(index=d1.index)
    sig['d_close'] = d1['close']
    sig['d_ema200'] = ema(sig['d_close'], 200)
    sig = sig.shift(1)
    sig['macro_ok_d1'] = sig['d_close'] > sig['d_ema200']

    out = sig.reset_index()
    out['d_date'] = pd.to_datetime(out['timestamp'], utc=True).dt.strftime('%Y%m%d').astype('int64')
    return out[['d_date', 'd_close', 'd_ema200', 'macro_ok_d1']]


def _prepare_h4(h4: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Normaliza H4 y añade 'd_date' (int YYYYMMDD) para merge con D1, evitando duplicados."""
    df = _ensure_cols(h4, name='h4').sort_values('timestamp').reset_index(drop=True)

    # Si ya existía 'ts', elimínala para no duplicar nombre de columna
    if 'ts' in df.columns:
        df = df.drop(columns=['ts'])

    # Crea 'ts' a partir de 'timestamp' (siempre tz-aware UTC)
    df['ts'] = pd.to_datetime(df['timestamp'], utc=True)

    # (cinturón y tirantes) elimina columnas duplicadas si las hubiera
    df = df.loc[:, ~df.columns.duplicated()]

    # Clave de merge homogénea: int YYYYMMDD
    df['d_date'] = df['ts'].dt.strftime('%Y%m%d').astype('int64')
    return df


# =============================
# Backtest
# =============================
def run_backtest(d1: pd.DataFrame, h4: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Núcleo v0.1 con:
      - Macro D1 (D-1) sin look-ahead
      - Entrada: ema21>ema55 y macro_ok
      - Salidas: (a) activa con confirmación (close<ema21 y NO recuperación al cierre siguiente),
                 (b) pasiva ema21<ema55,
                 (c) (opcional) SELL forzado si macro se pone rojo (flag YAML).
      - Dwell entre flips
      - Presupuesto semanal de flips (opcional)
      - Costes en bps por lado
      - Evaluación al cierre de la vela 4h; ejecución en el open de la siguiente
    """
    # --------- datos 4h
    df = _prepare_h4(h4, cfg)
    df['ema21'] = ema(df['close'], 21)
    df['ema55'] = ema(df['close'], 55)
    df['next_open'] = df['open'].shift(-1)
    df['prev_close'] = df['close'].shift(1)

    # salida activa vectorizada (usa barra previa y actual para poder ejecutar en la próxima)
    exit_active_sig = (df['prev_close'] < df['ema21'].shift(1)) & (df['close'] <= df['ema21'])
    df['exit_active_sig'] = exit_active_sig

    # --------- macro D1 (D-1) y merge por día (UTC)
    d1_macro = _prepare_macro_d1(d1, cfg)
    df = df.merge(d1_macro, on='d_date', how='left')
    df['macro_ok'] = df['macro_ok_d1'].fillna(False)

    # --------- parámetros / defaults
    backtest = cfg.get('backtest', {}) or {}
    seed_usd = float(backtest.get('seed_usd', 69259.12))
    seed_btc = backtest.get('seed_btc', None)

    anti = cfg.get('anti_whipsaw', {}) or {}
    dwell_min = int(anti.get('dwell_bars_min_between_flips', 6))

    costs_cfg = cfg.get('costs', {}) or {}
    # permite fee_bps + slip_bps o bps_per_side directamente
    bps_side = float(costs_cfg.get('bps_per_side', costs_cfg.get('fee_bps', 6.0) + costs_cfg.get('slip_bps', 6.0)))
    costs = Costs(bps_per_side=bps_side)

    modules = cfg.get('modules', {}) or {}
    weekly_cfg = modules.get('weekly_turnover_budget', {}) or {}
    weekly_on = bool(weekly_cfg.get('enabled', False))
    flips_per_week_max = int(weekly_cfg.get('flips_per_week_max', 2))
    force_sell_on_macro_red = bool(modules.get('force_sell_on_macro_red', False))

    # --- Presupuesto dinámico por ATR: 2-verde / 1-resto (opcional)
    atr_cfg = modules.get('atr_regime', {}) or {}
    dyn_by_atr = bool(weekly_cfg.get('dynamic_by_atr', False))

    # Serie con el "cap" de BUY permitido en cada barra (1 por defecto)
    buy_cap_series = pd.Series(int(flips_per_week_max), index=df.index)

    if weekly_on and dyn_by_atr and atr_cfg.get('enabled', False):
        lookback = int(atr_cfg.get('lookback_bars', 14))
        p = float(atr_cfg.get('percentile_p', 40.0))
        yb = float(atr_cfg.get('yellow_band_pct', 5.0))  # por defecto 5p en preset prudente

        c = df['close']; h = df['high']; l = df['low']; pc = c.shift(1)
        tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        atr14 = tr.ewm(span=lookback, adjust=False).mean()
        atr_pct = (atr14 / c.replace(0, np.nan)) * 100.0

        thr_green = np.nanpercentile(atr_pct, min(100.0, p + yb))

        # verde => 2; resto => 1
        buy_cap_series = pd.Series(np.where(atr_pct > thr_green, 2, 1), index=df.index).fillna(1).astype(int)

    # --------- estado
    position = 'STABLE'
    usd = float(seed_usd)
    if seed_btc is not None:
        usd = float(seed_btc) * float(df['open'].iloc[0])
    btc = 0.0

    last_flip_i = -10_000
    flips_total = 0
    flips_week: Dict[str, int] = {}
    pending_signal = None  # 'BUY' o 'SELL' para ejecutar en la próxima barra

    rows = []

    # --------- estado
    position = 'STABLE'
    ...
    rows = []

    # Buffer anti-microcruces (constante por corrida)
    eps = float(cfg.get('signals', {}).get('cross_buffer_bps', 0.0)) / 1e4

    # --------- bucle principal (cierre actual -> ejecutar en open siguiente)
    for i in range(len(df) - 1): # hasta penúltima porque ejecutamos en i+1
        ts = df.loc[i, 'ts']
        op = df.loc[i, 'open']
        cl = df.loc[i, 'close']
        next_op = df.loc[i + 1, 'open']

        # 1) ejecutar si había signal pendiente
        executed = None
        if pending_signal == 'BUY' and position == 'STABLE':
            usd *= (1.0 - costs.frac_side)
            price = op if (next_op and not np.isnan(next_op)) else op
            btc = usd / price
            usd = 0.0
            position = 'BTC'
            executed = 'BUY'
            pending_signal = None
            last_flip_i = i
            flips_total += 1
            if weekly_on:
                wk = _weekly_key(ts)
                flips_week[wk] = flips_week.get(wk, 0) + 1

        elif pending_signal == 'SELL' and position == 'BTC':
            usd = btc * op
            usd *= (1.0 - costs.frac_side)
            btc = 0.0
            position = 'STABLE'
            executed = 'SELL'
            pending_signal = None
            last_flip_i = i
            flips_total += 1
            # Removed weekly count increment for SELL

        # 2) equity al cierre de la barra i (marcamos ejecución si hubo al inicio)
        eq_usd = usd if position == 'STABLE' else btc * cl
        eq_btc = (0.0 if cl == 0 else eq_usd / cl)

        rows.append({
            'ts': ts, 'open': op, 'close': cl,
            'equity_usd': float(eq_usd), 'equity_btc': float(eq_btc),
            'executed': executed
        })

        # 3) generar nueva señal (que se ejecutará en la PRÓXIMA barra)
        if i < (len(df) - 2):  # aún hay una barra futura para ejecutar
            can_flip = (i - last_flip_i) >= dwell_min
            wk = _weekly_key(ts)
            # Budget BUY dinámico (por barra): cap=1 o 2 según ATR; SELL siempre permitido
            buy_cap = int(buy_cap_series.iloc[i]) if weekly_on else 1_000_000
            buy_ok = (not weekly_on) or (flips_week.get(wk, 0) < buy_cap)
            sell_ok = True

            atr_pause = _maybe_atr_pause(df, cfg, i)

            eps = float(cfg.get('signals', {}).get('cross_buffer_bps', 0.0)) / 1e4
            # micro-buffer configurable para evitar microcruces
            # justo antes de usar trend_up:
            eps = float(cfg.get('signals', {}).get('cross_buffer_bps', 0.0)) / 1e4
            trend_up = bool(df.loc[i, 'ema21'] > df.loc[i, 'ema55'] * (1.0 + eps))
            macro_ok = bool(df.loc[i, 'macro_ok'])

            if position == 'STABLE':
                if macro_ok and trend_up and can_flip and buy_ok and (not atr_pause):
                    pending_signal = 'BUY'
            elif position == 'BTC':
                if (not macro_ok) and can_flip and force_sell_on_macro_red:
                    pending_signal = 'SELL'
                else:
                    exit_confirm = bool(df.loc[i, 'exit_active_sig'])  # cierra<ema21 y NO recupera la barra actual
                    exit_passive = bool(df.loc[i, 'ema21'] < df.loc[i, 'ema55'])
                    if can_flip and (exit_confirm or exit_passive) and sell_ok:
                        pending_signal = 'SELL'

    equity_df = pd.DataFrame(rows)

    # --- Diagnóstico: presupuesto BUY semanal (no afecta resultados) ---
    try:
        base_cap = int(cfg.get('modules', {}).get('weekly_turnover_budget', {}).get('flips_per_week_max', 1))

        # BUYs/semana desde equity
        iso_eq = equity_df['ts'].dt.isocalendar()
        eq_week = iso_eq.year.astype(str) + '-W' + iso_eq.week.astype(str).str.zfill(2)
        buys_by_week = (
            equity_df.assign(week=eq_week)
                     .loc[lambda x: x['executed'] == 'BUY']
                     .groupby('week')['executed']
                     .count()
        )

        # Cap semanal desde H4 (máximo cap por barra dentro de la semana)
        iso_h4 = df['ts'].dt.isocalendar()
        h4_week = iso_h4.year.astype(str) + '-W' + iso_h4.week.astype(str).str.zfill(2)
        cap_by_week = (
            pd.Series(buy_cap_series.values, index=h4_week)
              .groupby(level=0)
              .max()
              .astype(int)
        )

        # Alinear índices y rellenar caps faltantes con base_cap
        buys_al, cap_al = buys_by_week.align(cap_by_week, join='outer', fill_value=0)
        cap_al = cap_al.replace(0, np.nan).fillna(base_cap).astype(int)
        buys_al = buys_al.astype(int)

        viol = int((buys_al > cap_al).sum())

        print("BUY/semana (últimas 12):")
        print(buys_al.tail(12))
        print("\nCap por semana (últimas 12):")
        print(cap_al.tail(12))
        print("\nMax BUY/semana:", int(buys_al.max() if not buys_al.empty else 0))
        print("Semanas con violación de cap:", viol)

        if viol:
            detail = (
                pd.DataFrame({'buys': buys_al, 'cap': cap_al})
                .query('buys > cap')
                .sort_index()
            )
            print("\nViolaciones:\n", detail)

        missing_in_cap = set(buys_al.index) - set(cap_al.index)
        missing_in_buys = set(cap_al.index) - set(buys_al.index)
        if missing_in_cap or missing_in_buys:
            print("\nSemanas sin correspondencia -> en cap:", missing_in_cap, " | en buys:", missing_in_buys)
    except Exception as e:
        print("[WARN] Diagnóstico weekly_turnover_budget falló:", repr(e))

    # --------- KPIs
    first_price = float(df['open'].iloc[0])
    init_btc_hodl = (seed_usd / first_price)
    hodl_equity = init_btc_hodl * df['close']
    mdd_model = _calc_mdd(equity_df['equity_usd'])
    mdd_hodl = _calc_mdd(hodl_equity)

    final_usd_model = float(equity_df['equity_usd'].iloc[-1])
    final_usd_hodl = float(hodl_equity.iloc[-1])
    net_btc_ratio = (final_usd_model / final_usd_hodl) if final_usd_hodl else 0.0

    days = (equity_df['ts'].iloc[-1] - equity_df['ts'].iloc[0]).days
    years = max(1e-9, days / 365.25)
    flips_per_year = flips_total / years

    kpis = {
        'net_btc_ratio': net_btc_ratio,
        'mdd_model_usd': mdd_model,
        'mdd_hodl_usd': mdd_hodl,
        'mdd_vs_hodl_ratio': (mdd_model / mdd_hodl) if mdd_hodl else np.nan,
        'flips_total': float(flips_total),
        'flips_per_year': float(flips_per_year),
    }
    return equity_df, kpis


# =============================
# API pública (con compatibilidad)
# =============================
def simulate(a, b=None, c=None):
    """
    Compatibilidad:
      - Nuevo: simulate(d1_df, h4_df, cfg_dict)
      - Antiguo (CLI): simulate(cfg_dict, df, trade_costs)  -> carga d1/h4 desde cfg
    """
    # Nuevo estilo
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame) and isinstance(c, dict):
        d1, h4, cfg = a, b, c
        return run_backtest(d1, h4, cfg)

    # Estilo CLI antiguo: a=cfg (dict)
    if isinstance(a, dict):
        cfg = a

        # Legacy CLI style: a=cfg (dict), b=h4 DataFrame (already filtered by --start/--end), c=costs
        if isinstance(b, pd.DataFrame):
            h4 = b
        else:
            from .io import load_ohlc
            h4 = load_ohlc(
                cfg['data']['ohlc_4h_csv'],
                cfg['data'].get('ts_col', 'timestamp'),
                cfg['data'].get('tz_input', 'UTC'),
            )

        # Always load D1 from cfg, but trim to the H4 range (+buffer for EMA/shift)
        from .io import load_ohlc
        d1_full = load_ohlc(
            cfg['data']['ohlc_d1_csv'],
            cfg['data'].get('ts_col', 'timestamp'),
            cfg['data'].get('tz_input', 'UTC'),
        )

        h4_range = _ensure_cols(h4, name='h4_range')
        t0 = pd.to_datetime(h4_range['timestamp'].min(), utc=True).normalize() - pd.Timedelta(days=3)
        t1 = pd.to_datetime(h4_range['timestamp'].max(), utc=True).normalize()
        d1_full = _ensure_cols(d1_full, name='d1_full')
        d1 = d1_full[(d1_full['timestamp'] >= t0) & (d1_full['timestamp'] <= t1)]

        return run_backtest(d1, h4, cfg)

    raise TypeError("simulate: firma no soportada. Usa (d1_df, h4_df, cfg_dict) o (cfg_dict, ...) legacy.")