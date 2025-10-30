from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
from collections import deque
import numpy as np
import pandas as pd


def max_drawdown(series: pd.Series) -> float:
    """Máx. drawdown en valor positivo (p.ej. 0.25 = -25%)."""
    if series is None or len(series) == 0:
        return 0.0
    rollmax = series.cummax()
    dd = series / rollmax - 1.0
    return float(-dd.min()) if len(dd) else 0.0


@dataclass
class TradeCosts:
    fee_bps_per_side: float
    slip_bps_per_side: float

    def __post_init__(self) -> None:
        # Sanity checks: no negative bps
        if self.fee_bps_per_side < 0 or self.slip_bps_per_side < 0:
            raise ValueError("fee_bps_per_side and slip_bps_per_side must be >= 0")

    @property
    def rate(self) -> float:
        """Total cost per side as a fraction, i.e., (fee + slip) / 10_000."""
        return (self.fee_bps_per_side + self.slip_bps_per_side) / 10_000.0


def simulate(cfg: dict, df4: pd.DataFrame, costs: TradeCosts) -> Tuple[pd.DataFrame, dict]:
    """
    Backtest binario BTC↔USDC:
      - Señal: EMA_fast > EMA_slow (con banda cross_buffer_bps) y macro verde (D1 close > D1 EMA200)
      - Salidas: activa con confirmación (confirm_bars) y pasiva por cruce (ema_fast<ema_slow)
      - Anti-whipsaw:
          * dwell mínimo entre flips (dwell_bars_min_between_flips)
          * pausa tras flip (pause_after_flip_bars), opcionalmente también bloquea ventas (pause_affects_sell)
      - Presupuesto anual (flip_budget.hard_per_year) y throttle semanal soft de BUY (flip_budget.soft_per_week)
      - Régimen ATR con “yellow band” (modules.atr_regime.{enabled,percentile_p,yellow_band_pct,pause_affects_sell})
      - Ejecución: al OPEN de la siguiente vela
    Requiere en df4: ['ts','open','high','low','close','d_close','d_ema200'] (merge previo).
    """
    # Guard por DF vacío
    if df4 is None or df4.empty:
        res = pd.DataFrame(columns=[
            'ts', 'open', 'close', 'd_close', 'd_ema200', 'ema21', 'ema55',
            'macro_green', 'trend_up', 'position', 'btc', 'usd',
            'equity_btc', 'equity_usd', 'executed', 'exec_reason', 'blocked_reason'
        ])
        kpis = {
            'net_btc_ratio': None,
            'mdd_model_usd': 0.0,
            'mdd_hodl_usd': 0.0,
            'mdd_vs_hodl_ratio': None,
            'flips_total': 0,
            'flips_blocked_hard': 0,
            'flips_per_year': None,
        }
        return res, kpis

    df = df4.copy()

    # --- EMAs 4h + cross buffer en bps ---
    ema_fast = int(cfg['signals']['ema_fast'])
    ema_slow = int(cfg['signals']['ema_slow'])
    df['ema21'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['ema55'] = df['close'].ewm(span=ema_slow, adjust=False).mean()

    xbps = float(cfg.get('signals', {}).get('cross_buffer_bps', 0.0)) / 10_000.0
    up_thr = (1.0 + xbps)
    down_thr = (1.0 - xbps)
    df['trend_up'] = df['ema21'] > (df['ema55'] * up_thr)
    df['trend_dn'] = df['ema21'] < (df['ema55'] * down_thr)

    # --- Exit guardrail by ATR (optional, controlled by YAML: filters.exit_atr.{enabled,period,mult}) ---
    ex_atr_cfg = (cfg.get('filters', {}) or {}).get('exit_atr', {}) or {}
    use_guard = bool(ex_atr_cfg.get('enabled', False))
    period_atr = int(ex_atr_cfg.get('period', 14))
    mult_atr = float(ex_atr_cfg.get('mult', 1.5))

    # Default: guard passes (no extra restriction)
    df['exit_guard_ok'] = True

    if use_guard and all(c in df.columns for c in ['high', 'low', 'close']):
        prev_close = df['close'].shift(1)
        tr = (df['high'] - df['low']).abs()
        tr = np.maximum(tr, (df['high'] - prev_close).abs())
        tr = np.maximum(tr, (df['low'] - prev_close).abs())
        atr = tr.ewm(span=period_atr, adjust=False).mean()
        # Allow passive exit only if close < ema21 - k*ATR
        df['exit_guard_ok'] = df['close'] < (df['ema21'] - mult_atr * atr)

    # Macro (debe venir del merge diario)
    if 'macro_green' not in df.columns:
        df['macro_green'] = df['d_close'] > df['d_ema200']

    # --- Anti-whipsaw (pausa y dwell) ---
    aw = cfg.get('anti_whipsaw', {}) or {}
    dwell_min = int(aw.get('dwell_bars_min_between_flips', aw.get('dwell_bars_min', 0)))
    pause_after_flip_bars = int(aw.get('pause_after_flip_bars', 0))
    pause_affects_sell = bool(aw.get('pause_affects_sell', False))

    # --- Exit Active params (seguros si faltan en el YAML) ---
    exit_cfg = cfg.get('signals', {}).get('exit_active', {}) or {}
    active_exit_enabled = bool(exit_cfg.get('enabled', True))
    confirm_bars = int(exit_cfg.get('confirm_bars', 1))
    max_wait_after_confirm = int(exit_cfg.get('max_wait_bars_after_confirm', 2))
    age_valve_enabled = bool(exit_cfg.get('age_valve_enabled', False))  # por defecto OFF

    # --- Presupuesto / throttle ---
    fb = cfg.get('flip_budget', {}) or {}
    hard_per_year = int(fb.get('hard_per_year', 10**9))  # si no está: infinito
    enforce_hard = bool(fb.get('enforce_hard_yearly', True))
    soft_per_week = int(fb.get('soft_per_week', fb.get('soft_weekly', 10**9)))  # throttle semanal de BUY (rolling 7d)

    # --- ATR Regime + Yellow band (opcional) ---
    atr_cfg = (cfg.get('modules', {}) or {}).get('atr_regime', {}) or {}
    atr_enabled = bool(atr_cfg.get('enabled', False))
    atr_p = float(atr_cfg.get('percentile_p', 36))
    yellow_pct = float(atr_cfg.get('yellow_band_pct', 0.0))
    atr_blocks_sell = bool(atr_cfg.get('pause_affects_sell', False))

    if atr_enabled and all(c in df.columns for c in ['high', 'low', 'close']):
        prev_close = df['close'].shift(1)
        tr = (df['high'] - df['low']).abs()
        tr = np.maximum(tr, (df['high'] - prev_close).abs())
        tr = np.maximum(tr, (df['low'] - prev_close).abs())
        atr14 = tr.ewm(span=14, adjust=False).mean()
        df['atr_nrm'] = (atr14 / df['close']).clip(lower=0)
        thr = float(np.nanpercentile(df['atr_nrm'].values, atr_p))
        low_band = thr * (1.0 - yellow_pct)
        high_band = thr * (1.0 + yellow_pct)
        df['atr_quiet'] = df['atr_nrm'] <= low_band
        df['atr_loud'] = df['atr_nrm'] >= high_band
        df['atr_yellow'] = ~(df['atr_quiet'] | df['atr_loud'])
    else:
        df['atr_quiet'] = True
        df['atr_loud'] = False
        df['atr_yellow'] = False

    # --- XB adaptativo por ATR (opcional) ---
    # Si está habilitado, recomputa trend_up/trend_dn usando un buffer por-vela
    # determinado por el régimen ATR (quiet / yellow / loud).
    xb_cfg = (cfg.get('modules', {}) or {}).get('xb_adaptive', {}) or {}
    if bool(xb_cfg.get('enabled', False)):
        # Helper: convertir bps a fracción
        def _bps_to_frac(v: float) -> float:
            return float(v) / 10_000.0

        # Valores por régimen; si faltan, caen al cross_buffer global
        default_bps = float(cfg.get('signals', {}).get('cross_buffer_bps', 0.0))
        xb_quiet = _bps_to_frac(xb_cfg.get('quiet_bps', default_bps))
        xb_yellow = _bps_to_frac(xb_cfg.get('yellow_bps', default_bps))
        xb_loud = _bps_to_frac(xb_cfg.get('loud_bps', default_bps))

        # Si el régimen ATR no está activo, atr_quiet=True para todo (definido arriba)
        xbps_row = np.where(df['atr_quiet'], xb_quiet,
                            np.where(df['atr_loud'], xb_loud, xb_yellow))
        up_thr_row = 1.0 + xbps_row
        down_thr_row = 1.0 - xbps_row

        # Recalcular las máscaras de tendencia con umbrales por fila
        df['trend_up'] = df['ema21'] > (df['ema55'] * up_thr_row)
        df['trend_dn'] = df['ema21'] < (df['ema55'] * down_thr_row)

    # --- Capital inicial ---
    seed_btc = float(cfg['backtest'].get('seed_btc', 1.0))
    btc = seed_btc
    usd = 0.0
    position = 'BTC' if (bool(df['macro_green'].iloc[0]) and bool(df['trend_up'].iloc[0])) else 'STABLE'
    if position == 'STABLE':
        first_open = float(df['open'].iloc[0])
        usd = btc * first_open
        btc = 0.0

    # --- Estado de control ---
    bars_since_flip = dwell_min  # permitir flip inicial
    pause_until_i = -1  # índice (inclusive) hasta el que dura la pausa tras flip

    flips_exec_ts: List[pd.Timestamp] = []
    flips_blocked_hard = 0
    last_buys: deque[pd.Timestamp] = deque()  # para throttle semanal soft (ventana 7 días)

    # Órdenes programadas (+ razones persistentes para el flip)
    schedule_buy_i: Optional[int] = None
    schedule_sell_i: Optional[int] = None
    schedule_buy_reason: Optional[str] = None
    schedule_sell_reason: Optional[str] = None

    pending_exit_i: Optional[int] = None  # índice donde se detectó close<ema_fast para salida activa

    out_rows: List[dict] = []

    def budget_allows(ts: pd.Timestamp) -> bool:
        if not enforce_hard:
            return True
        one_year_ago = ts - pd.Timedelta(days=365)
        recent = sum(t > one_year_ago for t in flips_exec_ts)
        return recent < hard_per_year

    n = len(df)
    # Deja 2 barras de margen por la ejecución "siguiente open"
    for i in range(0, max(0, n - 2)):
        row = df.iloc[i]
        executed = None
        exec_reason_cur = None
        blocked_reason_cur = None

        # === Ejecutar órdenes programadas al OPEN de esta barra ===
        if schedule_buy_i is not None and i == schedule_buy_i:
            price = float(row['open'])
            if usd > 0.0:
                btc += (usd / price) * (1.0 - costs.rate)
                usd = 0.0
                position = 'BTC'
                flips_exec_ts.append(row['ts'])
                last_buys.append(row['ts'])
                bars_since_flip = 0
                executed = 'BUY'
                exec_reason_cur = schedule_buy_reason or 'BUY_trend'
                if pause_after_flip_bars > 0:
                    pause_until_i = max(pause_until_i, i + pause_after_flip_bars)
            schedule_buy_i = None
            schedule_buy_reason = None

        if schedule_sell_i is not None and i == schedule_sell_i:
            price = float(row['open'])
            if btc > 0.0:
                usd += btc * price * (1.0 - costs.rate)
                btc = 0.0
                position = 'STABLE'
                flips_exec_ts.append(row['ts'])
                bars_since_flip = 0
                executed = 'SELL'
                exec_reason_cur = schedule_sell_reason or 'SELL_cross_or_active'
                if pause_after_flip_bars > 0:
                    pause_until_i = max(pause_until_i, i + pause_after_flip_bars)
            schedule_sell_i = None
            schedule_sell_reason = None

        # === Señales al CIERRE de esta barra ===
        macro_green = bool(row['macro_green'])
        trend_up = bool(row['trend_up'])
        trend_dn = bool(row['trend_dn'])

        # Dwell + Pausa
        bars_since_flip += 1
        can_dwell = (bars_since_flip >= dwell_min)
        under_pause = (i < pause_until_i)
        can_buy = can_dwell and (not under_pause)  # la pausa siempre afecta compras
        can_sell = can_dwell and (not under_pause if pause_affects_sell else True)

        # Gating por ATR yellow band (si está activo)
        if atr_enabled and bool(df['atr_yellow'].iloc[i]):
            can_buy = False
            if atr_blocks_sell:
                can_sell = False

        # ======= Salida ACTIVA =======
        if active_exit_enabled and position == 'BTC' and (row['close'] < row['ema21']) and (pending_exit_i is None):
            pending_exit_i = i

        # Cancelar salida activa si hay recuperación
        if active_exit_enabled and pending_exit_i is not None and (row['close'] > row['ema21']):
            pending_exit_i = None

        # Confirmación + gating de salida activa
        if active_exit_enabled and pending_exit_i is not None and (i >= pending_exit_i + confirm_bars):
            age_since_confirm = i - (pending_exit_i + confirm_bars)
            regime_allows_active_exit = (trend_dn or (not macro_green) or (row['close'] < row['ema55']))
            allow_due_to_age = (age_valve_enabled and (age_since_confirm >= max_wait_after_confirm))

            if (row['close'] <= row['ema21']) and (regime_allows_active_exit or allow_due_to_age) \
                    and can_sell and (schedule_sell_i is None):
                if budget_allows(row['ts']):
                    schedule_sell_i = i + 1
                    schedule_sell_reason = 'SELL_active'
                    pending_exit_i = None
                else:
                    flips_blocked_hard += 1
                    # mantenemos pending_exit_i para reintentar más adelante

        # ======= Salida PASIVA por cruce EMAs =======
        if position == 'BTC' and trend_dn and can_sell and bool(df['exit_guard_ok'].iat[i]) and schedule_sell_i is None:
            if budget_allows(row['ts']):
                schedule_sell_i = i + 1
                if schedule_sell_reason is None:
                    schedule_sell_reason = 'SELL_cross'
            else:
                flips_blocked_hard += 1

        # ======= Entrada a BTC =======
        if position == 'STABLE' and macro_green and trend_up and can_buy and schedule_buy_i is None:
            # Throttle semanal soft para BUY (rolling 7 días; no cuenta SELL)
            if soft_per_week < 10**8:
                seven_days_ago = row['ts'] - pd.Timedelta(days=7)
                while last_buys and last_buys[0] <= seven_days_ago:
                    last_buys.popleft()
                if len(last_buys) >= soft_per_week:
                    blocked_reason_cur = 'soft_week_buy_limit'
            if blocked_reason_cur is None:
                if budget_allows(row['ts']):
                    schedule_buy_i = i + 1
                    if schedule_buy_reason is None:
                        schedule_buy_reason = 'BUY_trend'
                else:
                    flips_blocked_hard += 1
                    blocked_reason_cur = 'hard_year_budget'

        # ======= Equity (al cierre de la barra i) =======
        price_now = float(row['close'])
        equity_btc = btc + (usd / price_now if price_now > 0 else 0.0)
        equity_usd = btc * price_now + usd

        out_rows.append({
            'ts': row['ts'],
            'open': float(row['open']),
            'close': price_now,
            'd_close': float(row.get('d_close', np.nan)),
            'd_ema200': float(row.get('d_ema200', np.nan)),
            'ema21': float(row['ema21']),
            'ema55': float(row['ema55']),
            'macro_green': macro_green,
            'trend_up': trend_up,
            'position': position,
            'btc': float(btc),
            'usd': float(usd),
            'equity_btc': float(equity_btc),
            'equity_usd': float(equity_usd),
            'executed': executed,
            'exec_reason': exec_reason_cur,
            'blocked_reason': blocked_reason_cur,
        })

    res = pd.DataFrame(out_rows)

    # ------- KPIs -------
    if res.empty:
        kpis = {
            'net_btc_ratio': None,
            'mdd_model_usd': 0.0,
            'mdd_hodl_usd': 0.0,
            'mdd_vs_hodl_ratio': None,
            'flips_total': 0,
            'flips_blocked_hard': int(flips_blocked_hard),
            'flips_per_year': None,
        }
        return res, kpis

    hodl_usd = res['close'] * seed_btc
    mdd_model_usd = max_drawdown(res['equity_usd'])
    mdd_hodl_usd = max_drawdown(hodl_usd)
    mdd_ratio = (mdd_model_usd / mdd_hodl_usd) if mdd_hodl_usd > 0 else np.nan

    total_days = (pd.Timestamp(res['ts'].iloc[-1]) - pd.Timestamp(res['ts'].iloc[0])).days
    years = total_days / 365.25 if total_days > 0 else np.nan

    flips_total = int(len([x for x in res['executed'] if isinstance(x, str)]))
    flips_per_year = (flips_total / years) if years and years > 0 else np.nan
    net_btc_ratio = float(res['equity_btc'].iloc[-1] / seed_btc)

    kpis = {
        'net_btc_ratio': net_btc_ratio if net_btc_ratio == net_btc_ratio else None,
        'mdd_model_usd': float(mdd_model_usd),
        'mdd_hodl_usd': float(mdd_hodl_usd),
        'mdd_vs_hodl_ratio': float(mdd_ratio) if mdd_ratio == mdd_ratio else None,
        'flips_total': int(flips_total),
        'flips_blocked_hard': int(flips_blocked_hard),
        'flips_per_year': float(flips_per_year) if flips_per_year == flips_per_year else None,
    }
    return res, kpis