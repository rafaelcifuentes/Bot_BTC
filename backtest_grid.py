#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evalúa señales históricas por ventanas / horizontes / umbrales, calcula P&L en múltiplos de R,
PF, WR, ROI aproximado, y vuelca resultados a CSV.
Asume LONGs. Maneja parcial 50/50 con BE tras TP1 opcional.
"""

import argparse, os, glob, math
from datetime import timedelta
import pandas as pd
pd.options.mode.chained_assignment = None  # silencio de warnings

def load_ohlc(path):
    df = pd.read_csv(path)
    # columnas: ts,open,high,low,close
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df = df.sort_values('ts').set_index('ts')
    return df[['open','high','low','close']]

def first_hit_in_horizon(row, ohlc, H_minutes, conservative_stop_first=True):
    """Devuelve ('tp2'|'tp1'|'sl'|'none', r_mult) sin costes. LONG only."""
    ts = pd.to_datetime(row['ts'], utc=True)
    entry = float(row['entry']); stop = float(row['stop'])
    tp1 = float(row['tp1']); tp2 = float(row['tp2'])
    if entry <= stop or tp1 <= entry or tp2 <= tp1:
        return ('none', 0.0)  # señal inválida para LONG

    H = ts + pd.Timedelta(minutes=int(H_minutes))
    slab = ohlc.loc[(ohlc.index > ts) & (ohlc.index <= H)]
    if slab.empty:
        return ('none', 0.0)

    # Encuentra primer toque por timestamp
    hit_tp1 = slab[slab['high'] >= tp1].head(1)
    hit_tp2 = slab[slab['high'] >= tp2].head(1)
    hit_sl  = slab[slab['low']  <= stop].head(1)

    t_tp1 = hit_tp1.index[0] if not hit_tp1.empty else pd.NaT
    t_tp2 = hit_tp2.index[0] if not hit_tp2.empty else pd.NaT
    t_sl  = hit_sl.index[0]  if not hit_sl.empty  else pd.NaT

    # Desempate conservador: si ocurren en misma vela, prioriza SL
    def earlier(a, b):
        if pd.isna(a): return False
        if pd.isna(b): return True
        if a == b:     return False  # lo resolvemos explícito abajo
        return a < b

    if not pd.isna(t_sl) and not pd.isna(t_tp2) and t_sl == t_tp2 and conservative_stop_first:
        return ('sl', -1.0)
    if not pd.isna(t_sl) and not pd.isna(t_tp1) and t_sl == t_tp1 and conservative_stop_first:
        return ('sl', -1.0)

    if earlier(t_tp2, t_sl):
        return ('tp2', +2.0)
    if earlier(t_tp1, t_sl):
        return ('tp1', +1.0)
    if not pd.isna(t_sl):
        return ('sl', -1.0)
    return ('none', 0.0)

def apply_partial_logic(outcome, ohlc, row, H_minutes, partial="none", breakeven_after_tp1=True):
    """Transforma el resultado en R considerando parcial 50/50 y BE tras TP1."""
    if partial != "50_50" or outcome[0] in ('sl','tp2','none'):
        return outcome  # ya definitivo o sin parcial

    # Si tocó TP1 primero:
    ts = pd.to_datetime(row['ts'], utc=True)
    H = ts + pd.Timedelta(minutes=int(H_minutes))
    slab = ohlc.loc[(ohlc.index > ts) & (ohlc.index <= H)]

    # ¿Llega a TP2 después?
    tp2 = float(row['tp2'])
    hit_tp2_after = slab[slab['high'] >= tp2].head(1)
    tp2_reached = not hit_tp2_after.empty

    # Mitad 1: +1R * 0.5 = +0.5R ya asegurado
    r = 0.5
    # Mitad 2:
    if breakeven_after_tp1:
        # segunda mitad pasa a BE: +1R*0.5 si llega a TP2, o 0R si no
        r += (1.0 * 0.5) if tp2_reached else 0.0
    else:
        # sin BE: si luego hace SL, penaliza -1R*0.5; si llega a TP2, +2R*0.5; si nada, 0
        entry = float(row['entry']); stop = float(row['stop'])
        sl_hit = slab[slab['low'] <= stop].head(1)
        if tp2_reached and (sl_hit.empty or hit_tp2_after.index[0] < sl_hit.index[0]):
            r += 2.0 * 0.5
        elif not sl_hit.empty:
            r += -1.0 * 0.5
        else:
            r += 0.0
    return ('tp1_partial', r)

def cost_in_R(entry, stop, fee_bps_total, slip_bps_total, exits=1):
    """
    Coste expresado en R (múltiplos de (entry-stop)).
    fee_bps_total: bps por lado; slip_bps_total: bps por lado.
    exits: nº de salidas (1=SL o TP directo; 2=parcial TP1 y TP2).
    """
    price_span = abs(entry - stop)
    if price_span <= 0: return 0.0
    # Coste total ≈ (entrada 1x) + (exits x)
    bps_total = (fee_bps_total + slip_bps_total) * (1 + exits)
    cost_frac = bps_total / 1e4  # de precio
    return (entry * cost_frac) / price_span

def eval_signals(signals_df, ohlc, H, th, fee_bps, slip_bps, partial, be_after_tp1):
    # Filtra por TH si hay columna proba
    df = signals_df.copy()
    if 'proba' in df.columns:
        df = df[df['proba'] >= th]
    if df.empty:
        return {'trades':0, 'wins':0, 'losses':0, 'wr':0.0, 'pf':0.0,
                'sum_R':0.0, 'avg_R':0.0, 'max_dd_R':0.0}

    # Orden temporal
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df = df.sort_values('ts')

    sum_R = 0.0
    equity_R = 0.0
    max_dd = 0.0
    peak = 0.0
    wins = 0
    losses = 0
    gross_win = 0.0
    gross_loss = 0.0

    for _, row in df.iterrows():
        entry, stop = float(row['entry']), float(row['stop'])
        out = first_hit_in_horizon(row, ohlc, H)
        out = apply_partial_logic(out, ohlc, row, H, partial=partial, breakeven_after_tp1=be_after_tp1)

        # nº de salidas para costes
        exits = 2 if out[0] in ('tp2','tp1_partial') else 1
        r_cost = cost_in_R(entry, stop, fee_bps, slip_bps, exits=exits)
        r_net = out[1] - r_cost

        sum_R += r_net
        equity_R += r_net
        peak = max(peak, equity_R)
        dd = peak - equity_R
        max_dd = max(max_dd, dd)

        if r_net > 0:
            wins += 1
            gross_win += r_net
        elif r_net < 0:
            losses += 1
            gross_loss += -r_net

    trades = len(df)
    wr = wins / trades if trades else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float('inf') if gross_win>0 else 0.0)
    avg_R = sum_R / trades if trades else 0.0
    return {'trades': trades, 'wins': wins, 'losses': losses, 'wr': wr,
            'pf': pf, 'sum_R': sum_R, 'avg_R': avg_R, 'max_dd_R': max_dd}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', nargs='+', required=True,
                    help='Lista tipo NOMBRE:YYYY-MM-DD:YYYY-MM-DD (ej. 2023Q4:2023-10-01:2023-12-31)')
    ap.add_argument('--assets', nargs='+', default=['BTC-USD'],
                    help='Activos a evaluar (ej. BTC-USD ETH-USD)')
    ap.add_argument('--horizons', nargs='+', type=int, default=[60])
    ap.add_argument('--thresholds', nargs='+', type=float, default=[0.60])
    ap.add_argument('--signals_root', default='reports/windows',
                    help='Carpeta raiz donde cada ventana tiene sus CSV de señales')
    ap.add_argument('--ohlc_root', default='data/ohlc/1m', help='Carpeta con OHLC 1m csv')
    ap.add_argument('--fee_bps', type=float, default=6.0, help='bps por lado (fee)')
    ap.add_argument('--slip_bps', type=float, default=6.0, help='bps por lado (slippage)')
    ap.add_argument('--partial', choices=['none','50_50'], default='50_50')
    ap.add_argument('--breakeven_after_tp1', action='store_true', default=True)
    ap.add_argument('--risk_total_pct', type=float, default=0.75,
                    help='Riesgo total de la cuenta por “ciclo” (%) para ROI aprox.')
    ap.add_argument('--weights', nargs='*', default=['BTC-USD=0.7','ETH-USD=0.3'],
                    help='Pesos por activo para repartir riesgo, ej. BTC-USD=0.7 ETH-USD=0.3')
    ap.add_argument('--out_csv', default='reports/kpis_grid.csv')
    ap.add_argument('--out_top', default='reports/top_configs.csv')
    ap.add_argument('--gate_pf', type=float, default=1.6)
    ap.add_argument('--gate_wr', type=float, default=0.60)
    ap.add_argument('--gate_trades', type=int, default=30)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    weights = {}
    for kv in args.weights:
        k,v = kv.split('='); weights[k]=float(v)
    # normaliza por si acaso
    s = sum(weights.get(a,0.0) for a in args.assets)
    if s>0:
        for a in args.assets:
            weights[a] = weights.get(a,0.0)/s

    # carga OHLC
    ohlc_map = {a: load_ohlc(os.path.join(args.ohlc_root, f'{a}.csv')) for a in args.assets}

    rows = []
    for w in args.windows:
        name, start, end = w.split(':')
        for H in args.horizons:
            for th in args.thresholds:
                # agrega por activo y además portafolio
                agg_R = 0.0
                agg_grosswin = 0.0
                agg_grossloss = 0.0
                agg_trades = 0
                agg_wins = 0
                agg_losses = 0
                agg_wr_contrib = []

                for a in args.assets:
                    sig_path = os.path.join(args.signals_root, name, f'diamante_signal_{a.replace("/","-")}_gate.csv')
                    if not os.path.exists(sig_path):
                        continue
                    sig = pd.read_csv(sig_path)
                    # recorta por ventana de fechas si el CSV tiene más rango
                    sig['ts'] = pd.to_datetime(sig['ts'], utc=True)
                    sig = sig[(sig['ts']>=pd.Timestamp(start, tz='UTC')) & (sig['ts']<=pd.Timestamp(end, tz='UTC'))]

                    res = eval_signals(sig[sig['asset']==a] if 'asset' in sig.columns else sig,
                                       ohlc_map[a], H, th, args.fee_bps, args.slip_bps,
                                       args.partial, args.breakeven_after_tp1)
                    # guarda por activo
                    row = dict(window=name, start=start, end=end, asset=a, H=H, TH=th,
                               trades=res['trades'], wins=res['wins'], losses=res['losses'],
                               wr=round(res['wr'],4), pf=round(res['pf'],4),
                               sum_R=round(res['sum_R'],4), avg_R=round(res['avg_R'],4),
                               max_dd_R=round(res['max_dd_R'],4),
                               fee_bps=args.fee_bps, slip_bps=args.slip_bps,
                               partial=args.partial, be_after_tp1=args.breakeven_after_tp1)
                    rows.append(row)

                    # acumula para portafolio (en R ponderado por riesgo relativo)
                    agg_trades += res['trades']
                    agg_wins += res['wins']; agg_losses += res['losses']
                    wgt = weights.get(a, 0.0)
                    agg_R += res['sum_R'] * wgt
                    # PF y WR de portafolio los recomputamos luego a partir de trades agregados
                    if res['sum_R']>0: agg_grosswin += res['sum_R']*wgt
                    elif res['sum_R']<0: agg_grossloss += (-res['sum_R'])*wgt
                    if res['trades']>0: agg_wr_contrib.append((res['wins'], res['trades']))

                # fila de portafolio (si hubo al menos un activo)
                if agg_trades>0:
                    wr_num = sum(w for w,t in agg_wr_contrib)
                    wr_den = sum(t for _,t in agg_wr_contrib)
                    wr_p = (wr_num/wr_den) if wr_den>0 else 0.0
                    pf_p = (agg_grosswin/agg_grossloss) if agg_grossloss>0 else (float('inf') if agg_grosswin>0 else 0.0)

                    # ROI aprox asumiendo riesgo total fijo por trade-ciclo (=risk_total_pct%):
                    # ROI% ≈ sum_R_portafolio * (risk_total_pct/100)
                    roi_pct = agg_R * (args.risk_total_pct/100.0) * 100.0

                    rows.append(dict(window=name, start=start, end=end, asset='PORT',
                                     H=H, TH=th, trades=agg_trades, wins=agg_wins, losses=agg_losses,
                                     wr=round(wr_p,4), pf=round(pf_p,4),
                                     sum_R=round(agg_R,4), avg_R=None, max_dd_R=None,
                                     fee_bps=args.fee_bps, slip_bps=args.slip_bps,
                                     partial=args.partial, be_after_tp1=args.breakeven_after_tp1,
                                     roi_pct=round(roi_pct,4)))

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)

    # TOP configs (PORT) por gate
    port = out[out['asset']=='PORT'].copy()
    port['trades'] = port['trades'].fillna(0).astype(int)
    top = port[(port['pf']>=args.gate_pf) & (port['wr']>=args.gate_wr) & (port['trades']>=args.gate_trades)]
    top = top.sort_values(['window','pf','wr'], ascending=[True,False,False])
    top.to_csv(args.out_top, index=False)

    print(f"[OK] KPIs → {args.out_csv}")
    print(f"[OK] TOP (gate PF>={args.gate_pf}, WR>={args.gate_wr}, trades>={args.gate_trades}) → {args.out_top}")

if __name__ == "__main__":
    main()