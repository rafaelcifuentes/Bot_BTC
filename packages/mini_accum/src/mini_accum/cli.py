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
