# -*- coding: utf-8 -*-
from __future__ import annotations

import os, glob, argparse
import pandas as pd, yaml

from .io import load_ohlc
from .sim import simulate, TradeCosts


# ---------- helpers ----------
def _rename_last_reports(reports_dir: str, suffix: str) -> int:
    """
    Renombra el trío más reciente {equity,kpis,summary} agregando __<suffix>.
    Devuelve cuántos archivos renombró (0..3).
    """
    if not suffix:
        return 0
    pats = [
        os.path.join(reports_dir, "base_v0_1_*_equity.csv"),
        os.path.join(reports_dir, "base_v0_1_*_kpis.csv"),
        os.path.join(reports_dir, "base_v0_1_*_summary.md"),
    ]
    renamed = 0
    for pat in pats:
        files = sorted(glob.glob(pat))
        if not files:
            continue
        src = files[-1]
        head, tail = os.path.split(src)
        name, ext = os.path.splitext(tail)
        dst = os.path.join(head, f"{name}__{suffix}{ext}")
        try:
            os.replace(src, dst)
            print(f"[RENAMED] {tail} -> {os.path.basename(dst)}")
            renamed += 1
        except Exception as e:
            print(f"[WARN] Falló rename {tail}: {e}")
    return renamed


def _read_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_costs(cfg: dict):
    """
    Acepta cualquiera de:
      - fee_bps_per_side / slip_bps_per_side
      - fee_bps / slip_bps (interpreta por lado)
      - bps_per_side
    """
    c = (cfg.get("costs") or {})
    if "fee_bps_per_side" in c or "slip_bps_per_side" in c:
        fee = float(c.get("fee_bps_per_side", c.get("fee_bps", 0.0)))
        slp = float(c.get("slip_bps_per_side", c.get("slip_bps", 0.0)))
        return TradeCosts(fee, slp)
    if "fee_bps" in c or "slip_bps" in c:
        fee = float(c.get("fee_bps", 0.0))
        slp = float(c.get("slip_bps", 0.0))
        return TradeCosts(fee, slp)
    if "bps_per_side" in c:
        bps = float(c["bps_per_side"])
        return TradeCosts(bps, 0.0)
    return TradeCosts(6.0, 6.0)


def _ensure_ts(df: pd.DataFrame, preferred: str | None = None) -> pd.DataFrame:
    """Garantiza una columna 'timestamp' tz-aware UTC y ordena por ella."""
    cand = [preferred] if preferred else []
    cand += ['timestamp', 'ts', 'date', 'datetime', 'time', 'Date', 'Datetime', 'Time']
    ts_col = None
    for c in cand:
        if c and (c in df.columns):
            ts_col = c
            break
    if ts_col is None:
        raise KeyError("No encuentro columna temporal en H4 (intenté: " + ", ".join(cand) + ")")
    out = df.copy()
    out['timestamp'] = pd.to_datetime(out[ts_col], utc=True, errors='coerce')
    out = out.dropna(subset=['timestamp']).sort_values('timestamp')
    return out


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mini_accum.yaml")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--suffix", default=os.environ.get("REPORT_SUFFIX", ""))
    args = ap.parse_args()

    cfg = _read_cfg(args.config)

    rep_dir = cfg.get("backtest", {}).get("reports_dir", "reports/mini_accum")
    os.makedirs(rep_dir, exist_ok=True)

    # Carga H4 crudo desde CSV según YAML
    ts_col_cfg = cfg.get("data", {}).get("ts_col", "timestamp")
    tz_in      = cfg.get("data", {}).get("tz_input", "UTC")
    h4 = load_ohlc(cfg["data"]["ohlc_4h_csv"], ts_col_cfg, tz_in)

    # Normaliza la columna temporal (evita KeyError: 'timestamp' en filtros)
    h4 = _ensure_ts(h4, preferred=ts_col_cfg)

    # Filtro temporal
    if args.start:
        t0 = pd.Timestamp(args.start, tz="UTC")
        h4 = h4[h4['timestamp'] >= t0]
    if args.end:
        t1 = pd.Timestamp(args.end, tz="UTC")
        h4 = h4[h4['timestamp'] <= t1]

    # Costes
    costs = _build_costs(cfg)

    # Simulación (modo legacy: a=cfg, b=h4, c=costs) -> D1 lo carga internamente y recorta
    equity_df, kpis = simulate(cfg, h4, costs)

    run_id  = pd.Timestamp.utcnow().strftime("base_v0_1_%Y%m%d_%H%M")
    eq_path = os.path.join(rep_dir, f"{run_id}_equity.csv")
    kp_path = os.path.join(rep_dir, f"{run_id}_kpis.csv")
    md_path = os.path.join(rep_dir, f"{run_id}_summary.md")

    equity_df.to_csv(eq_path, index=False)
    pd.DataFrame([kpis]).to_csv(kp_path, index=False)

    # Veredicto simple (si existen umbrales)
    acc = (cfg.get("kpis", {}).get("accept") or {})
    btc_min      = float(acc.get("net_btc_ratio_min", float("-inf")))
    mdd_vs_max   = float(acc.get("mdd_vs_hodl_ratio_max", float("inf")))
    flips_yr_max = float(acc.get("flips_per_year_max", float("inf")))

    ok_btc  = kpis.get("net_btc_ratio", float("-inf")) >= btc_min
    ok_mdd  = kpis.get("mdd_vs_hodl_ratio", float("inf")) <= mdd_vs_max
    ok_flip = kpis.get("flips_per_year", float("inf")) <= flips_yr_max
    verdict = "ACEPTAR" if (ok_btc and ok_mdd and ok_flip) else "RECHAZAR"

    with open(md_path, "w") as f:
        f.write(f"# Mini-BOT BTC v0.1 — Resumen {run_id}\n\n")
        for k, v in kpis.items():
            f.write(f"- **{k}**: {v}\n")
        f.write(f"\n**Veredicto:** {verdict}\n")

    print("[OK]", eq_path)
    print("[OK]", kp_path)
    print("[OK]", md_path)

    # Renombrado opcional con sufijo
    if args.suffix:
        _rename_last_reports(rep_dir, args.suffix)


if __name__ == "__main__":
    main()