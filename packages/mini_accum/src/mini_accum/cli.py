#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd
import yaml

from .io import load_ohlc, merge_daily_into_4h
from .sim import simulate, TradeCosts

__all__ = ["main"]


def _rename_with_suffix(path: str, suffix: Optional[str]) -> str:
    """
    Si hay suffix, renombra foo.csv -> foo__{suffix}.csv.
    Devuelve la ruta final (renombrada u original).
    """
    if not suffix:
        return path
    base, ext = os.path.splitext(path)
    new_path = f"{base}__{suffix}{ext}"
    try:
        os.replace(path, new_path)
        print(f"[RENAMED] {os.path.basename(path)} -> {os.path.basename(new_path)}")
        return new_path
    except Exception as e:
        print(f"[WARN] rename failed for {path}: {e}")
        return path

# --- guardias previos al rename ---
import csv, sys

def _kpi_netbtc_or_none(kpi_csv: str):
    """
    Lee el CSV de KPIs y devuelve el valor de 'netBTC' como float si existe.
    Fallback: intenta encontrar el primer valor numérico de la primera fila de datos.
    """
    try:
        with open(kpi_csv, newline='') as fh:
            r = csv.DictReader(fh)
            first = next(r, None)
            if not first:
                return None
            # Preferimos columna 'netBTC' explícita
            if "netBTC" in first and (first["netBTC"] or "").strip():
                try:
                    return float(first["netBTC"])
                except Exception:
                    return None
            # Fallback: primer valor numérico en la fila
            for v in first.values():
                s = (v or "").strip()
                if not s:
                    continue
                try:
                    return float(s)
                except Exception:
                    continue
            return None
    except Exception:
        return None

def _flips_has_executed(flips_csv: str):
    try:
        with open(flips_csv, newline='') as fh:
            r = csv.DictReader(fh)
            for row in r:
                if (row.get('executed') or '').strip():
                    return True
    except Exception:
        pass
    return False

def _print_summary_and_save_flips(res: pd.DataFrame, rep_dir: str, run_id: str, suffix: Optional[str]) -> None:
    """
    Guarda *_flips.csv y saca un mini-resumen al final.
    Requiere que res tenga columnas: ts, executed, open, close.
    """
    if res.empty or "executed" not in res.columns:
        print("[SUMMARY] flips_total=0 (sin filas ejecutadas)")
        return

    flips = res[res["executed"].notna()][["ts", "executed", "open", "close"]].copy()
    flips_path = os.path.join(rep_dir, f"{run_id}_flips.csv")
    flips.to_csv(flips_path, index=False)
    flips_path = _rename_with_suffix(flips_path, suffix)

    buy_count = int((flips["executed"] == "BUY").sum())
    sell_count = int((flips["executed"] == "SELL").sum())
    total = len(flips)
    print(f"[SUMMARY] flips_total={total} (BUY={buy_count}, SELL={sell_count}) | flips_csv={os.path.basename(flips_path)}")

    if total:
        print("\n== FLIPS ejecutados ==")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(flips.to_string(index=False))

        iso = pd.to_datetime(flips["ts"]).dt.isocalendar()
        weekly = flips.groupby([iso.year, iso.week])["executed"].count()
        print("\n== Flips por semana ==")
        print(weekly)


def main() -> None:
    ap = argparse.ArgumentParser(description="Mini-BOT BTC (mini_accum) — backtest CLI")
    ap.add_argument("--config", default="configs/mini_accum/config.yaml", help="Ruta al YAML de configuración")
    ap.add_argument("--start", default=None, help="ISO date (UTC). Ej: 2024-01-01")
    ap.add_argument("--end", default=None, help="ISO date (UTC). Ej: 2024-06-30")
    ap.add_argument("--suffix", default=None, help="Sufijo de reporte; si no se pasa, se usa $REPORT_SUFFIX si existe")
    args = ap.parse_args()

    # suffix precedence: CLI arg > env var
    suffix = args.suffix if args.suffix else os.environ.get("REPORT_SUFFIX")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    rep_dir = cfg["backtest"]["reports_dir"]
    os.makedirs(rep_dir, exist_ok=True)

    # Cargar datos
    df4 = load_ohlc(cfg["data"]["ohlc_4h_csv"], cfg["data"]["ts_col"], cfg["data"]["tz_input"])
    d1 = load_ohlc(cfg["data"]["ohlc_d1_csv"], cfg["data"]["ts_col"], cfg["data"]["tz_input"])

    # Merge D1→4h y filtro temporal
    df = merge_daily_into_4h(df4, d1)
    if args.start:
        df = df[df["ts"] >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        df = df[df["ts"] <= pd.Timestamp(args.end, tz="UTC")]

    # Costes
    costs = TradeCosts(
        fee_bps_per_side=float(cfg["costs"]["fee_bps_per_side"]),
        slip_bps_per_side=float(cfg["costs"]["slip_bps_per_side"]),
    )

    # Simulación
    res, kpis = simulate(cfg, df, costs)

    # Guardar salidas
    run_id = pd.Timestamp.utcnow().strftime("base_v0_1_%Y%m%d_%H%M")
    eq_path = os.path.join(rep_dir, f"{run_id}_equity.csv")
    kpi_path = os.path.join(rep_dir, f"{run_id}_kpis.csv")
    md_path = os.path.join(rep_dir, f"{run_id}_summary.md")

    res.to_csv(eq_path, index=False)
    kpi_df = pd.DataFrame([kpis])
    # Reordenar columnas para mostrar primero métricas clave (si existen)
    cols = list(kpi_df.columns)
    priority = [
        c for c in [
            "netBTC", "net_btc_ratio",
            "mdd_vs_HODL", "mdd_vs_hodl_ratio",
            "fpy", "flips_per_year"
        ] if c in cols
    ]
    others = [c for c in cols if c not in priority]
    kpi_df = kpi_df[priority + others] if priority else kpi_df
    kpi_df.to_csv(kpi_path, index=False)
    with open(md_path, "w") as f:
        f.write(f"# Mini-BOT BTC v0.1 — Resumen {run_id}\n\n")
        f.write("## KPIs\n")
        for k, v in kpis.items():
            f.write(f"- **{k}**: {v}\n")

    print(f"[OK] {eq_path}")
    print(f"[OK] {kpi_path}")
    print(f"[OK] {md_path}")

    # --- Guardarraíl antes de renombrar ---
    # Toleramos variaciones de nombre en KPIs entre engines: netBTC | net_btc_ratio | net_btc | net
    net_candidate = (
        kpis.get("netBTC")
        or kpis.get("net_btc_ratio")
        or kpis.get("net_btc")
        or kpis.get("net")
    )
    try:
        net = float(net_candidate) if net_candidate is not None else float(_kpi_netbtc_or_none(kpi_path) or 0.0)
    except Exception:
        net = float(_kpi_netbtc_or_none(kpi_path) or 0.0)

    has_flips = ("executed" in res.columns) and res["executed"].notna().any()

    if not (net > 0.0 and has_flips):
        # Caso inválido: NO renombrar; guardar flips sin sufijo para inspección
        print(f"[SKIP] KPI/Flips inválidos (netBTC={net:.4f}, flips={has_flips}); no renombro artefactos.")
        _print_summary_and_save_flips(res, rep_dir, run_id, suffix=None)
        return

    # Caso válido: renombrar artefactos y flips con el sufijo
    eq_path = _rename_with_suffix(eq_path, suffix)
    kpi_path = _rename_with_suffix(kpi_path, suffix)
    md_path = _rename_with_suffix(md_path, suffix)
    _print_summary_and_save_flips(res, rep_dir, run_id, suffix)

if __name__ == "__main__":
    main()