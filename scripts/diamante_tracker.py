#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diamante - Mini tracker de resultados por activo/umbral.
Lee CSVs de reports/ del tipo:
  - diamante_<asset>_t0XX_week1.csv   (umbral explícito 0.60/0.61/0.62)
  - diamante_<asset>_week1.csv        (baseline sin sufijo t0XX, opcional)

Produce:
  - reports/diamante_tracker.csv   (todas las filas)
  - reports/diamante_tracker_best.csv (mejor por activo)
  - reports/diamante_tracker.md    (resumen en Markdown)
Imprime el resumen en consola.
"""

import argparse
import glob
import os
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


PAT_T = re.compile(r"diamante_(?P<asset>[a-z]+)_t(?P<th>\d{3})_week1\.csv$", re.I)
PAT_BASE = re.compile(r"diamante_(?P<asset>[a-z]+)_week1\.csv$", re.I)


def _parse_filename(fp: str):
    fn = os.path.basename(fp)
    m = PAT_T.search(fn)
    if m:
        asset = m.group("asset").upper()
        th = round(int(m.group("th")) / 100.0, 2)
        return asset, th, "grid"
    m = PAT_BASE.search(fn)
    if m:
        asset = m.group("asset").upper()
        return asset, np.nan, "baseline"
    return None, None, None


def _pick_row_for_horizon(df: pd.DataFrame, horizon: int) -> pd.Series | None:
    # Espera columnas: days, net, pf, win_rate, trades, mdd
    if "days" not in df.columns:
        # intentar index como columna
        if df.index.name == "days":
            df = df.reset_index()
        else:
            return None
    try:
        df["days_i"] = df["days"].astype(int)
    except Exception:
        return None
    # Fila exacta o la más cercana
    if (df["days_i"] == horizon).any():
        row = df.loc[df["days_i"] == horizon].iloc[0]
    else:
        ix = (df["days_i"] - horizon).abs().idxmin()
        row = df.loc[ix]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--out_csv", default="reports/diamante_tracker.csv")
    ap.add_argument("--out_csv_best", default="reports/diamante_tracker_best.csv")
    ap.add_argument("--out_md", default="reports/diamante_tracker.md")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--gate_pf", type=float, default=1.6)
    ap.add_argument("--gate_wr", type=float, default=0.60)
    ap.add_argument("--gate_trades", type=float, default=30)
    args = ap.parse_args()

    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    # Buscar archivos relevantes
    files = []
    files += glob.glob(str(reports / "diamante_*_t0??_week1.csv"))
    files += glob.glob(str(reports / "diamante_*_week1.csv"))
    files = sorted(set(files))

    rows = []
    for fp in files:
        asset, th, kind = _parse_filename(fp)
        if not asset:
            continue
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        row = _pick_row_for_horizon(df, args.horizon)
        if row is None:
            continue

        # Limpieza de posibles strings "inf"
        pf = row.get("pf", np.nan)
        try:
            pf = float(pf)
        except Exception:
            pf = np.inf if str(pf).lower() in ("inf", "infinity") else np.nan

        wr = float(row.get("win_rate", np.nan))
        trades = float(row.get("trades", np.nan))
        mdd = float(row.get("mdd", np.nan))
        net = float(row.get("net", np.nan))

        rows.append(
            {
                "asset": asset,
                "threshold": th,     # NaN si baseline
                "kind": kind,        # grid/baseline
                "pf_60d": pf,
                "wr_60d": wr,
                "trades_60d": trades,
                "mdd_60d": mdd,
                "net_60d": net,
                "src": os.path.basename(fp),
            }
        )

    if not rows:
        print("[WARN] No se encontraron CSVs compatibles en", reports)
        return

    df_all = pd.DataFrame(rows).sort_values(["asset", "threshold", "kind"])

    # Score simple para desempatar: pondera PF y WR, penaliza MDD
    # safe guards por NaN/inf
    df_all["pf_capped"] = df_all["pf_60d"].replace(np.inf, 10.0).clip(0, 10)
    df_all["wr01"] = df_all["wr_60d"].clip(0, 1.0)
    df_all["mdd_pos"] = df_all["mdd_60d"].abs()
    df_all["score"] = 0.7 * df_all["pf_capped"] + 0.3 * (df_all["wr01"] * 2.0) - 0.2 * (df_all["mdd_pos"] * 100)

    # Gates plan Sem.1
    df_all["gate_ok"] = (
        (df_all["pf_60d"].replace(np.inf, 10.0) >= args.gate_pf)
        & (df_all["wr_60d"] >= args.gate_wr)
        & (df_all["trades_60d"] >= args.gate_trades)
    )

    # Mejor por activo (por PF; si PF empata, por score)
    df_all["rank_pf"] = df_all.groupby("asset")["pf_60d"].rank(ascending=False, method="first")
    best_pf = df_all.sort_values(["asset", "pf_60d", "score"], ascending=[True, False, False]).groupby("asset").head(1)

    # Guardar
    df_all.drop(columns=["pf_capped", "wr01", "mdd_pos"], errors="ignore").to_csv(args.out_csv, index=False)
    best_pf.drop(columns=["pf_capped", "wr01", "mdd_pos"], errors="ignore").to_csv(args.out_csv_best, index=False)

    # Markdown resumido
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = []
    lines.append(f"# Diamante · Tracker (h={args.horizon}d)\n")
    lines.append(f"_Generado_: **{ts}**  \n")
    lines.append("## Recomendación por activo (máx PF)\n")
    for _, r in best_pf.iterrows():
        th_txt = f"{r['threshold']:.2f}" if pd.notna(r["threshold"]) else "baseline"
        gate = "✅" if r["gate_ok"] else "⚠️"
        lines.append(
            f"- **{r['asset']}** → th **{th_txt}** | PF **{r['pf_60d']:.3f}** | WR **{r['wr_60d']:.1%}** | "
            f"trades **{int(r['trades_60d'])}** | MDD **{r['mdd_60d']:.3%}** | net **{r['net_60d']:.2f}** | {gate} "
            f"(_src: {r['src']}_)"
        )
    lines.append("\n## Tabla completa (top por PF ↓)\n")
    df_print = df_all.sort_values(["asset", "pf_60d"], ascending=[True, False]).copy()
    # Formateo amigable
    for c in ["pf_60d", "wr_60d", "mdd_60d", "net_60d", "score"]:
        if c in df_print.columns:
            df_print[c] = pd.to_numeric(df_print[c], errors="coerce")
    df_print["wr_60d"] = (df_print["wr_60d"] * 100).map(lambda x: f"{x:.1f}%")
    df_print["mdd_60d"] = (df_print["mdd_60d"] * 100).map(lambda x: f"{x:.2f}%")
    df_print["threshold"] = df_print["threshold"].map(lambda x: "baseline" if pd.isna(x) else f"{x:.2f}")
    lines.append(df_print[["asset", "threshold", "pf_60d", "wr_60d", "trades_60d", "mdd_60d", "net_60d", "kind", "src"]].to_markdown(index=False))

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")

    # Consola
    print("== Recomendación por activo (máx PF) ==")
    print(best_pf[["asset", "threshold", "pf_60d", "wr_60d", "trades_60d", "mdd_60d", "net_60d", "gate_ok", "src"]]
          .to_string(index=False))
    print(f"\nCSV → {args.out_csv}\nCSV(best) → {args.out_csv_best}\nMD → {args.out_md}")


if __name__ == "__main__":
    main()