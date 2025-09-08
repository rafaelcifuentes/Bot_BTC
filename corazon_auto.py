#!/usr/bin/env python3
# corazon_auto.py — corre presets (ALL y LONG_V2), extrae métricas y agrega fila a un CSV diario.

import argparse
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd


def run(cmd, env=None):
    # Log comando "bonito" igual que en tus logs previos
    human = " ".join(cmd)
    print(f"→ {sys.executable} {human}")
    res = subprocess.run([sys.executable] + cmd, env=env, check=False)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def read_plus_row(csv_base: str, horizon_days=60) -> dict:
    """
    runner_corazon.py genera dos archivos: base.csv y base_plus.csv
    Leemos el *_plus.csv y devolvemos los campos de interés (fila 'days'==horizon)
    """
    plus = Path(csv_base).with_name(Path(csv_base).stem + "_plus.csv").with_suffix(".csv")
    if not plus.exists():
        # fallback al CSV base si el + no existe
        df = pd.read_csv(csv_base)
    else:
        df = pd.read_csv(plus)

    # days está como columna (30/60/90)
    if "days" not in df.columns:
        raise RuntimeError(f"CSV sin columna 'days': {plus}")

    row = df.loc[df["days"] == horizon_days]
    if row.empty:
        # toma última fila si por alguna razón no existe 'horizon'
        row = df.tail(1)

    r = row.iloc[0].to_dict()
    # Normaliza nombres que usamos
    out = {
        "pf_30d": float(df.loc[df["days"] == 30, "pf"].iloc[0]) if 30 in df["days"].values else float("nan"),
        "pf_60d": float(r.get("pf", float("nan"))),
        "pf_90d": float(df.loc[df["days"] == 90, "pf"].iloc[0]) if 90 in df["days"].values else float("nan"),
        "wr_60d": float(r.get("win_rate", float("nan"))),
        "mdd_60d": float(r.get("mdd", float("nan"))),
        "trades_60d": float(r.get("trades", float("nan"))),
        "net_60d": float(r.get("net", float("nan"))),
    }
    return out


def append_report(report_csv: Path, rows: list[dict]):
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    dfrows = pd.DataFrame(rows)
    header = not report_csv.exists()
    dfrows.to_csv(report_csv, mode="a", header=header, index=False)
    print(f"📄 Reporte actualizado → {report_csv}")
    print(dfrows.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exchange", default="binanceus")
    p.add_argument("--symbol", default="BTC/USD")  # aceptado por compatibilidad (no se pasa al runner)
    p.add_argument("--fg_csv", default="./data/sentiment/fear_greed.csv")
    p.add_argument("--funding_csv", default="./data/sentiment/funding_rates.csv")
    p.add_argument("--max_bars", type=int, default=975)
    p.add_argument("--freeze_end", default="2025-08-05 00:00")
    p.add_argument("--threshold", type=float, default=0.60)
    p.add_argument("--compare_both", action="store_true", help="Corre ALL y LONG_V2 y compara")
    p.add_argument("--report_csv", default="reports/corazon_auto_daily.csv")

    # Gates LONG_V2 por defecto (tus presets)
    p.add_argument("--fg_long_min", type=float, default=-0.15)
    p.add_argument("--fg_short_max", type=float, default=0.15)
    p.add_argument("--funding_bias", type=float, default=0.005)
    p.add_argument("--adx1d_len", type=int, default=14)
    # Aceptamos --adx_min por comodidad; si viene, se usa como adx1d_min si no fue seteado explícitamente
    p.add_argument("--adx_min", type=float, default=None)
    p.add_argument("--adx1d_min", type=float, default=22.0)
    p.add_argument("--adx4_min", type=float, default=12.0)

    args = p.parse_args()

    env = os.environ.copy()
    env["EXCHANGE"] = args.exchange

    # === 1) ALL (sin gates) ===
    out_all = "reports/corazon_metrics_auto_all.csv"
    cmd_all = [
        "runner_corazon.py",
        "--max_bars", str(args.max_bars),
        "--fg_csv", args.fg_csv,
        "--funding_csv", args.funding_csv,
        "--threshold", f"{args.threshold:.2f}",
        "--out_csv", out_all,
        "--freeze_end", args.freeze_end,
        "--no_gates",
    ]
    run(cmd_all, env=env)
    all_metrics = read_plus_row(out_all)

    rows = []
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows.append({
        "ts": now_ts,
        "exchange": args.exchange,
        "symbol": "BTC/USD",
        "freeze_end": args.freeze_end,
        "label": "ALL",
        "threshold": args.threshold,
        **all_metrics,
    })

    # === 2) LONG_V2 (con gates) ===
    if args.adx_min is not None and (args.adx1d_min is None or args.adx1d_min == 0):
        adx1d_min = args.adx_min
    else:
        adx1d_min = args.adx1d_min

    out_long = "reports/corazon_metrics_auto_longv2.csv"
    cmd_long = [
        "runner_corazon.py",
        "--max_bars", str(args.max_bars),
        "--fg_csv", args.fg_csv,
        "--funding_csv", args.funding_csv,
        "--threshold", f"{args.threshold:.2f}",
        "--out_csv", out_long,
        "--freeze_end", args.freeze_end,
        "--fg_long_min", str(args.fg_long_min),
        "--fg_short_max", str(args.fg_short_max),
        "--funding_bias", str(args.funding_bias),
        "--adx1d_len", str(args.adx1d_len),
        "--adx1d_min", str(adx1d_min),
        "--adx4_min", str(args.adx4_min),
    ]

    if args.compare_both:
        run(cmd_long, env=env)
        long_metrics = read_plus_row(out_long)
        rows.append({
            "ts": now_ts,
            "exchange": args.exchange,
            "symbol": "BTC/USD",
            "freeze_end": args.freeze_end,
            "label": "LONG_V2",
            "threshold": args.threshold,
            **long_metrics,
        })

    # === Ranking y guardado ===
    df = pd.DataFrame(rows)
    # rank por pf_60d (desc), NaN al final
    df["rank"] = (-df["pf_60d"]).rank(method="min", na_option="bottom")
    df = df.sort_values("rank")

    append_report(Path(args.report_csv), df.to_dict(orient="records"))

    # Recomendación
    best = df.iloc[0]
    best_label = best["label"]
    th = float(best["threshold"])
    print(f"⭐ Recomendación (pf_60d): {best_label} @ th={th:.2f}")


if __name__ == "__main__":
    main()