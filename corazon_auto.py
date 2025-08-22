#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Dependencias: ccxt y pandas_ta ya están en tu entorno
import ccxt
import pandas_ta as ta


def fetch_ohlcv_4h(exchange_id: str, symbol: str, limit: int = 1000):
    ex_class = getattr(ccxt, exchange_id)
    ex = ex_class({"enableRateLimit": True})
    ex.load_markets()
    data = ex.fetch_ohlcv(symbol, timeframe="4h", limit=limit)
    if not data or len(data) == 0:
        raise RuntimeError("No OHLCV data returned")
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(df["ts"], unit="ms", utc=True).tz_convert(None)
    df.index = ts
    df = df.drop(columns=["ts"])
    return df


def adx_daily_and_atr_pct(df_4h: pd.DataFrame, adx_len: int = 14, atr_len: int = 14):
    # Resample 4h → 1D OHLC
    o = df_4h["open"].resample("1D").first()
    h = df_4h["high"].resample("1D").max()
    l = df_4h["low"].resample("1D").min()
    c = df_4h["close"].resample("1D").last()

    # Elimina días vacíos (si los hubiera)
    valid = (~o.isna()) & (~h.isna()) & (~l.isna()) & (~c.isna())
    o, h, l, c = o[valid], h[valid], l[valid], c[valid]

    # ADX
    adx_df = ta.adx(h, l, c, length=adx_len)
    if adx_df is None or adx_df.empty:
        raise RuntimeError("ADX calc returned empty frame")
    adx_col = [col for col in adx_df.columns if col.startswith("ADX_")]
    adx_last = float(adx_df[adx_col[0]].dropna().iloc[-1])

    # ATR% = ATR / Close
    atr_series = ta.atr(h, l, c, length=atr_len)
    if atr_series is None or atr_series.empty:
        raise RuntimeError("ATR calc returned empty series")
    atr_last = float(atr_series.dropna().iloc[-1])
    close_last = float(c.dropna().iloc[-1])
    atr_pct = float(atr_last / close_last)

    return adx_last, atr_pct


def run_corazon(threshold: float, args, out_csv: str):
    """
    Ejecuta runner_corazon.py con un umbral dado y devuelve el DataFrame de métricas (30/60/90d).
    """
    cmd = [
        sys.executable, "runner_corazon.py",
        "--max_bars", str(args.max_bars),
        "--fg_csv", args.fg_csv,
        "--funding_csv", args.funding_csv,
        "--no_gates",
        "--threshold", f"{threshold:.2f}",
        "--out_csv", out_csv,
    ]
    if args.freeze_end:
        cmd.extend(["--freeze_end", args.freeze_end])

    env = os.environ.copy()
    env["EXCHANGE"] = args.exchange

    print(f"▶️  Ejecutando Corazón con threshold={threshold:.2f} …")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f"runner_corazon.py fallo (umbral {threshold}):\n{res.stdout}")

    # Lee métricas resultantes
    if not os.path.exists(out_csv):
        raise FileNotFoundError(f"No se encontró {out_csv}")
    df = pd.read_csv(out_csv)
    # Esperado: days,net,pf,win_rate,trades,mdd,sortino,roi_pct
    return df


def pick_threshold(adx1d: float, atr_pct: float, adx_min: float, atr_max: float,
                   th_default: float = 0.60, th_trend: float = 0.63):
    """
    Regla simple:
      - Si mercado tendencial y tranquilo → 0.63
      - Si no → 0.60
    """
    if adx1d >= adx_min and atr_pct <= atr_max:
        return th_trend, "ADX1D≥{:.0f} & ATR%≤{:.1f}% (tendencia + baja vol)".format(adx_min, atr_max*100.0)
    else:
        return th_default, "Condición base (mixto o vol alta)"


def assemble_row(date_utc: str, symbol: str, th_sel: float, reason: str,
                 adx1d: float, atr_pct: float,
                 df_sel: pd.DataFrame,
                 th_alt: float = None, df_alt: pd.DataFrame = None):
    # Extrae métricas por horizonte
    def pick_row(days_val):
        r = df_sel.loc[df_sel["days"] == days_val]
        return r.iloc[0] if not r.empty else None

    out = {
        "date_utc": date_utc,
        "symbol": symbol,
        "threshold_sel": th_sel,
        "reason": reason,
        "adx1d": round(adx1d, 3),
        "atr14_pct": round(atr_pct * 100.0, 3),
    }

    for d in (30, 60, 90):
        r = pick_row(d)
        if r is not None:
            out[f"pf_{d}d"] = r["pf"]
            out[f"wr_{d}d"] = r["win_rate"]
            out[f"mdd_{d}d"] = r["mdd"]
            out[f"net_{d}d"] = r["net"]
            out[f"trades_{d}d"] = r["trades"]

    if th_alt is not None and df_alt is not None:
        # también guardamos referencia del alternativo
        for d in (30, 60, 90):
            r = df_alt.loc[df_alt["days"] == d]
            if not r.empty:
                out[f"pf_{d}d_alt"] = r.iloc[0]["pf"]
                out[f"net_{d}d_alt"] = r.iloc[0]["net"]
        out["threshold_alt"] = th_alt

    return out


def main():
    parser = argparse.ArgumentParser(description="Selector automático de umbral + informe Corazón")
    parser.add_argument("--exchange", default=os.environ.get("EXCHANGE", "binanceus"))
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--max_bars", type=int, default=975)
    parser.add_argument("--freeze_end", default=os.environ.get("FREEZE", ""), help="Opcional (freeze)")
    parser.add_argument("--fg_csv", default=os.environ.get("FG", "./data/sentiment/fear_greed.csv"))
    parser.add_argument("--funding_csv", default=os.environ.get("FU", "./data/sentiment/funding_rates.csv"))
    parser.add_argument("--out_tmp", default="reports/corazon_metrics_auto_tmp.csv")
    parser.add_argument("--out_tmp_alt", default="reports/corazon_metrics_auto_alt.csv")
    parser.add_argument("--report_csv", default="reports/corazon_auto_daily.csv",
                        help="CSV ensamblado (se agrega una fila por ejecución)")
    # Reglas del selector
    parser.add_argument("--adx_min", type=float, default=30.0)
    parser.add_argument("--atr_pct_max", type=float, default=0.025, help="2.5% = 0.025")
    parser.add_argument("--th_default", type=float, default=0.60)
    parser.add_argument("--th_trend", type=float, default=0.63)
    # Opcional: comparar con el otro umbral
    parser.add_argument("--compare_both", action="store_true", help="Corre también el umbral alterno")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.report_csv), exist_ok=True)

    # 1) Datos 4h → ADX1D & ATR%
    print(f"⚙️  EXCHANGE={args.exchange}  symbol={args.symbol}  timeframe=4h")
    df4h = fetch_ohlcv_4h(args.exchange, args.symbol, limit=args.max_bars)
    adx1d, atr_pct = adx_daily_and_atr_pct(df4h, adx_len=14, atr_len=14)
    print(f"🔎 Diagnóstico mercado → ADX1D={adx1d:.2f} | ATR14%={atr_pct*100:.2f}%")

    # 2) Selección de umbral
    th_sel, reason = pick_threshold(adx1d, atr_pct, args.adx_min, args.atr_pct_max,
                                    th_default=args.th_default, th_trend=args.th_trend)
    th_alt = args.th_trend if abs(th_sel - args.th_default) < 1e-9 else args.th_default

    # 3) Backtest con umbral seleccionado (y opcional con alterno)
    df_sel = run_corazon(threshold=th_sel, args=args, out_csv=args.out_tmp)

    df_alt = None
    if args.compare_both:
        df_alt = run_corazon(threshold=th_alt, args=args, out_csv=args.out_tmp_alt)

    # 4) Ensamble de registro diario
    now_utc = datetime.now(timezone.utc).replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    row = assemble_row(now_utc, args.symbol, th_sel, reason, adx1d, atr_pct, df_sel,
                       th_alt=th_alt if args.compare_both else None,
                       df_alt=df_alt)
    # Append al CSV de reporte
    df_out = pd.DataFrame([row])
    if os.path.exists(args.report_csv):
        df_prev = pd.read_csv(args.report_csv)
        df_out = pd.concat([df_prev, df_out], ignore_index=True)

    df_out.to_csv(args.report_csv, index=False)
    print(f"✅ Informe ensamblado → {args.report_csv}")
    print(f"   Umbral elegido: {th_sel:.2f}  | Motivo: {reason}")
    print("   Columns:", ", ".join(df_out.columns))


if __name__ == "__main__":
    main()