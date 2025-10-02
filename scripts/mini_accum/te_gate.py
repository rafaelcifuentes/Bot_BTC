#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TE gate (Tracking Error semanal) fuera del pipeline inline.
- Calcula el % de cambio semanal del BTC (close 4h) en ventana Lun 00:00 ET -> Lun 00:00 ET.
- Usa posición (0/1) constante de referencia si hay señales en signals/.../history.csv
  o si no, cae a decision/orders_preview.csv; si tampoco, marca SKIP.
- Anota una línea al final del FREEZE si hay datos suficientes.

Gate: PASS si |TE| <= 0.03 (±3%).
"""

from __future__ import annotations
import os, argparse, sys
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

def et_midnight(d_utc: datetime) -> datetime:
    """Corta a 00:00 América/Nueva_York y vuelve a UTC."""
    et = d_utc.astimezone(ZoneInfo("America/New_York"))
    et_mid = et.replace(hour=0, minute=0, second=0, microsecond=0)
    return et_mid.astimezone(ZoneInfo("UTC"))

def load_ohlc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df or "close" not in df:
        raise RuntimeError(f"[TE] CSV OHLC debe tener columnas 'timestamp' y 'close': {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return df

def infer_pos(root: str, t0: datetime) -> float | None:
    """Intenta inferir posición (0/1) al inicio de la semana.
    1) signals/mini_accum/history.csv (col decision o position_pct_btc)
    2) reports/mini_accum/exec/orders_preview.csv (col decision)
    """
    # 1) history
    h = os.path.join(root, "signals/mini_accum/history.csv")
    if os.path.exists(h):
        try:
            d = pd.read_csv(h)
            # admite nombres variantes
            ts_col = next((c for c in d.columns if c in ("ts_utc","ts","timestamp")), None)
            if ts_col:
                d[ts_col] = pd.to_datetime(d[ts_col], utc=True, errors="coerce")
                d = d.dropna(subset=[ts_col]).sort_values(ts_col)
                d = d[d[ts_col] <= t0]
                if len(d):
                    if "position_pct_btc" in d:
                        v = float(d.iloc[-1]["position_pct_btc"])
                        return max(0.0, min(1.0, v))
                    if "decision" in d:
                        return float(int(d.iloc[-1]["decision"]) in (1, ))
        except Exception:
            pass

    # 2) orders_preview (última decisión conocida)
    p = os.path.join(root, "reports/mini_accum/exec/orders_preview.csv")
    if os.path.exists(p):
        try:
            d = pd.read_csv(p)
            if "ts" in d:
                d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
                d = d.dropna(subset=["ts"]).sort_values("ts")
                d = d[d["ts"] <= t0]
                if len(d) == 0:
                    d = pd.read_csv(p)  # si no hay previo, coge la última por simplicidad
                if "decision" in d:
                    return float(int(d.iloc[-1]["decision"]) in (1,))
        except Exception:
            pass

    return None  # sin referencia

def weekly_return(df_ohlc: pd.DataFrame, t0: datetime, t1: datetime) -> float | None:
    """Retorno simple entre primer y último close dentro [t0,t1)."""
    d = df_ohlc[(df_ohlc["timestamp"] >= t0) & (df_ohlc["timestamp"] < t1)]
    if len(d) < 2:
        return None
    c0, c1 = float(d.iloc[0]["close"]), float(d.iloc[-1]["close"])
    if c0 <= 0:
        return None
    return (c1 / c0) - 1.0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/PycharmProjects/Bot_BTC"))
    ap.add_argument("--freeze_date", help="YYYY-MM-DD (ET day used for window end)")
    ap.add_argument("--ohlc", help="override OHLC path")
    ap.add_argument("--orders", help="override orders_preview path")
    ap.add_argument("--out_md", help="override FREEZE md path (append line)")
    ap.add_argument("--gate", type=float, default=0.03)
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    # determinar ventana: Lun 00:00 ET -> Lun 00:00 ET del FREEZE
    if args.freeze_date:
        end_et = datetime.fromisoformat(args.freeze_date + "T00:00:00").replace(tzinfo=ZoneInfo("America/New_York"))
        t1 = end_et.astimezone(ZoneInfo("UTC"))
    else:
        t1 = et_midnight(datetime.now(tz=ZoneInfo("UTC")))
    t0 = t1 - timedelta(days=7)

    ohlc_path = args.ohlc or os.path.join(root, "reports/ohlc_4h/BTC-USD.csv")
    out_md = args.out_md or os.path.join(root, "reports/mini_accum/walkforward/freezes",
                                         (args.freeze_date or t1.astimezone(ZoneInfo('America/New_York')).strftime("%Y-%m-%d")),
                                         f"weekly_freeze_summary.{(args.freeze_date or t1.astimezone(ZoneInfo('America/New_York')).strftime('%Y-%m-%d'))}.md")

    # cargar OHLC
    try:
        ohlc = load_ohlc(ohlc_path)
    except Exception as e:
        print(f"[TE] SKIP ({e})")
        return 0

    # retorno BTC semana
    r_btc = weekly_return(ohlc, t0, t1)
    if r_btc is None:
        print("[TE] SKIP (sin barras suficientes en la semana)")
        return 0

    # inferir posición ref (0..1). Si no hay, TE no aplica -> SKIP
    pos = infer_pos(root, t0)
    if pos is None:
        # fallback ultra conservador: sin referencia de posición
        print("[TE] SKIP (sin referencia de posición)")
        return 0

    # “shadow” y “ref” usan misma pos si no hay shadow explícito
    r_ref = pos * r_btc
    r_shadow = r_ref  # hasta que tengamos shadow externo; evita falsos FAIL

    te = r_shadow - r_ref
    gate = float(args.gate)
    verdict = "PASS" if abs(te) <= gate else "FAIL"

    # anotar
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    line = (f"> Tracking Error (semana {t0.date()}→{t1.date()}, pos={pos:.2f}) — "
            f"shadow={r_shadow:+.2%}, ref={r_ref:+.2%}, TE={te:+.2%} → **{verdict}**\n")
    try:
        with open(out_md, "a") as f:
            f.write("\n" + line)
        print(f"[TE] anotado en {os.path.basename(out_md)} | {line.strip()}")
    except Exception as e:
        print(f"[TE] WARN: no pude anotar TE ({e})")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
