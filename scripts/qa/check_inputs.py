#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chequeos rápidos de entrada antes del pipeline:
- Valida ohlc_4h (timestamp, close) y orders_preview (ts, decision).
- Reporta duplicados y rango temporal.
- Falla con exit!=0 si encuentra problemas.
"""

from __future__ import annotations
import os
import sys
from datetime import timezone

import pandas as pd

# Importa validadores
from schemas import validate_ohlc_4h, validate_orders_preview  # type: ignore


def main() -> int:
    root = os.path.expanduser(os.getenv("ROOT", "~/PycharmProjects/Bot_BTC"))
    ohlc = f"{root}/reports/ohlc_4h/BTC-USD.csv"
    ords = f"{root}/reports/mini_accum/exec/orders_preview.csv"

    # --- ohlc_4h ---
    df = pd.read_csv(ohlc)
    validate_ohlc_4h(df)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    print(f"[QA] ohlc_4h: rows={len(df)}  first={ts.min()}  last={ts.max()}  dups={ts.duplicated().sum()}")

    # --- orders_preview ---
    dfo = pd.read_csv(ords, parse_dates=["ts"])
    validate_orders_preview(dfo)
    dups_o = dfo["ts"].duplicated().sum()
    print(f"[QA] orders_preview: rows={len(dfo)}  dups_ts={dups_o}  unique_decisions={sorted(dfo['decision'].unique())}")

    print("[QA] schemas OK ✅")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[QA] ERROR: {e}", file=sys.stderr)
        raise