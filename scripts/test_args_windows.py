#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import Any, Iterator, Tuple
import pandas as pd

from tzsafe_window import parse_windows_arg  # tu parseador

# ---------- helpers de zona horaria ----------
def _to_utc_ts(x: Any) -> pd.Timestamp | None:
    if x is None:
        return None
    ts = x if isinstance(x, pd.Timestamp) else pd.Timestamp(x)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")     # naive -> UTC
    else:
        ts = ts.tz_convert("UTC")      # aware -> UTC
    return ts

# ---------- iterador robusto de ventanas ----------
def _iter_windows_generic(windows: Any) -> Iterator[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """
    Normaliza a tuplas (label, start_utc, end_utc).
    Soporta:
      - dict: {label: (start, end)}
      - lista/tupla de (label, start, end)
      - lista/tupla de objetos con .label/.start/.end (p.ej. dataclass Window)
      - un solo objeto con .label/.start/.end
    """
    if isinstance(windows, dict):
        for label, se in windows.items():
            if not isinstance(se, (tuple, list)) or len(se) != 2:
                raise ValueError(f"Valor inválido para '{label}': {se!r}")
            start, end = se
            yield str(label), _to_utc_ts(start), _to_utc_ts(end)
        return

    if isinstance(windows, (list, tuple)):
        for item in windows:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                label, start, end = item
                yield str(label), _to_utc_ts(start), _to_utc_ts(end)
                continue
            if all(hasattr(item, a) for a in ("label", "start", "end")):
                yield str(item.label), _to_utc_ts(item.start), _to_utc_ts(item.end)
                continue
            if isinstance(item, dict) and {"label", "start", "end"} <= item.keys():
                yield str(item["label"]), _to_utc_ts(item["start"]), _to_utc_ts(item["end"])
                continue
            raise ValueError(f"Item inesperado en lista: {type(item)} - {item!r}")
        return

    if all(hasattr(windows, a) for a in ("label", "start", "end")):
        yield str(windows.label), _to_utc_ts(windows.start), _to_utc_ts(windows.end)
        return

    raise TypeError(f"Estructura de 'windows' no soportada: {type(windows)} - {windows!r}")

# ---------- CLI ----------
def main() -> None:
    ap = argparse.ArgumentParser(description="Probar parseo y normalización de --windows")
    ap.add_argument(
        "--windows",
        nargs="+",
        required=True,
        help='Ej.:  --windows "2023Q4:2023-10-01:2023-12-31" "2024H1:2024-01-01:2024-06-30"',
    )
    args = ap.parse_args()

    windows_parsed = parse_windows_arg(args.windows)

    print("Tipo devuelto por parse_windows_arg:", type(windows_parsed).__name__)
    if isinstance(windows_parsed, dict):
        print("Keys:", list(windows_parsed.keys()))
    elif isinstance(windows_parsed, (list, tuple)) and len(windows_parsed) > 0:
        print("Primer item:", type(windows_parsed[0]).__name__, "-", repr(windows_parsed[0])[:120], "...")

    print("\n=== Ventanas normalizadas (tz-aware, UTC) ===")
    n = 0
    for label, start_ts, end_ts in _iter_windows_generic(windows_parsed):
        n += 1
        if end_ts is not None and start_ts is not None and end_ts < start_ts:
            raise ValueError(f"Rango inválido en '{label}': end < start ({end_ts} < {start_ts})")
        print(f"{label}: {start_ts} -> {end_ts}")
    if n == 0:
        print("(sin ventanas)")

if __name__ == "__main__":
    main()