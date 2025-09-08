#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from textwrap import dedent

DESC = """Convert USD/USDT returns to BTC-numeraire returns.

Formula:
  r_btcnum = (1 + r_usd) / (1 + r_btc) - 1

Input:
  - in_csv: signals CSV with a timestamp column (flexible names) and either 'ret_usd' or 'ret'.
    If it already has 'retD_btc' or 'retP_btc', the script exits without changes.
  - btc_csv: BTC spot/ohlc with a timestamp column and either 'ret_btc' or 'close'.

Output:
  - Adds 'retD_btc' (for Diamante-like files) or 'retP_btc' (for Perla-like files)
    and writes to --out_csv or overwrites in place.
"""

# ---- helpers ---------------------------------------------------------------

TS_CANDIDATES = (
    "timestamp", "ts", "time", "datetime", "date", "Date", "open_time", "close_time"
)


def pick_ts_col(df, preferred=None):
    """Return an existing timestamp-like column name from df.
    If preferred is provided and exists, use it. Otherwise try common names.
    If none found but index is datetime-like, materialize it as 'timestamp'.
    """
    if preferred and preferred in df.columns:
        return preferred
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {x.lower() for x in TS_CANDIDATES}:
            return c
    if isinstance(df.index, pd.DatetimeIndex):
        df.reset_index(inplace=True)
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
        return "timestamp"
    raise SystemExit(
        f"No encuentro columna de tiempo. Pasa --ts_col / --btc_ts_col o renombra alguna a 'timestamp'.\n"
        f"Columnas disponibles: {list(df.columns)}"
    )


def to_utc(series):
    """Robust timestamp parsing to UTC. Handles strings and epoch seconds/ms."""
    # First attempt: let pandas infer
    out = pd.to_datetime(series, utc=True, errors="coerce")
    # If too many NaN and it's numeric, try epoch seconds/ms
    if out.isna().mean() > 0.50 and pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        if vals.notna().any():
            mx = float(vals.max())
            unit = "ms" if mx > 1e12 else "s"
            out = pd.to_datetime(vals, unit=unit, utc=True, errors="coerce")
    return out


def auto_ret_name(cols):
    low = [c.lower() for c in cols]
    # If already present, return None to early-exit without touching timestamps
    if any("retd_btc" in c for c in low):
        return None
    if any("retp_btc" in c for c in low):
        return None
    if "ret_usd" in low:
        return cols[low.index("ret_usd")]
    if "ret" in low:
        return cols[low.index("ret")]
    return None


def is_perla(cols):
    low = [c.lower() for c in cols]
    keys = ("perla", "sp", "w_perla", "w_perla_raw")
    return any(any(k in c for k in keys) for c in low)


# ---- main ------------------------------------------------------------------

def main():
    epilog = dedent(
        """
        Examples
        --------
        # Diamante: have ret_usd -> create retD_btc
        python3 scripts/to_btc_numeraire.py --in_csv signals/diamante.csv --btc_csv btc_4h.csv

        # Perla: have ret (usd) -> create retP_btc
        python3 scripts/to_btc_numeraire.py --in_csv signals/perla.csv --btc_csv btc_4h.csv
        """
    )
    ap = argparse.ArgumentParser(
        description=DESC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    ap.add_argument("--in_csv", required=True, help="signals/*.csv (diamante o perla)")
    ap.add_argument("--btc_csv", required=True, help="BTC OHLC/spot CSV with 'timestamp'+'close' or 'ret_btc'")
    ap.add_argument("--out_csv", help="Output CSV path (default: overwrite --in_csv)")
    ap.add_argument("--tolerance", default="3h", help="asof merge tolerance (default: 3h)")
    ap.add_argument("--ts_col", default=None, help="Nombre de la columna timestamp en --in_csv (auto si no)")
    ap.add_argument("--btc_ts_col", default=None, help="Nombre de la columna timestamp en --btc_csv (auto si no)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # Early exit if already in BTC numeraire or no recognizable USD return column
    ret_name = auto_ret_name(df.columns)
    if ret_name is None:
        print("[INFO] ya existe ret*_btc o no hay columna de retorno en USD reconocible; no se modifica.")
        return

    spot = pd.read_csv(args.btc_csv)

    # Pick and parse timestamps
    ts_in = pick_ts_col(df, args.ts_col)
    ts_btc = pick_ts_col(spot, args.btc_ts_col)

    df[ts_in] = to_utc(df[ts_in])
    spot[ts_btc] = to_utc(spot[ts_btc])

    df.dropna(subset=[ts_in], inplace=True)
    spot.dropna(subset=[ts_btc], inplace=True)

    df.sort_values(ts_in, inplace=True)
    spot.sort_values(ts_btc, inplace=True)

    # BTC price return per period
    if "ret_btc" in spot.columns:
        rbtc = spot[[ts_btc, "ret_btc"]].copy()
        rbtc["ret_btc"] = pd.to_numeric(rbtc["ret_btc"], errors="coerce")
    else:
        if "close" not in spot.columns:
            raise SystemExit("btc_csv debe tener 'close' o 'ret_btc'")
        spot["close"] = pd.to_numeric(spot["close"], errors="coerce")
        rbtc = spot[[ts_btc]].copy()
        rbtc["ret_btc"] = spot["close"].pct_change()

    # Ensure numeric USD return
    df[ret_name] = pd.to_numeric(df[ret_name], errors="coerce")

    for df, col in ((left_df, "timestamp"), (right_df, "timestamp")):
        s = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert(None)
        df[col] = s.dt.floor("4h")  # si también trabajas en 4h aquí; si no, quita esta línea

    merged = pd.merge_asof(
        left_df.sort_values("timestamp"),
        right_df.sort_values("timestamp"),
        left_on="timestamp", right_on="timestamp",
        direction="nearest", tolerance=pd.Timedelta("2min")
    )


    merged = pd.merge_asof(
        df[[ts_in, ret_name]].sort_values(ts_in),
        rbtc.sort_values(ts_btc),
        left_on=ts_in,
        right_on=ts_btc,
        direction="backward",
        tolerance=pd.Timedelta(args.tolerance),
    )

    if merged["ret_btc"].isna().all():
        raise SystemExit("No se pudo alinear BTC spot con la serie. Revisa timestamps/tolerance/--btc_ts_col.")

    # Compute BTC-numeraire return
    r_usd = merged[ret_name]
    r_btc = merged["ret_btc"]
    r_btcnum = (1.0 + r_usd) / (1.0 + r_btc) - 1.0

    out_col = "retP_btc" if is_perla(df.columns) else "retD_btc"
    df[out_col] = r_btcnum.values

    out_path = args.out_csv or args.in_csv
    df.to_csv(out_path, index=False)
    print(f"[OK] escrito {out_path} con columna {out_col} (n={len(df)})")


if __name__ == "__main__":
    main()