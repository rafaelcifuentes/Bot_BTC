#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

RET_BASE_PREF = ["ret_4h", "ret", "ret_b", "ret_base"]
RET_OVER_PREF = ["ret_4h_overlay", "ret_overlay", "ret_o", "ret_net", "ret"]

TS_CANDIDATES = ("timestamp", "ts", "time", "datetime", "date", "dt")


def find_ts(df: pd.DataFrame, preferred: str | None = None) -> str | None:
    """Encuentra la columna timestamp en df."""
    if preferred and preferred in df.columns:
        return preferred
    for c in df.columns:
        if c.lower() in TS_CANDIDATES:
            return c
    return None


def pick_col(df: pd.DataFrame, prefs: list[str], ts: str | None) -> str:
    """Elige columna de retorno preferida; si no, la primera que no sea ts."""
    cols = [c for c in prefs if c in df.columns]
    if cols:
        return cols[0]
    for c in df.columns:
        if ts is None or c != ts:
            return c
    raise ValueError("No encuentro columna de retorno")


def kpis_from_ret(ret: pd.Series) -> dict:
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    if ret.empty:
        return dict(pf=np.nan, wr=np.nan, mdd=np.nan, vol=np.nan, net=np.nan, rows=0)
    gains = ret[ret > 0].sum()
    losses = -ret[ret < 0].sum()
    pf = float(gains / losses) if losses > 0 else np.inf
    wr = float((ret > 0).mean() * 100.0)
    eq = (1.0 + ret).cumprod()
    dd = eq / eq.cummax() - 1.0
    return dict(
        pf=pf,
        wr=wr,
        mdd=float(dd.min()),  # negativo (drawdown)
        vol=float(ret.std()),
        net=float(ret.sum()),
        rows=int(len(ret)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", required=True)
    ap.add_argument("--overlay_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--ts_col", default="timestamp")
    args = ap.parse_args()

    # 1) Leer
    b = pd.read_csv(args.baseline_csv)
    o = pd.read_csv(args.overlay_csv)

    # 2) Detectar columnas de tiempo
    tsb = find_ts(b, args.ts_col)
    tso = find_ts(o, args.ts_col)

    # 3) Normalizar timestamps → UTC-aware → naive + floor 4h (¡clave!)
    if tsb:
        b[tsb] = pd.to_datetime(b[tsb], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("4h")
        b = b.dropna(subset=[tsb]).sort_values(tsb)
    if tso:
        o[tso] = pd.to_datetime(o[tso], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("4h")
        o = o.dropna(subset=[tso]).sort_values(tso)

    # 4) Elegir columnas de retorno
    rb = pick_col(b, RET_BASE_PREF, tsb)
    ro = pick_col(o, RET_OVER_PREF, tso)

    # 5) Emparejar baseline/overlay
    if tsb and tso:
        bb = b[[tsb, rb]].rename(columns={tsb: "ts", rb: "ret_b"})
        oo = o[[tso, ro]].rename(columns={tso: "ts", ro: "ret_o"})
        m = pd.merge_asof(
            bb.sort_values("ts"),
            oo.sort_values("ts"),
            on="ts",
            direction="nearest",
            tolerance=pd.Timedelta("2min"),
        ).dropna(subset=["ret_b", "ret_o"])
        ret_b = m["ret_b"]
        ret_o = m["ret_o"]
    else:
        # Sin timestamps: recortar a longitudes iguales
        ret_b = pd.to_numeric(b[rb], errors="coerce").dropna()
        ret_o = pd.to_numeric(o[ro], errors="coerce").dropna()
        n = min(len(ret_b), len(ret_o))
        ret_b, ret_o = ret_b.iloc[:n], ret_o.iloc[:n]

    # 6) KPIs y salida
    kb, ko = kpis_from_ret(ret_b), kpis_from_ret(ret_o)
    out = pd.DataFrame(
        [
            {
                "pf_base": kb["pf"],
                "pf_overlay": ko["pf"],
                "wr_base": kb["wr"],
                "wr_overlay": ko["wr"],
                "mdd_base": kb["mdd"],
                "mdd_overlay": ko["mdd"],
                "vol_base": kb["vol"],
                "vol_overlay": ko["vol"],
                "net_base": kb["net"],
                "net_overlay": ko["net"],
                "rows": min(kb["rows"], ko["rows"]),
            }
        ]
    )
    out.to_csv(args.out_csv, index=False)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# Corazón vs Baseline — KPIs\n\n")
        f.write(out.to_markdown(index=False))

    print(f"[OK] KPIs → {args.out_csv}")
    print(f"[OK] Resumen → {args.out_md}")


if __name__ == "__main__":
    main()