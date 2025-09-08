#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, math
from pathlib import Path
import pandas as pd, numpy as np

# ---------- util de tiempo ----------
def _parse_any_datetime(s: pd.Series) -> pd.DatetimeIndex:
    # Reemplaza en _parse_any_datetime:
    # s = pd.to_numeric(s, errors="ignore")

    s_num = pd.to_numeric(s, errors="coerce")
    # Si la mayoría son números, usamos s_num; si no, dejamos s tal cual (strings/fechas)
    if pd.isna(s_num).mean() < 0.2:
        s = s_num
    if np.issubdtype(s.dtype, np.number):
        med = np.nanmedian(s.values.astype(float))
        # Heurística de unidades
        if med > 1e13:   # ns
            dt = pd.to_datetime(s, unit="ns", utc=True, errors="coerce")
        elif med > 1e12: # us
            dt = pd.to_datetime(s, unit="us", utc=True, errors="coerce")
        elif med > 1e10: # ms
            dt = pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
        elif med > 1e9:  # s (muy grandes)
            dt = pd.to_datetime(s, unit="s", utc=True, errors="coerce")
        else:
            # números pequeños → probablemente índices 0..N → inválido
            dt = pd.to_datetime(s, errors="coerce", utc=True)
    else:
        # string/obj → parse normal
        dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt

def _ensure_utc_index_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    # candidatos típicos
    ts_candidates = [
        "timestamp","time","datetime","date","open_time","close_time"
    ]
    use_col = None
    for key in ts_candidates:
        if key in cols:
            use_col = cols[key]; break
    if use_col is None:
        # fallback: primer columna si parece temporal
        first = df.columns[0]
        dt = _parse_any_datetime(df[first])
        if dt.notna().sum() >= max(10, int(0.2*len(dt))):
            use_col = first
        else:
            raise ValueError("No encuentro columna temporal (timestamp/date/...); "
                             "por favor incluye una columna de tiempo.")
    dt = _parse_any_datetime(df[use_col])
    if dt.isna().all():
        raise ValueError(f"No pude parsear la columna temporal '{use_col}'.")
    df = df.drop(columns=[use_col])
    df.index = dt
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

def _norm_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    # map case-insensitive y alias comunes
    norm = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in norm: return norm[n]
        return None
    o = pick("open","o")
    h = pick("high","h")
    l = pick("low","l")
    c = pick("close","c","price")
    missing = [name for name,var in [("open",o),("high",h),("low",l),("close",c)] if var is None]
    if missing:
        raise ValueError(f"Faltan columnas OHLC: {missing}. Columnas vistas: {list(df.columns)}")
    return df.rename(columns={o:"open",h:"high",l:"low",c:"close"})[["open","high","low","close"]]

def _maybe_resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    # si la frecuencia mediana es < 4h, agregamos a 4h
    if len(df.index) < 3:
        return df
    deltas = np.diff(df.index.view("i8"))  # ns
    med_ns = np.median(deltas)
    four_h_ns = 4*60*60*1_000_000_000
    if med_ns < four_h_ns - 1_000_000:  # claramente más fino que 4h
        # usamos OHLCV de cierre: last; high=max; low=min; open=first
        out = pd.DataFrame(index=pd.date_range(df.index[0], df.index[-1], freq="4h", tz="UTC"))
        gb = df.resample("4h")
        out["open"]  = gb["open"].first()
        out["high"]  = gb["high"].max()
        out["low"]   = gb["low"].min()
        out["close"] = gb["close"].last()
        out = out.dropna(how="all").ffill()
        return out
    # si ya está en 4h o más grueso, alineamos a rejilla 4h y ffill
    return df.resample("4h").last().ffill()

# ---------- métricas ----------
def ann_vol(r, bars_per_year=6*365):
    return float(r.ewm(span=60, adjust=False).std().median()*math.sqrt(bars_per_year))

def max_dd(r):
    eq=(1+r.fillna(0)).cumprod(); peak=eq.cummax(); dd=eq/peak-1
    return float(dd.min()) if len(dd) else float("nan")

def profit_factor(r):
    g=r[r>0].sum(); l=-r[r<0].sum()
    return float(g/l) if l>0 else (float("inf") if g>0 else 0.0)

def kpis(r):
    return dict(vol=ann_vol(r), mdd=max_dd(r), pf=profit_factor(r),
                wr=float((r>0).mean()), net=float((1+r).prod()-1))

# ---------- señales ----------
def donchian_channels(df, up, dn):
    h,l,c = df["high"], df["low"], df["close"]
    ch_up = h.rolling(int(up), min_periods=int(up)).max()
    ch_dn = l.rolling(int(dn), min_periods=int(dn)).min()
    return ch_up, ch_dn, c

def simulate(df, up, dn, mode="longflat"):
    ch_up, ch_dn, c = donchian_channels(df, up, dn)
    r = c.pct_change().fillna(0.0)

    long_trig  = (c > ch_up.shift(1))
    short_trig = (c < ch_dn.shift(1))

    if mode == "longflat":
        # exposición sólo cuando hay breakout alcista
        w_raw = long_trig.shift(1).fillna(False).astype(float)  # 0/1
        sP = pd.Series(1.0, index=df.index)                     # convenio allocator
        ret = (w_raw * r).fillna(0.0)
        return ret, sP, w_raw
    elif mode == "longshort":
        pos = np.where(long_trig, 1, np.where(short_trig, -1, np.nan))
        pos = pd.Series(pos, index=df.index).shift(1).ffill().fillna(1.0).astype(float)
        sP = pos.copy()                                         # ±1
        w_raw = pd.Series(1.0, index=df.index)                  # siempre expuesto
        ret = (pos * r).fillna(0.0)
        return ret, sP, w_raw
    else:
        raise ValueError("mode debe ser 'longflat' o 'longshort'")

# ---------- pipeline ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ohlc", required=True, help="Ruta a OHLC (cualquier 1m/1h/4h).")
    ap.add_argument("--freeze_end", required=True, help="YYYY-MM-DD (corte IS/OOS UTC).")
    ap.add_argument("--mode", choices=["longflat","longshort"], default="longflat")
    ap.add_argument("--select_by", choices=["oos_net","oos_pf","oos_wr"], default="oos_net")
    ap.add_argument("--out_csv", default="reports/heart/perla_grid_oos.csv")
    ap.add_argument("--write_best_signals", action="store_true")
    args=ap.parse_args()

    Path("reports/heart").mkdir(parents=True, exist_ok=True)
    Path("signals").mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.ohlc)
    raw = _ensure_utc_index_df(raw)
    raw = _norm_ohlc_columns(raw)
    df  = _maybe_resample_4h(raw)

    # Validación freeze
    cutoff = pd.to_datetime(args.freeze_end, utc=True)
    if cutoff < df.index.min() or cutoff > df.index.max():
        raise ValueError(f"--freeze_end {args.freeze_end} fuera de rango "
                         f"[{df.index.min()} .. {df.index.max()}]")

    df_is  = df[df.index <= cutoff]
    df_oos = df[df.index  > cutoff]

    ups  = [20, 30, 40]
    dns  = [10, 15, 20]

    rows=[]
    for u in ups:
        for d in dns:
            ret_all, _, _ = simulate(df, u, d, mode=args.mode)
            ret_is  = ret_all.loc[df_is.index]
            ret_oos = ret_all.loc[df_oos.index]
            mi, mo = kpis(ret_is), kpis(ret_oos)
            rows.append(dict(
                up=u, dn=d,
                is_vol=mi["vol"], is_mdd=mi["mdd"], is_pf=mi["pf"], is_wr=mi["wr"], is_net=mi["net"],
                oos_vol=mo["vol"], oos_mdd=mo["mdd"], oos_pf=mo["pf"], oos_wr=mo["wr"], oos_net=mo["net"]
            ))
    res = pd.DataFrame(rows)
    sort_cols = {"oos_net":["oos_net"], "oos_pf":["oos_pf","oos_net"], "oos_wr":["oos_wr","oos_net"]}[args.select_by]
    res = res.sort_values(sort_cols, ascending=[False]*len(sort_cols)).reset_index(drop=True)
    res.to_csv(args.out_csv, index=False)

    print("[OK] grid →", args.out_csv)
    print(res.head(10).to_string(index=False))
    best = res.iloc[0]

    if args.write_best_signals:
        # Señales para TODO el periodo con el mejor u/d
        ret_all, sP, w_raw = simulate(df, int(best.up), int(best.dn), mode=args.mode)
        out = pd.DataFrame({
            "timestamp": df.index,
            "sP": sP.values,                 # convenio allocator: [-1,1] o 1
            "w_perla_raw": w_raw.values,     # 0/1 en longflat, 1 en longshort
            "retP_btc": df["close"].pct_change().fillna(0.0).values
        })
        out.to_csv("signals/perla.csv", index=False)
        print(f"[OK] signals/perla.csv escrito (up={int(best.up)}, dn={int(best.dn)}, mode={args.mode})")

if __name__=="__main__":
    main()