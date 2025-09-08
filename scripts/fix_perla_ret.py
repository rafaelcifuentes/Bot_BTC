#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/fix_perla_ret.py  —  v6 (prefer_default_exposure + fallback si E plana)

Objetivo: crear/rehacer 'retP_btc' para Perla sin tocar su “hilo” semanal.
- Construye E (exposición) desde múltiples columnas o usa --default_exposure.
- Si --prefer_default_exposure está activo, usa el default por encima de columnas.
- Si la exposición elegida queda plana (std≈0) y hay --default_exposure, se hace fallback al default.
- Obtiene retornos spot desde --spot o archivos candidatos.
- Alinea spot a la malla de Perla con modo 'nearest' y tolerancia (por defecto 3h, case-insensitive).
- retP_btc = E × r_spot_alineado
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

CANDIDATE_FILES = [
    "signals/diamante.csv",
    "data/ohlc/4h/BTC-USD.csv",
    "data/ohlc/BTC-USD_4h.csv",
    "data/ohlc/BTCUSD_4h.csv",
    "data/btc_4h.csv",
]

def load_4h(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()]
    df = df.sort_index()
    return df.resample("4h").last().ffill()

def spot_returns_from_df(df: pd.DataFrame) -> pd.Series | None:
    # 1) precio -> pct_change
    for col in ("close","price","Close"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            r = s.pct_change().fillna(0.0)
            if r.std(skipna=True) > 0:
                return r
    # 2) retornos ya calculados
    for col in ("ret_btc","ret","retD_btc","r","returns"):
        if col in df.columns:
            r = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            if r.std(skipna=True) > 0:
                return r
    return None

def find_spot_returns(perla_index: pd.DatetimeIndex, forced_spot: str | None) -> tuple[pd.Series | None, str]:
    # a) forzado
    if forced_spot:
        p = Path(forced_spot)
        if p.exists():
            df = load_4h(p)
            r = spot_returns_from_df(df)
            if r is not None:
                return r.reindex(perla_index).ffill().fillna(0.0), str(p)
    # b) candidatos
    for path in CANDIDATE_FILES:
        p = Path(path)
        if not p.exists():
            continue
        df = load_4h(p)
        r = spot_returns_from_df(df)
        if r is not None:
            return r.reindex(perla_index).ffill().fillna(0.0), str(p)
    return None, ""

def map_label_to_exposure(series: pd.Series) -> pd.Series:
    mapping = {
        "long": 1, "buy": 1, "bull": 1, "l": 1, "1": 1,
        "short": -1, "sell": -1, "bear": -1, "s": -1, "-1": -1,
        "flat": 0, "neutral": 0, "none": 0, "0": 0,
        # aceptar numéricos como string
        "0.0": 0, "1.0": 1, "-1.0": -1,
    }
    out = []
    for v in series.astype(str).fillna(""):
        out.append(mapping.get(v.strip().lower(), 0))
    return pd.Series(out, index=series.index, dtype=float)

def binarize_signal(series: pd.Series, thr: float = 0.5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return (s > thr).astype(float).fillna(0.0)

def build_exposure_auto(dfp: pd.DataFrame) -> tuple[pd.Series | None, str]:
    """Devuelve (E, fuente) o (None, '') si no encuentra."""
    for col in ("exposure","E","position","pos"):
        if col in dfp.columns:
            return pd.to_numeric(dfp[col], errors="coerce").clip(-1,1).fillna(0.0), col
    if "sP" in dfp.columns:
        sP = pd.to_numeric(dfp["sP"], errors="coerce")
        src = "sP"
        E = (sP if sP.abs().max() <= 1.0 else np.sign(sP)).fillna(0.0)
        return E, src
    if "w_perla_raw" in dfp.columns:
        w = pd.to_numeric(dfp["w_perla_raw"], errors="coerce")
        src = "w_perla_raw"
        if (w.min() >= 0) and (w.max() <= 1):
            E = (w > 0.5).astype(float).fillna(0.0)
        else:
            E = w.clip(-1,1).fillna(0.0)
        return E, src
    if "label" in dfp.columns:
        return map_label_to_exposure(dfp["label"]), "label"
    if "signal" in dfp.columns:
        return binarize_signal(dfp["signal"]), "signal"
    return None, ""

df_t["ts"] = pd.to_datetime(df_t["ts"], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("4h")
df_s["ts"] = pd.to_datetime(df_s["ts"], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("4h")

out = pd.merge_asof(df_t, df_s, on="ts", direction="nearest", tolerance=tol)["val"]
# ... y en el otro camino ...
out = pd.merge_asof(df_t, df_s, on="ts", direction="backward", tolerance=tol)["val"]

def align_to_index(series: pd.Series, target_index: pd.DatetimeIndex, mode: str, tol_str: str) -> pd.Series:
    tol = pd.Timedelta(str(tol_str).lower())  # evita FutureWarning por 'H'
    s = series.sort_index()
    t = pd.DatetimeIndex(target_index).tz_convert("UTC")
    if mode == "nearest":
        try:
            out = s.reindex(t, method="nearest", tolerance=tol)
        except Exception:
            # fallback: merge_asof nearest
            df_s = s.rename("val").to_frame()
            df_s["ts"] = df_s.index
            df_t = pd.DataFrame({"ts": t})
            out = pd.merge_asof(df_t, df_s, on="ts", direction="nearest", tolerance=tol)["val"]
            out.index = t
    elif mode == "asof_prev":
        df_s = s.rename("val").to_frame()
        df_s["ts"] = df_s.index
        df_t = pd.DataFrame({"ts": t})
        out = pd.merge_asof(df_t, df_s, on="ts", direction="backward", tolerance=tol)["val"]
        out.index = t
    else:  # ffill
        out = s.reindex(t).ffill()
    return pd.to_numeric(out, errors="coerce")

def main():
    ap = argparse.ArgumentParser(description="Repara/crea retP_btc para Perla alineando spot con tolerancia.")
    ap.add_argument("--perla", type=str, default="signals/perla.csv", help="CSV de Perla (entrada)")
    ap.add_argument("--out",   type=str, default=None, help="CSV de salida (si se omite, sobreescribe --perla)")
    ap.add_argument("--spot",  type=str, default=None, help="CSV con precio o retorno spot (timestamp + close/price o ret*)")
    ap.add_argument("--default_exposure", type=float, default=None, help="Exposición por defecto si no hay columnas reconocibles (ej. 1, 0, -1)")
    ap.add_argument("--prefer_default_exposure", action="store_true",
                    help="Si está activo, usa --default_exposure por encima de cualquier columna encontrada.")
    ap.add_argument("--align", type=str, default="nearest", choices=["nearest","asof_prev","ffill"], help="Método de alineación de spot a Perla")
    ap.add_argument("--tolerance", type=str, default="3h", help="Tolerancia para nearest/asof_prev (ej. 2h, 30min, 1d). Case-insensitive.")
    args = ap.parse_args()

    perla_path = Path(args.perla)
    if not perla_path.exists():
        raise SystemExit(f"[ERROR] No existe {perla_path}")

    out_path = Path(args.out) if args.out else perla_path

    dfp = load_4h(perla_path)
    idx = dfp.index

    # 1) Si ya hay retP_btc con varianza -> no tocar
    if "retP_btc" in dfp.columns and float(dfp["retP_btc"].std(skipna=True)) > 0:
        print("[OK] retP_btc ya existe y tiene varianza. No se modifica.")
        print(f"[INFO] std(retP_btc) = {dfp['retP_btc'].std(skipna=True):.6f}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dfp.to_csv(out_path, date_format="%Y-%m-%d %H:%M:%S%z")
        print(f"[SAVED] {out_path} (sin cambios en retP_btc)")
        return

    # 2) ¿precio interno?
    for col in ("close","price","Close"):
        if col in dfp.columns:
            s = pd.to_numeric(dfp[col], errors="coerce")
            dfp["retP_btc"] = s.pct_change().fillna(0.0)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            dfp.to_csv(out_path, date_format="%Y-%m-%d %H:%M:%S%z")
            print("[FIX] retP_btc derivado del precio interno de perla.csv")
            print(f"[SAVED] {out_path}  std(retP_btc)={dfp['retP_btc'].std(skipna=True):.6f}")
            return

    # 3) Construcción E
    E_auto, src = build_exposure_auto(dfp)

    use_default = False
    if args.prefer_default_exposure and args.default_exposure is not None:
        use_default = True
        print("[INFO] prefer_default_exposure=ON → usar default_exposure ignorando columnas.")
    elif E_auto is None:
        if args.default_exposure is not None:
            use_default = True
            print("[WARN] No se encontraron columnas de exposición; uso default_exposure.")
        else:
            print("[WARN] No hay columnas de exposición reconocibles y no se pasó --default_exposure; E=0.")
            E_auto = pd.Series(0.0, index=idx)
    else:
        # diagnóstico de la E auto
        uniq = np.unique(E_auto.dropna().values)
        print(f"[INFO] Exposición detectada desde '{src}': std={E_auto.std(skipna=True):.6f}  min={E_auto.min():.2f}  max={E_auto.max():.2f}  uniques={uniq[:8]}{'...' if len(uniq)>8 else ''}")

        # si está plana y tenemos default → fallback
        if E_auto.std(skipna=True) == 0.0 and args.default_exposure is not None:
            print("[WARN] Exposición detectada está plana (std=0). Fallback a default_exposure.")
            use_default = True

    if use_default:
        E = pd.Series(float(args.default_exposure), index=idx)
        print(f"[INFO] Usando default_exposure={args.default_exposure}")
    else:
        E = E_auto

    # 4) Spot
    r_spot_raw, src_spot = find_spot_returns(idx, args.spot)
    if r_spot_raw is None:
        print("[ERROR] No encontré serie de precio/retorno spot.")
        print("       Usa --spot RUTA (con 'timestamp' y 'close' o algún 'ret*') o añade un archivo candidato.")
        return
    print(f"[INFO] Spot fuente: {src_spot or '(derivado de candidatos)'}")
    print(f"[INFO] Rango Perla: {idx.min()} → {idx.max()}  |  N={len(idx)}")
    print(f"[INFO] std(r_spot_raw)={r_spot_raw.std(skipna=True):.6f}")

    # 5) Alineación
    r_spot_al = align_to_index(r_spot_raw, idx, mode=args.align, tol_str=str(args.tolerance)).fillna(0.0)
    overlap = r_spot_al.replace(0.0, np.nan).count()
    print(f"[INFO] std(r_spot_alineado)={r_spot_al.std(skipna=True):.6f}  |  puntos_no_cero={overlap}")

    # 6) retP_btc
    dfp["retP_btc"] = (E * r_spot_al).fillna(0.0)

    # Diagnóstico final
    std_final = float(dfp["retP_btc"].std(skipna=True))
    print(f"[INFO] std(retP_btc) final = {std_final:.6f}")
    if std_final == 0.0 and float(E.std(skipna=True)) == 0.0:
        print("[HINT] La exposición E sigue plana. Si quieres forzar E=1, usa: --prefer_default_exposure --default_exposure 1")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dfp.to_csv(out_path, date_format="%Y-%m-%d %H:%M:%S%z")
    print("[FIX] retP_btc = E × r_spot_alineado")
    print(f"[SAVED] {out_path}  std(retP_btc)={dfp['retP_btc'].std(skipna=True):.6f}")

if __name__ == "__main__":
    main()