import pandas as pd, numpy as np, os

_TS = {"timestamp","ts","time","date","datetime"}
_RET = ["ret_4h","ret","return","r","pnl","pnl_4h","ret_overlay","ret_net","ret_cost","ret_diamante"]
_EQ  = ["equity","eq","balance","bal","curve","equity_curve","capital"]
_PX  = ["close","price","px","close_4h","close_price"]
_POS = ["position","pos","side","signal","dir","direction","y_pred","yhat","pred","state"]

def _pick_ts(df):
    for c in df.columns:
        if c in _TS: return c
    return None

def _try_from_returns(df):
    for c in _RET:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.dropna().size > 10:
                return s
    return None

def _try_from_equity(df):
    for c in _EQ:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").pct_change()
            if s.dropna().size > 10:
                return s
    return None

def _try_from_price_and_pos(df):
    px = None; pos = None
    for c in _PX:
        if c in df.columns:
            px = pd.to_numeric(df[c], errors="coerce"); break
    for c in _POS:
        if c in df.columns:
            pos = pd.to_numeric(df[c], errors="coerce"); break
    if px is None or pos is None: return None
    pos = np.sign(pos)  # map a -1/0/1
    return px.pct_change() * pd.Series(pos, index=df.index).shift(1)

def _best_dataframe_candidate(ns):
    best = None
    for name, obj in ns.items():
        try:
            if isinstance(obj, pd.DataFrame) and len(obj) > 50:
                ts = _pick_ts(obj)
                if not ts: continue
                df = obj.copy()
                df[ts] = pd.to_datetime(df[ts], utc=True, errors="coerce")
                df = df.dropna(subset=[ts]).set_index(ts).sort_index()
                r = _try_from_returns(df) or _try_from_equity(df) or _try_from_price_and_pos(df)
                if r is None: continue
                out = pd.DataFrame({"ret_4h": r}).dropna()
                if out.empty: continue
                if best is None or len(out) > len(best):
                    out["timestamp"] = out.index
                    best = out[["timestamp","ret_4h"]].copy()
        except Exception:
            continue
    return best

def _best_series_candidate(ns, base_ts, base_close):
    # Busca Series/numpy arrays que parezcan "posición/señal" y casen en longitud
    n = len(base_ts)
    best = None; best_score = -1
    for name, obj in ns.items():
        try:
            if isinstance(obj, pd.Series):
                arr = obj.values
            elif isinstance(obj, np.ndarray):
                arr = obj
            else:
                continue
            if arr.ndim != 1: continue
            # aceptar si longitudes iguales o “casi” (±5)
            if not (abs(len(arr) - n) <= 5): continue
            # normaliza a -1/0/1
            ser = pd.Series(arr[:n], index=base_ts)
            if ser.dtype == bool:
                pos = ser.astype(int)
            else:
                pos = np.sign(pd.to_numeric(ser, errors="coerce"))
            # score: cuántas barras con exposición (|pos|>0)
            score = int((pos.abs() > 0).sum())
            if score < 10:  # ignora señales casi todo 0
                continue
            ret = base_close.pct_change() * pos.shift(1)
            out = pd.DataFrame({"timestamp": base_ts, "ret_4h": ret}).dropna()
            if out.empty: continue
            if score > best_score:
                best = out; best_score = score
        except Exception:
            continue
    return best

def emit_bars_if_possible(ns, out_path="reports/diamante_btc_costes_week1_bars.csv",
                          ohlc_csv="reports/ohlc_4h/BTC-USD.csv"):
    if not os.path.exists(ohlc_csv):
        raise RuntimeError(f"No encuentro OHLC 4h base en {ohlc_csv} (ejecuta scripts/build_ohlc_4h.py primero).")
    o = pd.read_csv(ohlc_csv)
    ts_o = None
    for c in ["timestamp","ts","date","datetime","time"]:
        if c in o.columns:
            ts_o = c; break
    if ts_o is None: raise RuntimeError(f"OHLC 4h sin timestamp válido: {list(o.columns)}")
    o[ts_o] = pd.to_datetime(o[ts_o], utc=True, errors="coerce")
    o = o.dropna(subset=[ts_o]).sort_values(ts_o)
    if "close" not in o.columns:
        raise RuntimeError("OHLC 4h debe tener columna 'close'")
    base_ts = o[ts_o].values
    base_close = pd.Series(pd.to_numeric(o["close"], errors="coerce").values, index=o[ts_o])

    # 1) Intenta DataFrames locales
    out = _best_dataframe_candidate(ns)
    # 2) Si no, intenta Series/ndarray locales (posición/señal)
    if out is None:
        out = _best_series_candidate(ns, o[ts_o], base_close)

    if out is None or out.dropna().empty:
        raise RuntimeError("No encontré en locals() ni DataFrames ni Series/arrays compatibles para derivar retornos por barra.")

    out = out.sort_values("timestamp")
    out.to_csv(out_path, index=False)
    print(f"[OK] Barras exportadas → {out_path} (rows={len(out)})")
