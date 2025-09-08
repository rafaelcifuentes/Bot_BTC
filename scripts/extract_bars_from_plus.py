import argparse, pandas as pd, numpy as np

TS = ["timestamp","ts","time","date","datetime"]
RET = ["ret_4h","ret","return","r","pnl","pnl_4h","ret_net","ret_cost","ret_diamante"]
EQ  = ["equity","eq","balance","bal","curve","equity_curve","capital"]
PX  = ["close","price","px","close_4h","close_price"]
POS = ["position","pos","side","signal","dir","direction"]

def pick_ts(cols):
    for c in TS:
        if c in cols: return c
    raise ValueError(f"No encuentro columna de tiempo en {list(cols)}")

def extract_ret(df):
    # 1) retornos explícitos
    for c in RET:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.dropna().size>5:
                return s, f"returns:{c}"
    # 2) equity -> pct_change
    for c in EQ:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").pct_change()
            if s.dropna().size>5:
                return s, f"equity_pct:{c}"
    # 3) close + position -> pct_change * pos.shift(1)
    px = None; pos = None
    for c in PX:
        if c in df.columns:
            px = pd.to_numeric(df[c], errors="coerce"); break
    for c in POS:
        if c in df.columns:
            pos = pd.to_numeric(df[c], errors="coerce"); break
    if px is not None and pos is not None:
        s = px.pct_change() * pos.shift(1)
        if s.dropna().size>5:
            return s, f"px_pct*pos:{(c if pos is not None else '?')}"
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    ts = pick_ts(df.columns)
    df[ts] = pd.to_datetime(df[ts], utc=True, errors="coerce")
    df = df.dropna(subset=[ts]).sort_values(ts)

    ret, how = extract_ret(df)
    if ret is None:
        raise RuntimeError(f"No pude derivar retornos desde columnas {list(df.columns)}")

    out = pd.DataFrame({"timestamp": df[ts], "ret_4h": ret}).dropna()
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Escrito {args.out_csv} ({len(out)} filas) usando {how}")
if __name__ == "__main__":
    main()
