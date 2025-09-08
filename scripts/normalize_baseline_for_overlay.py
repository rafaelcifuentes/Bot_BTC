import argparse, pandas as pd, numpy as np

TS_CANDIDATES = ["timestamp","ts","time","date","datetime"]
RET_PREF = ["ret_overlay","ret_4h","ret","return","r","ret_net","ret_cost","ret_diamante","pnl_pct","pnl_4h_pct"]
EQUITY_CANDS = ["equity","eq","balance","bal","curve","equity_curve","capital"]
CLOSE_CANDS = ["close","price","px","close_4h"]

def pick_ts(df):
    for c in TS_CANDIDATES:
        if c in df.columns: return c
    raise ValueError(f"No encuentro columna de tiempo en {list(df.columns)}")

def pick_ret(df):
    # 1) intentamos retornos explícitos (preferidos)
    for c in RET_PREF:
        if c in df.columns:
            return c, "ret"
    # 2) si hay equity/balance → retornos por pct_change
    for c in EQUITY_CANDS:
        if c in df.columns:
            return c, "equity"
    # 3) si hay close → retornos por pct_change
    for c in CLOSE_CANDS:
        if c in df.columns:
            return c, "close"
    raise ValueError("No encuentro columna de retornos ni equity/close para derivarlos.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    ts = pick_ts(df)
    df[ts] = pd.to_datetime(df[ts], utc=True, errors="coerce")
    df = df.dropna(subset=[ts]).sort_values(ts)

    col, mode = pick_ret(df)
    s = pd.to_numeric(df[col], errors="coerce")

    if mode == "ret":
        ret = s
    elif mode in ("equity","close"):
        ret = s.pct_change()
    else:
        raise RuntimeError("Modo desconocido")

    out = pd.DataFrame({"timestamp": df[ts], "ret_4h": ret}).dropna()
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Escrito {args.out_csv} con {len(out)} filas; columnas: timestamp, ret_4h")

if __name__ == "__main__":
    main()
