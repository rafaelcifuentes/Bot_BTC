import argparse, pandas as pd, numpy as np
TS = ["timestamp","ts","time","date","datetime"]
ENTRY = ["entry_time","entry_ts","entry","enter_time","start_time","open_time"]
EXIT  = ["exit_time","exit_ts","exit","close_time","end_time","stop_time","target_time"]
SIDE  = ["side","direction","pos","position","signal"]
def pick(df, cands, name):
    for c in cands:
        if c in df.columns: return c
    raise ValueError(f"Falta columna {name} en {list(df.columns)}")
def to_utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce")
def side_to_num(x):
    if pd.isna(x): return 0
    xs = str(x).strip().lower()
    if xs in ["long","buy","1","true","l"]: return 1
    if xs in ["short","sell","-1","s"]:    return -1
    try: return int(float(xs))
    except: return 0
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades_csv", required=True)
    ap.add_argument("--ohlc4h_csv", required=True)  # reports/ohlc_4h/BTC-USD.csv
    ap.add_argument("--out_csv", required=True)     # reports/diamante_btc_costes_week1_bars.csv
    args = ap.parse_args()
    o = pd.read_csv(args.ohlc4h_csv)
    ts_o = next(c for c in TS if c in o.columns)
    o[ts_o] = to_utc(o[ts_o]); o = o.dropna(subset=[ts_o]).sort_values(ts_o).rename(columns={ts_o:"timestamp"})
    if "close" not in o.columns: raise ValueError("OHLC 4h debe tener 'close'")
    o["ret_4h_raw"] = o["close"].pct_change()
    t = pd.read_csv(args.trades_csv)
    ent = pick(t, ENTRY, "entry_time"); ex = pick(t, EXIT, "exit_time"); sd = pick(t, SIDE, "side")
    t[ent] = to_utc(t[ent]); t[ex] = to_utc(t[ex])
    t = t.dropna(subset=[ent,ex]).sort_values(ent)
    t["sd"] = t[sd].map(side_to_num).clip(-1,1)
    # línea de tiempo 4h
    pos = pd.Series(0, index=o["timestamp"])
    for _,row in t.iterrows():
        mask = (o["timestamp"]>=row[ent]) & (o["timestamp"]<=row[ex])
        if row["sd"]==0: continue
        pos.loc[mask] = row["sd"]  # si hay solapamiento, última gana (simple y conservador)
    # SHIFT(1) para evitar lookahead
    ret_4h = o["ret_4h_raw"] * pos.shift(1).fillna(0)
    out = pd.DataFrame({"timestamp": o["timestamp"], "ret_4h": ret_4h}).dropna()
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Escrito {args.out_csv} con {len(out)} filas")
if __name__ == "__main__":
    main()
