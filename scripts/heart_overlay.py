import argparse, pandas as pd, numpy as np, yaml

TS_CANDIDATES = ["timestamp","ts","date","datetime","time"]

def pick_col(cols, cands, name):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low: return low[c]
    raise ValueError(f"Falta columna '{name}' en el CSV (tiene {list(cols)})")

def parse_ts(s):
    if pd.api.types.is_numeric_dtype(s):
        arr = s.astype("int64")
        unit = "ms" if arr.max() > 10**12 else "s"
        return pd.to_datetime(arr, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")

def hysteresis_labels(raw_state, min_hold=3, cooloff=2):
    state = raw_state.copy()
    last = state.iloc[0]
    hold = 0
    for i in range(1, len(state)):
        cur = state.iloc[i]
        if cur != last:
            if hold < min_hold:
                state.iloc[i] = last
                hold += 1
            else:
                state.iloc[i] = cur
                last = cur
                hold = -cooloff
        else:
            if hold < 0:
                hold += 1
            else:
                hold += 1
    return state

def compute_weights(df, rules):
    up_trend = (df['ema50'] > df['ema200'])
    adx_ok   = (df['adx1d'] >= rules['regime']['adx1d_min']) & (df['adx4h'] >= rules['regime']['adx4_min'])
    fg_bias  = (df['fear_greed_z'] >= rules['regime']['fg_long_min'])
    fund_bias= (df['funding'] >= rules['regime']['funding_bias'])
    raw_state = np.select(
        [(adx_ok & up_trend & fg_bias & fund_bias),
         (adx_ok & up_trend) | fg_bias | fund_bias],
        ['strong', 'neutral'], default='counter'
    )
    st = pd.Series(raw_state, index=df.index)
    st = hysteresis_labels(st, rules['hysteresis']['min_steps_hold'], rules['hysteresis']['cooloff_steps'])
    w_map = {'strong': rules['weights']['w_strong'],
             'neutral': rules['weights']['w_neutral'],
             'counter': rules['weights']['w_counter']}
    w = st.map(w_map).clip(lower=rules['safety']['w_floor_on_signal'], upper=rules['safety']['w_cap'])
    return st, w

def load_prices(prices_csv):
    p = pd.read_csv(prices_csv)
    ts = pick_col(p.columns, TS_CANDIDATES, "timestamp (prices)")
    p[ts] = parse_ts(p[ts]); p = p.dropna(subset=[ts]).sort_values(ts).set_index(ts)
    if "close" not in p.columns: raise ValueError("prices_csv debe tener 'close'")
    if "adx1d" not in p.columns or "adx4h" not in p.columns:
        raise ValueError("prices_csv debe incluir 'adx1d' y 'adx4h' (usa scripts/build_ohlc_4h.py)")
    return p

def load_fg(fg_csv):
    fg = pd.read_csv(fg_csv)
    ts = pick_col(fg.columns, TS_CANDIDATES, "timestamp (FG)")
    fg[ts] = parse_ts(fg[ts]); fg = fg.dropna(subset=[ts]).sort_values(ts).set_index(ts)
    # aceptar varias columnas de valor (incluye 'fgi')
    cand = None
    for c in ["fear_greed_z","fg_z","z","fg","value","score","index","fgi"]:
        if c in fg.columns: cand = c; break
    if cand is None:
        raise ValueError(f"No se encontró columna de valor de Fear&Greed en {list(fg.columns)}")
    s = pd.to_numeric(fg[cand], errors="coerce")
    # si parece 0..100 → normaliza a [-1,1]
    if s.max() > 1.5 or s.min() < -1.5:
        fg['fear_greed_z'] = ((s - 50.0) / 50.0).clip(-1, 1)
    else:
        fg['fear_greed_z'] = s.clip(-1, 1)
    return fg[['fear_greed_z']]

def load_funding(fu_csv):
    fu = pd.read_csv(fu_csv)
    ts = pick_col(fu.columns, TS_CANDIDATES, "timestamp (Funding)")
    fu[ts] = parse_ts(fu[ts]); fu = fu.dropna(subset=[ts]).sort_values(ts).set_index(ts)
    cand = None
    for c in ["funding","funding_rate","rate","fundingRate","fr"]:
        if c in fu.columns: cand = c; break
    if cand is None:
        raise ValueError(f"No se encontró columna de funding en {list(fu.columns)}")
    fu['funding'] = pd.to_numeric(fu[cand], errors="coerce")
    return fu[['funding']]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prices_csv', required=True)
    ap.add_argument('--fg_csv', required=True)
    ap.add_argument('--funding_csv', required=True)
    ap.add_argument('--rules_yaml', required=True)
    ap.add_argument('--out_csv', required=True)
    args = ap.parse_args()

    with open(args.rules_yaml) as f:
        rules = yaml.safe_load(f)

    p  = load_prices(args.prices_csv)
    fg = load_fg(args.fg_csv)
    fu = load_funding(args.funding_csv)

    df = p.copy()
    df['ema50']  = df['close'].ewm(span=rules['regime']['ema_fast'], adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=rules['regime']['ema_slow'], adjust=False).mean()
    df = df.join(fg, how='left').join(fu, how='left').ffill()

    # SHIFT(1) estricto
    for col in ['ema50','ema200','adx1d','adx4h','fear_greed_z','funding']:
        if col in df.columns:
            df[col] = df[col].shift(1)

    st, w = compute_weights(df, rules)
    out = pd.DataFrame({'timestamp': df.index, 'state': st, 'w_diamante': w}).dropna()
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Pesos escritos en {args.out_csv}  (rows={len(out)})")

if __name__ == "__main__":
    main()
