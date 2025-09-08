def _prepare_macro_d1(d1, cfg=None):
    """
    Prepara D1 (EMA200 y macro_ok) y deja el índice como int YYYYMMDD
    para que el merge `left_on='d_date', right_index=True` funcione sin
    conflictos de tipo.
    
    Acepta:
      - d1: DataFrame OHLC diario
      - d1: dict (cfg YAML) -> auto-carga D1 desde cfg['data']
    """
    import pandas as pd
    if isinstance(d1, dict) and cfg is None:
        cfg, d1 = d1, None
    if d1 is None:
        if not isinstance(cfg, dict):
            raise ValueError("_prepare_macro_d1: necesito cfg para auto-cargar D1")
        from .io import load_ohlc
        d1 = load_ohlc(
            cfg['data']['ohlc_d1_csv'],
            cfg['data'].get('ts_col', 'timestamp'),
            cfg['data'].get('tz_input', 'UTC'),
        )

    df = _ensure_cols(d1, name='d1').sort_values('timestamp').copy()
    if 'close' not in df.columns:
        raise ValueError("_prepare_macro_d1: falta columna 'close' en D1.")
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['macro_ok'] = df['close'] > df['ema200']

    out = df[['timestamp', 'close', 'ema200', 'macro_ok']].copy()
    # Clave uniforme de merge -> int YYYYMMDD y la ponemos como índice
    out['d_date'] = pd.to_datetime(out['timestamp'], utc=True).dt.strftime('%Y%m%d').astype(int)
    out = out.set_index('d_date')
    return out[['close', 'ema200', 'macro_ok']]

# --- later in the file, in run_backtest function ---

# find line:
# d1_macro = _prepare_macro_d1(d1)
# replace with:
# d1_macro = _prepare_macro_d1(d1, cfg)
