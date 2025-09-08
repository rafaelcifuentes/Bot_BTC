from mini_accum.indicators import ema
import pandas as pd

def test_ema_runs():
    s = pd.Series([1,2,3,4,5])
    out = ema(s, span=3)
    assert len(out) == 5
