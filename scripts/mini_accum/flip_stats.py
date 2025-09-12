#!/usr/bin/env python3
import sys
import pandas as pd

if len(sys.argv) < 2:
    print("usage: flip_stats.py <flips_csv>", file=sys.stderr)
    sys.exit(1)

f = sys.argv[1]
df = pd.read_csv(f, parse_dates=["ts"])
dhrs = df["ts"].diff().dt.total_seconds().div(3600).iloc[1:]
wk = df["ts"].dt.isocalendar().week.value_counts().sort_index()
print(f"{f}\tmin4h={float((dhrs/4).min()):.1f}\tweeks>1={int((wk>1).sum())}")
