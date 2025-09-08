#!/usr/bin/env python3
import argparse, os, sys, pandas as pd
p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--asset", required=True)
p.add_argument("--windows", nargs="+", required=True) # "NAME:YYYY-MM-DD:YYYY-MM-DD"
args = p.parse_args()

bad = []
for w in args.windows:
    name, s, e = w.split(":")
    f = os.path.join(args.root, name, f"{args.asset}.csv")
    if not os.path.exists(f):
        bad.append((name, "missing"))
        continue
    df = pd.read_csv(f, parse_dates=["ts"])
    if len(df)==0:
        bad.append((name, "empty"))
        continue
    if not (df["ts"].min()>=pd.Timestamp(s, tz="UTC") and df["ts"].max()<=pd.Timestamp(e, tz="UTC")+pd.Timedelta(days=1)):
        bad.append((name, f"ts out of range [{df['ts'].min()}..{df['ts'].max()}]"))
if bad:
    print("[FAIL] windows:", bad); sys.exit(1)
print("[OK] all windows healthy")