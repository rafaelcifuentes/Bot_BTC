#!/usr/bin/env python3
import os, sys, glob, json, os.path as _p
from datetime import datetime, timezone

REGISTRY = "registry_runs.csv"

def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

paths = sorted(
    glob.glob("reports/summary_rf_sentiment_EXP_*.json"),
    key=os.path.getmtime, reverse=True
)

if len(paths) < 2:
    sys.stderr.write("⚠️  No hay suficientes summaries de SentimentEXP en reports/\n")
    sys.exit(1)

live, freeze = paths[0], paths[1]

def wf(p):
    base = os.path.basename(p)
    return _p.join("reports", base.replace("summary_", "walkforward_").replace(".json",".csv"))

os.makedirs("reports", exist_ok=True)
# nos aseguramos que exista el registry
open(REGISTRY, "a").close()

with open(REGISTRY, "a") as f:
    f.write(f"{now()},SentimentEXP,freeze,{freeze},{wf(freeze)}\n")
    f.write(f"{now()},SentimentEXP,live,  {live},{wf(live)}\n")

print("✅ Registradas SentimentEXP (freeze & live) en registry_runs.csv")
