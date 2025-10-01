#!/usr/bin/env python3
import sys, yaml, os

if len(sys.argv) < 4:
    print("Uso: heart_rules_snapshot.py <rules_src.yaml> <rules_out.yaml> <ATR_MAX>")
    sys.exit(1)

src, out, atrS = sys.argv[1], sys.argv[2], sys.argv[3]
atr = float(atrS)

cfg = {}
if os.path.exists(src):
    with open(src, 'r') as f:
        d = yaml.safe_load(f) or {}
        if isinstance(d, dict):
            cfg.update(d)

cfg['ATR_MAX'] = atr

os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"[SNAP] {out} con ATR_MAX={atr}")
