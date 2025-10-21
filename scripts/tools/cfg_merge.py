#!/usr/bin/env python3
import sys, yaml, os, copy
def dmerge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        r = copy.deepcopy(a)
        for k, v in b.items():
            r[k] = dmerge(r.get(k), v)
        return r
    return copy.deepcopy(b)
def main():
    if len(sys.argv) < 4:
        print("uso: cfg_merge.py BASE.yaml OVERLAY.yaml OUT.yaml"); sys.exit(2)
    base, ovl, out = sys.argv[1:4]
    with open(base) as f: A = yaml.safe_load(f) or {}
    with open(ovl)  as f: B = yaml.safe_load(f) or {}
    M = dmerge(A, B)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f: yaml.safe_dump(M, f, sort_keys=False)
    print(f"[OK] merged → {out}")
if __name__ == "__main__":
    main()
