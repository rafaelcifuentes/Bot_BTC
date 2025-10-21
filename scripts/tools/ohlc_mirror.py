#!/usr/bin/env python3
import argparse, hashlib
from pathlib import Path
import pandas as pd

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts":"timestamp"})
    need = ["timestamp","open","high","low","close"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"[ERROR] Falta columna requerida: {c}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return df[["timestamp","open","high","low","close","volume"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)  # d1
    ap.add_argument("--dst", required=True)  # 1d
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    df = normalize(df)

    tmp = dst.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(dst)

    print(f"[OK] Espejado d1 → 1d")
    print(f"     src: {src}")
    print(f"     dst: {dst}")
    print(f"sha256 src: {sha256(src)}")
    print(f"sha256 dst: {sha256(dst)}")

if __name__ == "__main__":
    main()
