import argparse
from pathlib import Path
import pandas as pd

WINDOWS = {
    "2022H2": ("2022-06-01", "2023-01-31"),
    "2023Q4": ("2023-10-01", "2023-12-31"),
    "2024H1": ("2024-01-01", "2024-06-30"),
}

def load_and_fix(in_path: Path):
    df = pd.read_csv(in_path)
    cols = {c.lower(): c for c in df.columns}
    # renombrar ts -> timestamp si hace falta
    if "timestamp" in cols:
        tcol = cols["timestamp"]
    elif "ts" in cols:
        tcol = cols["ts"]
        df = df.rename(columns={tcol: "timestamp"})
        tcol = "timestamp"
    else:
        raise SystemExit(f"[ERROR] {in_path} no trae ts/timestamp.")

    # proba
    if "proba" not in cols and "proba" not in df.columns:
        raise SystemExit(f"[ERROR] {in_path} no trae proba.")
    pcol = "proba"

    # timestamp -> UTC naive (sin tz)
    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.tz_localize(None)
    proba = pd.to_numeric(df[pcol], errors="coerce")
    out = pd.DataFrame({"timestamp": ts, "proba": proba}).dropna().sort_values("timestamp")
    return out

def clip_window(df: pd.DataFrame, start: str, end: str):
    mask = (df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end))
    return df.loc[mask].reset_index(drop=True)

def describe(df: pd.DataFrame, label: str):
    q = df["proba"].quantile([0, 0.25, 0.5, 0.75, 1.0]).round(4).to_dict() if not df.empty else {}
    c50 = int((df["proba"] >= 0.50).sum())
    c58 = int((df["proba"] >= 0.58).sum())
    c60 = int((df["proba"] >= 0.60).sum())
    c62 = int((df["proba"] >= 0.62).sum())
    print(f"[{label}] rows={len(df)}  proba_q={q}  >=0.50:{c50}  >=0.58:{c58}  >=0.60:{c60}  >=0.62:{c62}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", default="reports/windows")
    ap.add_argument("--out_root", default="reports/windows_fixed")
    ap.add_argument("--asset", default="BTC-USD")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    for w, (start, end) in WINDOWS.items():
        inp = in_root / w / f"{args.asset}.csv"
        if not inp.exists():
            print(f"[MISS] {inp}"); continue
        df = load_and_fix(inp)
        describe(df, f"{w}-ALL")
        dfw = clip_window(df, start, end)
        describe(dfw, f"{w}-CLIPPED {start}->{end}")

        outp = out_root / w / f"{args.asset}.csv"
        outp.parent.mkdir(parents=True, exist_ok=True)
        dfw.to_csv(outp, index=False)
        print(f"[OK] wrote {outp}  rows={len(dfw)}")

if __name__ == "__main__":
    main()