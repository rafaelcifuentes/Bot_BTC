#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, datetime, pathlib
import pandas as pd
from typing import List, Dict, Any

def load_fold_csv(path: str, fold_tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"asset","horizon","threshold","pf","wr","trades"}
    missing = need - set(map(str.lower, df.columns))
    # normaliza columnas esperadas (respetando mayúsculas originales si existen)
    cols = {c.lower():c for c in df.columns}
    def col(name): return cols.get(name, name)

    sel = df[[col("asset"), col("horizon"), col("threshold"),
              col("pf"), col("wr"), col("trades")]].copy()
    sel.columns = ["asset","horizon","threshold",
                   f"pf_{fold_tag}",f"wr_{fold_tag}",f"trades_{fold_tag}"]
    # coerción segura
    for c in ["horizon","threshold",f"pf_{fold_tag}",f"wr_{fold_tag}",f"trades_{fold_tag}"]:
        sel[c] = pd.to_numeric(sel[c], errors="coerce")
    sel = sel.dropna(subset=["asset","horizon","threshold"])
    sel = sel.drop_duplicates(subset=["asset","horizon","threshold"], keep="first")
    return sel

def dump_yaml(d: Dict[str,Any]) -> str:
    # yaml simple y portable
    def _dump(obj, indent=0):
        sp = "  " * indent
        if isinstance(obj, dict):
            out = []
            for k,v in obj.items():
                if isinstance(v,(dict,list)):
                    out.append(f"{sp}{k}:")
                    out.append(_dump(v, indent+1))
                else:
                    val = "null" if v is None else v
                    out.append(f"{sp}{k}: {val}")
            return "\n".join(out)
        elif isinstance(obj, list):
            out = []
            for it in obj:
                if isinstance(it,(dict,list)):
                    out.append(f"{sp}-")
                    out.append(_dump(it, indent+1))
                else:
                    out.append(f"{sp}- {it}")
            return "\n".join(out)
        else:
            return f"{sp}{obj}"
    return _dump(d) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q4", required=True, help="CSV de Q4 (ej. reports/val_Q4_p5050.csv)")
    ap.add_argument("--h1", required=True, help="CSV de 2024H1 (ej. reports/val_2024H1_p5050.csv)")
    ap.add_argument("--out", required=True, help="Ruta YAML de salida (ej. configs/diamante_selected.yaml)")
    ap.add_argument("--pf_min", type=float, default=1.6)
    ap.add_argument("--wr_min", type=float, default=0.60)
    ap.add_argument("--trades_min", type=int, default=30)
    ap.add_argument("--allow_near_miss", type=int, default=1)
    ap.add_argument("--threshold_bump", type=float, default=0.02)
    args = ap.parse_args()

    q4 = load_fold_csv(args.q4, "Q4")
    h1 = load_fold_csv(args.h1, "2024H1")
    merged = pd.merge(q4, h1, on=["asset","horizon","threshold"], how="inner")

    winners = merged[
        (merged["pf_Q4"]>=args.pf_min) & (merged["wr_Q4"]>=args.wr_min) & (merged["trades_Q4"]>=args.trades_min) &
        (merged["pf_2024H1"]>=args.pf_min) & (merged["wr_2024H1"]>=args.wr_min) & (merged["trades_2024H1"]>=args.trades_min)
    ].copy()

    winners = winners.sort_values(
        by=["pf_Q4","wr_Q4","trades_Q4","pf_2024H1","wr_2024H1","trades_2024H1"],
        ascending=[False]*6
    )

    outd: Dict[str,Any] = {
        "name": "diamante_day3_selection",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "gates": {"pf_min": args.pf_min, "wr_min": args.wr_min, "trades_min": args.trades_min},
        "source_files": [args.q4, args.h1],
        "winners": []
    }

    for _,r in winners.iterrows():
        outd["winners"].append({
            "asset": r["asset"],
            "horizon": int(r["horizon"]),
            "threshold": float(r["threshold"]),
            "metrics": {
                "Q4": {"pf": float(r["pf_Q4"]), "wr": float(r["wr_Q4"]), "trades": int(r["trades_Q4"])},
                "2024H1": {"pf": float(r["pf_2024H1"]), "wr": float(r["wr_2024H1"]), "trades": int(r["trades_2024H1"])},
            }
        })

    # Near-miss si no hay ganadores: toma el mejor de Q4 y bump +0.02
    near_miss = None
    if not outd["winners"] and args.allow_near_miss:
        best_q4 = q4.sort_values(by=["pf_Q4","wr_Q4","trades_Q4"], ascending=[False,False,False]).head(1)
        if not best_q4.empty:
            b = best_q4.iloc[0]
            near_miss = {
                "asset": b["asset"],
                "horizon": int(b["horizon"]),
                "threshold": round(float(b["threshold"]) + args.threshold_bump, 2),
                "from": {"threshold": float(b["threshold"]), "bump": args.threshold_bump},
                "note": "near-miss Q4 (aplicar mismo H/TH en 2024H1)"
            }
            outd["near_miss"] = near_miss

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(dump_yaml(outd), encoding="utf-8")

    if outd["winners"]:
        print(f"[ok] winners -> {args.out} (n={len(outd['winners'])})")
    elif near_miss:
        print(f"[ok] near-miss -> {args.out}  H={near_miss['horizon']} TH={near_miss['threshold']}")
    else:
        print(f"[warn] sin ganadores ni near-miss: revisa gates o CSVs")

if __name__ == "__main__":
    main()