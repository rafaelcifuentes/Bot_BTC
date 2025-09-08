#!/usr/bin/env python3
# select_winners.py
import argparse, json, math, sys
from pathlib import Path
import pandas as pd

def top_near_miss(df_q4: pd.DataFrame):
    # Ordena por (pf desc, wr desc, trades desc) y toma la primera
    cand = df_q4.sort_values(
        by=["pf", "wr", "trades"],
        ascending=[False, False, False],
        na_position="last"
    ).head(1)
    return None if cand.empty else cand.iloc[0].to_dict()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q4", required=True, help="CSV de validación Q4")
    ap.add_argument("--h1", required=True, help="CSV de validación 2024H1")
    ap.add_argument("--pf", type=float, default=1.6)
    ap.add_argument("--wr", type=float, default=0.60)
    ap.add_argument("--trades", type=int, default=30)
    ap.add_argument("--prune", default="th", choices=["th", "rearm"],
                    help="Estrategia de poda suave (threshold +0.02 ó REARM_MIN +2)")
    ap.add_argument("--out_csv", default="reports/winners_BothFolds.csv")
    ap.add_argument("--out_json", default="reports/candidate.json")
    args = ap.parse_args()

    q4 = pd.read_csv(args.q4)
    h1 = pd.read_csv(args.h1)

    # Campos mínimos esperados
    base_cols = ["asset","horizon","threshold","trades","pf","wr"]
    for name,df in [("Q4",q4),("2024H1",h1)]:
        missing = [c for c in base_cols if c not in df.columns]
        if missing:
            print(f"[WARN] {name} sin columnas {missing}, revisa el lector.", file=sys.stderr)

    # Renombra para merge
    q4_r = q4.rename(columns={
        "pf":"pf_Q4","wr":"wr_Q4","trades":"trades_Q4"
    })
    h1_r = h1.rename(columns={
        "pf":"pf_2024H1","wr":"wr_2024H1","trades":"trades_2024H1"
    })

    on_keys = ["asset","horizon","threshold"]
    merged = q4_r.merge(h1_r, on=on_keys, how="inner")

    winners = merged[
        (merged["pf_Q4"] >= args.pf) & (merged["wr_Q4"] >= args.wr) & (merged["trades_Q4"] >= args.trades) &
        (merged["pf_2024H1"] >= args.pf) & (merged["wr_2024H1"] >= args.wr) & (merged["trades_2024H1"] >= args.trades)
    ].copy()

    cols = on_keys + ["pf_Q4","wr_Q4","trades_Q4","pf_2024H1","wr_2024H1","trades_2024H1"]
    if not winners.empty:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        winners[cols].to_csv(args.out_csv, index=False)
        print("[OK] Ganadores en ambos folds:")
        print(winners[cols].to_string(index=False))
        # Nada más que hacer
        Path(args.out_json).write_text(json.dumps({"winners": winners[cols].to_dict(orient="records")}, indent=2))
        return

    # Near-miss Q4 -> sugerir poda suave
    cand = top_near_miss(q4)
    if cand is None:
        print("[INFO] Sin near-miss en Q4; no hay sugerencia.", file=sys.stderr)
        Path(args.out_json).write_text(json.dumps({"suggestion": None}, indent=2))
        return

    suggestion = {"asset": cand["asset"], "horizon": int(cand["horizon"])}
    if args.prune == "th":
        th_new = round(float(cand["threshold"]) + 0.02, 2)
        suggestion.update({"threshold_new": th_new, "prune": "threshold+0.02"})
    else:
        suggestion.update({"rearm_min_delta": 2, "prune": "rearm+2"})

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"suggestion": suggestion, "near_miss_Q4": cand}, indent=2))
    print("[OK] Sugerencia de poda suave guardada en", args.out_json)
    print(json.dumps({"suggestion": suggestion}, indent=2))

if __name__ == "__main__":
    main()