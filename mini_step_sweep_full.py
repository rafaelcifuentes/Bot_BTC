#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini-step sweep runner for Sentiment EXP (without touching V2).

It runs small, controlled experiments and writes a CSV summary with the
key metrics parsed from the runner's stdout:
  - Selected → Net, PF, Win%, Trades, MDD, Score  ("180d base")
  - Holdout 75d → Net, PF, Win%, Trades, MDD, Score

Experiments included (enable via --phases):
  1) src    : Compare ADX daily source: resample vs yfinance
  2) len    : Sweep ADX1D len: e.g., 10,14,20
  3) ptsl   : Micro-grid for SL/TP/Trail around the current defaults

Outputs:
  - ./reports/mini_step_logs/<timestamp>/*.log (raw logs per run)
  - ./reports/mini_step_summary_<timestamp>.csv (aggregated results)

Requirements:
  - runner_profileA_RF_sentiment_EXP.py available in the working dir
  - Python 3.8+

Usage examples:
  $ export FREEZE=2025-08-10
  $ python mini_step_sweep.py --phases src len ptsl
  # or only the first two phases
  $ python mini_step_sweep.py --phases src len
"""

import argparse
import datetime as dt
import os
import re
import shlex
import subprocess
import sys
import csv
from pathlib import Path
from typing import Dict, Any, List

RUNNER = "runner_profileA_RF_sentiment_EXP.py"

SELECTED_RE = re.compile(
    r"Selected\s*→\s*Net\s*([+\-]?\d+(?:\.\d+)?)\s*,\s*PF\s*(\d+(?:\.\d+)?)\s*,\s*"
    r"Win%\s*(\d+(?:\.\d+)?)\s*,\s*Trades\s*(\d+)\s*,\s*MDD\s*(\d+(?:\.\d+)?)\s*,\s*"
    r"Score\s*([+\-]?\d+(?:\.\d+)?)"
)

HOLDOUT_RE = re.compile(
    r"Holdout\s*75d\s*→\s*Net\s*([+\-]?\d+(?:\.\d+)?)\s*,\s*PF\s*(\d+(?:\.\d+)?)\s*,\s*"
    r"Win%\s*(\d+(?:\.\d+)?)\s*,\s*Trades\s*(\d+)\s*,\s*MDD\s*(\d+(?:\.\d+)?)\s*,\s*"
    r"Score\s*([+\-]?\d+(?:\.\d+)?)"
)

JSON_PATH_RE = re.compile(r"Saved report JSON\s*→\s*([^\s]+)")
ADX_PASS_RE = re.compile(r"ADX1D gate.*?→\s*.*?\s*→\s*(\d+)/(\d+)\s*bars\s*\((\d+\.\d+)%\)\s*pass")


def run_and_capture(cmd: List[str], log_dir: Path) -> Dict[str, Any]:
    """Runs a command, captures stdout, parses metrics, returns dict."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout

    # save raw log
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"run_{stamp}.log"
    log_file.write_text(out)

    sel = SELECTED_RE.search(out)
    hold = HOLDOUT_RE.search(out)
    jpath = JSON_PATH_RE.search(out)
    adx = ADX_PASS_RE.search(out)

    def fnum(m, i):
        return float(m.group(i)) if m else None

    def inum(m, i):
        return int(float(m.group(i))) if m else None

    res = {
        "net180": fnum(sel, 1),
        "pf180": fnum(sel, 2),
        "win180": fnum(sel, 3),
        "trades180": inum(sel, 4),
        "mdd180": fnum(sel, 5),
        "score180": fnum(sel, 6),
        "hold_net": fnum(hold, 1),
        "hold_pf": fnum(hold, 2),
        "hold_win": fnum(hold, 3),
        "hold_trades": inum(hold, 4),
        "hold_mdd": fnum(hold, 5),
        "hold_score": fnum(hold, 6),
        "summary_json": jpath.group(1) if jpath else None,
        "log_path": str(log_file),
    }
    if adx:
        res.update({
            "adx_pass": inum(adx, 1),
            "adx_total": inum(adx, 2),
            "adx_pct": fnum(adx, 3),
        })
    return res


def base_cmd(args) -> List[str]:
    cmd = [
        sys.executable, RUNNER,
        "--primary_only",
        "--freeze_end", args.freeze_end,
        "--repro_lock",
        "--no_sentiment",
        "--threshold", f"{args.threshold:.2f}",
        "--pt_mode", "pct",
    ]
    return cmd


def add_adx(cmd: List[str], source: str, adx_len: int, adx_min: float) -> List[str]:
    cmd += [
        "--adx_daily_source", source,
        "--adx1d_len", str(adx_len),
        "--adx1d_min", str(adx_min),
    ]
    return cmd


def add_ptsl(cmd: List[str], sl: float, tp: float, trail: float) -> List[str]:
    cmd += [
        "--sl_pct", f"{sl}",
        "--tp_pct", f"{tp}",
        "--trail_pct", f"{trail}",
    ]
    return cmd


def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    if not rows:
        return
    cols = [
        "phase", "source", "adx_len", "adx_min", "sl", "tp", "trail", "threshold",
        "net180", "pf180", "win180", "trades180", "mdd180", "score180",
        "hold_net", "hold_pf", "hold_win", "hold_trades", "hold_mdd", "hold_score",
        "adx_pass", "adx_total", "adx_pct", "summary_json", "log_path"
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def main():
    parser = argparse.ArgumentParser(description="Mini-step sweep runner (EXP)")
    parser.add_argument("--freeze_end", default=os.environ.get("FREEZE", "2025-08-10"))
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--phases", nargs="+", default=["src", "len", "ptsl"],
                        choices=["src", "len", "ptsl", "all"],
                        help="Which phases to run")
    parser.add_argument("--sources", nargs="+", default=["resample", "yfinance"])
    parser.add_argument("--adx_lens", nargs="+", type=int, default=[10, 14, 20])
    parser.add_argument("--adx_min", type=float, default=10.0)
    parser.add_argument("--sl_grid", nargs="+", type=float, default=[0.02, 0.03])
    parser.add_argument("--tp_grid", nargs="+", type=float, default=[0.06, 0.08])
    parser.add_argument("--trail_grid", nargs="+", type=float, default=[0.008, 0.012])

    args = parser.parse_args()

    phases = args.phases
    if "all" in phases:
        phases = ["src", "len", "ptsl"]

    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path("./reports/mini_step_logs") / ts
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(f"./reports/mini_step_summary_{ts}.csv")

    rows: List[Dict[str, Any]] = []

    # Phase 1: source comparison
    if "src" in phases:
        for src in args.sources:
            cmd = base_cmd(args)
            cmd = add_ptsl(cmd, 0.025, 0.05, 0.01)  # current defaults
            cmd = add_adx(cmd, src, 14, args.adx_min)
            print("\n[PHASE src] ⇒", " ".join(shlex.quote(c) for c in cmd))
            res = run_and_capture(cmd, logs_dir)
            res.update({
                "phase": "src", "source": src,
                "adx_len": 14, "adx_min": args.adx_min,
                "sl": 0.025, "tp": 0.05, "trail": 0.01,
                "threshold": args.threshold,
            })
            rows.append(res)

    # Phase 2: ADX length sweep (keep resample by default)
    if "len" in phases:
        for L in args.adx_lens:
            cmd = base_cmd(args)
            cmd = add_ptsl(cmd, 0.025, 0.05, 0.01)
            cmd = add_adx(cmd, "resample", L, args.adx_min)
            print("\n[PHASE len] ⇒", " ".join(shlex.quote(c) for c in cmd))
            res = run_and_capture(cmd, logs_dir)
            res.update({
                "phase": "len", "source": "resample",
                "adx_len": L, "adx_min": args.adx_min,
                "sl": 0.025, "tp": 0.05, "trail": 0.01,
                "threshold": args.threshold,
            })
            rows.append(res)

    # Phase 3: PT/SL micro-grid (resample, len=14)
    if "ptsl" in phases:
        for sl in args.sl_grid:
            for tp in args.tp_grid:
                for tr in args.trail_grid:
                    cmd = base_cmd(args)
                    cmd = add_ptsl(cmd, sl, tp, tr)
                    cmd = add_adx(cmd, "resample", 14, args.adx_min)
                    print("\n[PHASE ptsl] ⇒", " ".join(shlex.quote(c) for c in cmd))
                    res = run_and_capture(cmd, logs_dir)
                    res.update({
                        "phase": "ptsl", "source": "resample",
                        "adx_len": 14, "adx_min": args.adx_min,
                        "sl": sl, "tp": tp, "trail": tr,
                        "threshold": args.threshold,
                    })
                    rows.append(res)

    write_csv(rows, out_csv)

    # Print a short top-5 by pf180 and by score180
    def top_n(key: str):
        valid = [r for r in rows if r.get(key) is not None]
        return sorted(valid, key=lambda r: r[key], reverse=True)[:5]

    print("\nSaved summary →", out_csv)
    print("\nTop-5 by PF (180d):")
    for i, r in enumerate(top_n("pf180"), 1):
        print(f" {i:>2}. phase={r['phase']}, src={r['source']}, len={r['adx_len']}, adx_min={r['adx_min']}, "
              f"sl={r['sl']}, tp={r['tp']}, trail={r['trail']} | PF={r['pf180']} Score={r['score180']} Net={r['net180']}")

    print("\nTop-5 by Score (180d):")
    for i, r in enumerate(top_n("score180"), 1):
        print(f" {i:>2}. phase={r['phase']}, src={r['source']}, len={r['adx_len']}, adx_min={r['adx_min']}, "
              f"sl={r['sl']}, tp={r['tp']}, trail={r['trail']} | Score={r['score180']} PF={r['pf180']} Net={r['net180']}")


if __name__ == "__main__":
    main()
