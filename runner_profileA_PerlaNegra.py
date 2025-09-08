#!/usr/bin/env python3
import sys, subprocess, glob, os, shutil
from pathlib import Path

def latest_ts():
    files = sorted(glob.glob("reports/summary_rf_sentiment_EXP_*.json"),
                   key=os.path.getmtime, reverse=True)
    if not files:
        return None
    base = os.path.basename(files[0])
    return base.rsplit("_", 1)[1].split(".")[0]

def alias_reports(tag, ts):
    src_sum  = Path(f"reports/summary_rf_sentiment_EXP_{ts}.json")
    src_wf   = Path(f"reports/walkforward_rf_sentiment_EXP_{ts}.csv")
    src_comp = Path(f"reports/competitiveness_summary_rf_sentiment_EXP_{ts}.json")
    dst_sum  = Path(f"reports/summary_{tag}_{ts}.json")
    dst_wf   = Path(f"reports/walkforward_{tag}_{ts}.csv")
    dst_comp = Path(f"reports/competitiveness_summary_{tag}_{ts}.json")
    for s, d in [(src_sum, dst_sum), (src_wf, dst_wf), (src_comp, dst_comp)]:
        if s.exists():
            shutil.copy2(s, d)
            print(f"📎 Aliased {d.name} → {d}")
    return dst_sum.exists() and dst_wf.exists()

def main():
    Path("reports").mkdir(exist_ok=True)
    cmd = [sys.executable, "runner_profileA_RF_sentiment_EXP.py"] + sys.argv[1:]
    print(">>> Delegating to EXP runner:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    ts = latest_ts()
    if ts:
        alias_reports("perla_negra", ts)

if __name__ == "__main__":
    main()
