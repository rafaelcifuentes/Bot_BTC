#!/usr/bin/env bash
set -euo pipefail

# === Config ===
CSV_DIR="reports/mini_accum/kiss_v1"
WF_CSV="configs/mini_accum/windows_walkforward.csv"
OUT_DIR="reports/mini_accum/walkforward"
mkdir -p "$OUT_DIR"

# Candidato por env vars o defaults
CAND_DD="${CAND_DD:-15}"
CAND_RB="${CAND_RB:-1}"
CAND_H="${CAND_H:-30}"

python - <<'PY'
import pandas as pd, numpy as np, re, glob, os
from pathlib import Path

CSV_DIR = "reports/mini_accum/kiss_v1"
WF_CSV  = "configs/mini_accum/windows_walkforward.csv"
OUT_DIR = "reports/mini_accum/walkforward"

# --- util ---
def year_frac(start, end):
    s = pd.to_datetime(start); e = pd.to_datetime(end)
    return max((e - s).days, 1) / 365.25

def cagr(mult, yrs):
    try:
        m = float(mult)
        if m <= 0 or yrs <= 0: return np.nan
        return m**(1/yrs) - 1.0
    except Exception:
        return np.nan

def read_kpis(path):
    """
    Devuelve dict con llaves estándar:
    sats_mult, USD_net, mdd_vsHODL, fpy, flips, run_ok, skip_reason
    """
    d = dict(sats_mult=np.nan, USD_net=np.nan, mdd_vsHODL=np.nan,
             fpy=np.nan, flips=np.nan, run_ok=False, skip_reason="")
    try:
        df = pd.read_csv(path)
        cols = {c.strip():c for c in df.columns}
        # Mapeo flexible por presencia de columnas
        d["sats_mult"]  = float(df.get(cols.get("net_btc_vs_hodl",""), [np.nan])[0]) \
                          if "net_btc_vs_hodl" in cols else \
                          float(df.get(cols.get("sats_mult",""), [np.nan])[0])
        d["USD_net"]    = float(df.get(cols.get("net_btc_ratio",""), [np.nan])[0]) \
                          if "net_btc_ratio" in cols else \
                          float(df.get(cols.get("USD_net",""), [np.nan])[0])
        # mdd vs hodl
        if "mdd_vs_hodl_ratio" in cols:
            d["mdd_vsHODL"] = float(df.iloc[0][cols["mdd_vs_hodl_ratio"]])
        elif "mdd_vsHODL" in cols:
            d["mdd_vsHODL"] = float(df.iloc[0][cols["mdd_vsHODL"]])
        # flips
        if "flips_total" in cols:
            d["flips"] = int(df.iloc[0][cols["flips_total"]])
        elif "flips" in cols:
            d["flips"] = int(df.iloc[0][cols["flips"]])
        # fpy
        if "flips_per_year" in cols:
            d["fpy"] = float(df.iloc[0][cols["flips_per_year"]])
        elif "fpy" in cols:
            d["fpy"] = float(df.iloc[0][cols["fpy"]])
        # run_ok / skip_reason
        if "run_ok" in cols:
            v = df.iloc[0][cols["run_ok"]]
            d["run_ok"] = bool(v) if isinstance(v, (bool,np.bool_)) else str(v).strip().lower()=="true"
        if "skip_reason" in cols:
            d["skip_reason"] = str(df.iloc[0][cols["skip_reason"]])
        return d
    except Exception:
        # Intento fallback: leer summary md en formato "KISS v1 — Resumen ..."
        md = re.sub(r"_kpis__", "_summary__", path).rsplit(".",1)[0] + ".md"
        if not Path(md).exists(): return d
        txt = Path(md).read_text(encoding="utf-8", errors="ignore")
        def grab(label):
            m = re.search(rf"\*\*{re.escape(label)}\*\*:\s*([0-9\.Ee+-]+)", txt)
            return float(m.group(1)) if m else np.nan
        d["USD_net"]    = grab("net_btc_ratio")
        d["sats_mult"]  = grab("net_btc_vs_hodl")
        d["mdd_vsHODL"] = grab("mdd_vs_hodl_ratio")
        # flips / fpy (pueden venir en bullets)
        m = re.search(r"\*\*flips_total\*\*:\s*([0-9]+)", txt)
        d["flips"] = int(m.group(1)) if m else np.nan
        m = re.search