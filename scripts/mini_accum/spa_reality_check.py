#!/usr/bin/env python3
import os, json, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from arch.bootstrap import SPA, RealityCheck

ROOT = Path(os.environ.get("ROOT", str(Path.home() / "PycharmProjects" / "Bot_BTC")))
REPORTS = ROOT / "reports" / "mini_accum"
LOGS = ROOT / "logs"
REPORTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

# Config
INPUT = Path(os.environ.get("SPA_INPUT_CSV", str(REPORTS / "spa_input.csv")))  # CSV con benchmark + modelos
REPS = int(os.environ.get("SPA_REPS", "1000"))
ALPHA = float(os.environ.get("SPA_ALPHA", "0.10"))
BLOCK_ENV = os.environ.get("SPA_BLOCK", "").strip()
BLOCK = int(BLOCK_ENV) if BLOCK_ENV.isdigit() else None
SEED = int(os.environ.get("SPA_SEED", "42"))

def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def split_benchmark_models(df: pd.DataFrame):
    # benchmark = 'benchmark'/'baseline' si existe; si no, 1ª columna. Resto = modelos
    cand = [c for c in df.columns if str(c).lower() in ("benchmark","baseline","bh","buy_hold","buy&hold","buy_and_hold")]
    bcol = cand[0] if cand else df.columns[0]
    bench = df[bcol].to_numpy()
    mods  = df.drop(columns=[bcol]).to_numpy()
    if mods.size == 0:
        raise ValueError("No hay columnas de modelos (k=0); se requiere ≥2 columnas en total.")
    return bench, mods, str(bcol)

def to_losses(df: pd.DataFrame) -> pd.DataFrame:
    # Si parece ya retornos (valores razonables en [-1.5, 1.5]), los usa.
    # Si parece equity nivel, calcula pct_change.
    arr = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if np.isfinite(arr).all() and np.nanmax(np.abs(arr)) <= 1.5:
        returns = df
    else:
        returns = df.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    losses = -returns
    return losses

def save_report(payload: dict):
    out = REPORTS / "spa_reality_report.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False))
    return out

def main():
    ts = now_utc()
    if not INPUT.exists():
        payload = {"status":"SKIP","reason":"no_input_csv","input":str(INPUT),"ts_utc":ts}
        save_report(payload); sys.exit(0)

    df = pd.read_csv(INPUT)
    # Sólo columnas numéricas
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] < 2:
        payload = {"status":"SKIP","reason":"need>=2_numeric_cols","cols":list(df.columns),"ts_utc":ts}
        save_report(payload); sys.exit(0)

    losses = to_losses(df)
    bench, mods, bcol = split_benchmark_models(losses)
    T = losses.shape[0]
    block = BLOCK if BLOCK is not None else max(5, int(np.sqrt(T)))

    # SPA
    spa = SPA(bench, mods, block_size=block, reps=REPS, studentize=True, seed=SEED)
    spa.compute()
    spa_p = tuple(float(x) for x in spa.pvalues)  # (lower, consistent, upper)

    # Reality Check
    rc = RealityCheck(bench, mods, block_size=block, reps=REPS, studentize=True, seed=SEED)
    rc.compute()
    rc_p = tuple(float(x) for x in rc.pvalues)

    # Regla simple: pasar si min p-valor "consistent" <= alpha en ambos tests
    pass_consistent = (spa_p[1] <= ALPHA) and (rc_p[1] <= ALPHA)

    payload = {
        "status": "OK",
        "ts_utc": ts,
        "input_csv": str(INPUT),
        "rows": int(T),
        "cols_total": int(df.shape[1]),
        "benchmark_col": bcol,
        "block_size": int(block),
        "reps": int(REPS),
        "alpha": ALPHA,
        "spa":{"p_lower":spa_p[0], "p_consistent":spa_p[1], "p_upper":spa_p[2]},
        "rc": {"p_lower":rc_p[0], "p_consistent":rc_p[1], "p_upper":rc_p[2]},
        "decision_consistent": "PASS" if pass_consistent else "FAIL"
    }
    save_report(payload)

if __name__ == "__main__":
    main()
