# scripts/quick_kpi_diagnosis.py
import sys, argparse
from pathlib import Path
import pandas as pd
import numpy as np

def find_kpis(path_hint: str) -> Path:
    hint = Path(path_hint)
    if hint.exists():
        return hint

    here = Path.cwd()
    candidates = []
    # 1) buscar por nombre en el cwd
    candidates += list(here.rglob("kpis_grid.csv"))
    # 2) si el script está en .../scripts, asumir repo = padre
    try:
        repo = Path(__file__).resolve().parents[1]
        candidates += list(repo.rglob("kpis_grid.csv"))
    except Exception:
        pass

    # filtrar duplicados y elegir el más reciente
    uniq = {c.resolve() for c in candidates if c.is_file()}
    if not uniq:
        sys.exit("[ERROR] No se encontró 'kpis_grid.csv'. Genera primero el grid con backtest_grid.py.")

    best = max(uniq, key=lambda p: p.stat().st_mtime)
    print(f"[info] Usando CSV detectado: {best}")
    return best

def coalesce_cols(df, target, *aliases):
    cols = [c for c in df.columns]
    low = {c.lower(): c for c in cols}
    for a in (target,) + aliases:
        if a in low:
            return low[a]
    # alias compuestos
    alias_sets = {
        "pf": {"profit_factor","pf"},
        "wr": {"win_rate","winrate","wr"},
        "trades": {"n_trades","trades_count","trades"},
        "window":{"window"},
        "horizon":{"horizon"},
        "threshold":{"threshold"},
    }
    if target in alias_sets:
        for a in alias_sets[target]:
            if a in low:
                return low[a]
    raise KeyError(f"Columna requerida no encontrada: {target} (disponibles: {cols})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kpis_csv", default="reports/kpis_grid.csv")
    ap.add_argument("--gate_pf", type=float, default=1.6)
    ap.add_argument("--gate_wr", type=float, default=0.60)
    ap.add_argument("--gate_trades", type=int, default=30)
    args = ap.parse_args()

    path = find_kpis(args.kpis_csv)
    print(f"[cwd] {Path.cwd()}")
    print(f"[read] {path}")

    df = pd.read_csv(path)
    # columnas
    c_pf = coalesce_cols(df, "pf")
    c_wr = coalesce_cols(df, "wr")
    c_tr = coalesce_cols(df, "trades")
    c_w  = coalesce_cols(df, "window")
    c_h  = coalesce_cols(df, "horizon")
    c_th = coalesce_cols(df, "threshold")

    # normaliza WR (0..1)
    df["_pf"] = pd.to_numeric(df[c_pf], errors="coerce")
    df["_wr"] = pd.to_numeric(df[c_wr], errors="coerce")
    if df["_wr"].max(skipna=True) and df["_wr"].max() > 1.0:
        df["_wr"] = df["_wr"] / 100.0
    df["_tr"] = pd.to_numeric(df[c_tr], errors="coerce").fillna(0).astype(int)
    df["_w"]  = df[c_w].astype(str)
    df["_h"]  = df[c_h].astype(str)
    df["_th"] = pd.to_numeric(df[c_th], errors="coerce")

    # gates
    df["_pass_pf"] = df["_pf"] >= args.gate_pf
    df["_pass_wr"] = df["_wr"] >= args.gate_wr
    df["_pass_tr"] = df["_tr"] >= args.gate_trades
    df["_passes"]  = df[["_pass_pf","_pass_wr","_pass_tr"]].sum(axis=1)

    # Cobertura por ventana
    print("\n=== Cobertura por ventana (proporción que pasa cada gate) ===")
    cov = (df.groupby("_w")[["_pass_pf","_pass_wr","_pass_tr"]]
             .mean(numeric_only=True)
             .rename(columns={"_pass_pf":"%PF","_pass_wr":"%WR","_pass_tr":"%Trades"})).round(3)
    print(cov.to_string())

    # Near-misses
    near_pf = ((df["_pf"]>=args.gate_pf-0.15) & (df["_pf"]<args.gate_pf)).sum()
    near_wr = ((df["_wr"]>=args.gate_wr-0.02) & (df["_wr"]<args.gate_wr)).sum()
    near_tr = ((df["_tr"]>=args.gate_trades-5) & (df["_tr"]<args.gate_trades)).sum()
    print("\n=== Near-misses (casi pasan) ===")
    print(f"PF cerca: {near_pf} | WR cerca: {near_wr} | Trades cerca: {near_tr}")

    # Candidatos que fallan 1 solo gate
    print("\n=== Top candidatos que fallan 1 solo gate ===")
    cand = df[df["_passes"]==2].sort_values(["_pf","_wr","_tr"], ascending=False).head(12)
    if not cand.empty:
        print(cand[["_w","_h","_th","_pf","_wr","_tr"]]
              .rename(columns={"_w":"window","_h":"horizon","_th":"threshold",
                               "_pf":"pf","_wr":"wr","_tr":"trades"}).to_string(index=False))
    else:
        print("(no hay filas con solo 1 fallo)")

    # Agregación por (horizon, threshold) con peor ventana
    print("\n=== (horizon, threshold) — min por ventana y medianas (solo full_coverage) ===")
    g = (df.groupby(["_h","_th"])
           .agg(pf_min=("_pf","min"), wr_min=("_wr","min"), trades_min=("_tr","min"),
                pf_med=("_pf","median"), wr_med=("_wr","median"), trades_med=("_tr","median"),
                coverage=("_w","nunique"))
           .reset_index())
    g = g[g["coverage"]>=3]
    if not g.empty:
        out = g.sort_values(["pf_min","wr_min","trades_min"], ascending=False)\
               .head(12)[["_h","_th","pf_min","wr_min","trades_min","pf_med","wr_med","trades_med"]]\
               .rename(columns={"_h":"horizon","_th":"threshold"})
        print(out.to_string(index=False))
    else:
        print("(no hay combos con coverage=3; una ventana es el cuello de botella)")

if __name__ == "__main__":
    main()