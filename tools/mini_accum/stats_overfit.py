#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KISS v1 — Tests anti-overfitting (CSCV/PBO + DSR)
- Lee wf_summary_kpis.csv
- Filtra ventanas (--windows)
- Calcula PBO (CSCV) con n_samples
- Si existen ret_mean/ret_std/n_obs, calcula DSR agregado
- Anexa resultados a Roadmap_PDCA.md
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import random
import math

def _sanitize(df: pd.DataFrame, windows, metric: str) -> pd.DataFrame:
    d = df.copy()
    d["window"] = d["window"].astype(str)
    if windows:
        d = d[d["window"].isin(windows)]
    # métrica numérica
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    # ids válidos
    d = d.dropna(subset=["config_id","window", metric])
    d["config_id"] = d["config_id"].astype(str)
    d = d[d["config_id"].str.match(r"DD\d+_RB\d+_H\d+_G\d+_BULL\d+")]
    # requiere ≥2 configs por ventana para ranking
    counts = d.groupby("window")["config_id"].nunique()
    valid_wins = counts[counts >= 2].index
    d = d[d["window"].isin(valid_wins)]
    return d

def pbo_cscv(df: pd.DataFrame, windows=None, metric="sats_mult", n_samples=2048, rng_seed=42):
    """
    CSCV (Combinatorially Symmetric Cross-Validation):
    - Divide aleatoriamente las ventanas en IS/OOS (mitades) muchas veces.
    - Selecciona el mejor config por IS (mediana de la métrica).
    - Marca 'overfit' si su OOS < mediana OOS del conjunto.
    - PBO = frecuencia de 'overfit' ∈ [0,1].
    """
    rng = random.Random(rng_seed)
    d = _sanitize(df, windows, metric)
    if d.empty:
        raise ValueError("CSCV: datos insuficientes tras saneo (necesitas ≥2 configs por ventana).")

    # Matriz ventanas x configs
    piv = d.pivot_table(index="window", columns="config_id", values=metric, aggfunc="median")
    piv = piv.replace([np.inf, -np.inf], np.nan)

    W = [w for w in piv.index if piv.loc[w].notna().sum() >= 2]  # ventanas con ≥2 configs con dato
    if len(W) < 2:
        raise ValueError("CSCV: necesitas ≥2 ventanas válidas para partir IS/OOS.")

    overfits = 0
    valid_draws = 0

    for _ in range(n_samples):
        rng.shuffle(W)
        mid = max(1, len(W)//2)
        IS = W[:mid]
        OOS = W[mid:]
        if len(OOS) == 0:
            continue

        is_meds = piv.loc[IS].median(axis=0, skipna=True)
        # configs con algún dato en IS
        is_meds = is_meds.dropna()
        if is_meds.empty:
            continue

        # elegir mejor IS (si hay empate, lexicográfico para estabilidad)
        best_is_val = is_meds.max()
        top_is = sorted(is_meds[is_meds == best_is_val].index.tolist())[0]

        # rendimiento OOS del top IS
        oos_vals = piv.loc[OOS]
        oos_med_cfg = oos_vals[top_is].median(skipna=True)

        # baseline OOS (mediana entre configs en cada ventana, luego mediana global)
        oos_med_all = oos_vals.median(axis=1, skipna=True).median(skipna=True)

        if np.isnan(oos_med_cfg) or np.isnan(oos_med_all):
            continue

        valid_draws += 1
        if oos_med_cfg < oos_med_all:
            overfits += 1

    if valid_draws == 0:
        raise ValueError("CSCV: no hubo draws válidos (revisa que haya datos OOS para configs).")

    pbo = overfits / valid_draws
    return float(pbo), int(valid_draws)

def dsr_summary(df: pd.DataFrame):
    """
    Calcula un indicador DSR agregado:
    - Necesita columnas: ret_mean, ret_std, n_obs
    - Sharpe simple = ret_mean/ret_std * sqrt(n_obs)
    - Umbral deflactado ~ sqrt(log(N)/n_obs) (aprox sencilla)
    Devuelve: (ratio_configs_con_DSR_pos, N)
    """
    needed = {"ret_mean","ret_std","n_obs"}
    if not needed.issubset(set(c.lower() for c in df.columns.str.lower())):
        return None

    # normaliza nombres
    cols = {c.lower(): c for c in df.columns}
    d = df.copy()
    d["ret_mean"] = pd.to_numeric(d[cols["ret_mean"]], errors="coerce")
    d["ret_std"]  = pd.to_numeric(d[cols["ret_std"]],  errors="coerce")
    d["n_obs"]    = pd.to_numeric(d[cols["n_obs"]],    errors="coerce")
    d = d.dropna(subset=["ret_mean","ret_std","n_obs"])
    d = d[(d["ret_std"] > 0) & (d["n_obs"] > 1)]

    if d.empty:
        return None

    d["sharpe"] = d["ret_mean"] / d["ret_std"] * np.sqrt(d["n_obs"])
    N = max(1, d.shape[0])
    d["thr"] = np.sqrt(np.log(N) / d["n_obs"].clip(lower=1))
    d["dsr_pos"] = d["sharpe"] > d["thr"]
    ratio = float(d["dsr_pos"].mean())
    return ratio, int(N)

def append_md(out_md, lines):
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_md_append", required=True)
    ap.add_argument("--windows", nargs="+", default=None)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--n_samples", type=int, default=2048)
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)

    # PBO (CSCV)
    try:
        pbo, draws = pbo_cscv(df, windows=args.windows, metric="sats_mult", n_samples=args.n_samples)
        pbo_str = f"{pbo:.3f}"
    except Exception as e:
        pbo, draws = (np.nan, 0)
        pbo_str = f"N/A ({e})"

    # DSR (si hay columnas)
    dsr = dsr_summary(df)
    dsr_str = "N/A"
    if dsr is not None:
        dsr_str = f"{dsr[0]*100:.1f}% (N={dsr[1]})"

    lines = []
    lines.append("## Tests anti-overfitting (refuerzo CSCV)")
    lines.append(f"- **PBO/CSCV**: p̂ = {pbo_str}  *(n_samples={args.n_samples}, draws={draws}; ventanas: {', '.join(args.windows) if args.windows else 'auto'})*")
    lines.append(f"- **DSR**: {dsr_str}")
    lines.append("")
    lines.append(f"**Candidato:** `{args.candidate}`")
    lines.append("")

    append_md(args.out_md_append, lines)
    print("[OK] Tests anti-overfitting añadidos al Roadmap.")

if __name__ == "__main__":
    main()
