#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit Diamante Edge — v0.2
Lee signals/diamante.csv, construye PnL bruto (exposición unitaria según sD)
y compara contra buy&hold del subyacente (retorno de BTC que venga en el CSV
o derivado de close/price). Imprime resumen y guarda:
  - reports/heart/diamante_audit.md
  - reports/heart/eq_diamante_raw.csv
  - reports/heart/eq_buyhold.csv
Robusto a columnas faltantes/planas.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import math

# --------- utils ----------
def _ensure_utc_index(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    if ts_col in df.columns:
        idx = pd.to_datetime(df[ts_col], utc=True)
        df = df.drop(columns=[ts_col])
        df.index = idx
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("4h").last().ffill()

def _coerce(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _derive_returns(df: pd.DataFrame) -> pd.Series:
    for cand in ["retD_btc","ret_btc","ret","returns","r"]:
        if cand in df.columns:
            s = pd.to_numeric(df[cand], errors="coerce").fillna(0.0)
            if s.std(skipna=True) > 0: return s
    for px in ["close","price","px","Close"]:
        if px in df.columns:
            s = pd.to_numeric(df[px], errors="coerce").pct_change().fillna(0.0)
            if s.std(skipna=True) > 0: return s
    return pd.Series(0.0, index=df.index)

def ann_vol(r: pd.Series, bars_per_year=6*365) -> float:
    return float(r.std(ddof=0) * math.sqrt(bars_per_year))

def mdd_from_eq(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = eq/peak - 1.0
    return float(dd.min()) if len(dd) else float("nan")

def profit_factor(r: pd.Series) -> float:
    g = r[r>0].sum()
    l = -r[r<0].sum()
    if l <= 0:
        return float("inf") if g>0 else 0.0
    return float(g / l)

def sortino(r: pd.Series, bars_per_year=6*365) -> float:
    mu = r.mean()
    dn = r[r<0].std(ddof=0)
    if dn == 0 or pd.isna(dn): return float("inf") if mu>0 else 0.0
    return float((mu * math.sqrt(bars_per_year)) / dn)

def turnover_from_signal(sig: pd.Series) -> float:
    # soporte sD ∈ {-1,0,1} o {0,1}; normalizamos a exposición en [-1,1]
    s = pd.to_numeric(sig, errors="coerce").fillna(0.0)
    u = s.copy()
    if set(s.dropna().unique()).issubset({0.0,1.0}):  # binaria
        u = (s > 0).astype(float)
    delta = u.diff().abs().fillna(0.0)
    return float(delta.sum())  # “cambios de estado” acumulados

# --------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diamante", default="signals/diamante.csv")
    ap.add_argument("--out_dir", default="reports/heart/")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.diamante)
    df = _ensure_utc_index(df)
    df = _resample_4h(df)
    df = _coerce(df, ["sD","w_diamante_raw","retD_btc","ret","returns","r","close","price"])

    # Señal
    if "sD" in df.columns:
        sD_raw = df["sD"].fillna(0.0)
        if set(sD_raw.dropna().unique()).issubset({0.0,1.0}):
            expo = (sD_raw > 0).astype(float)   # long/flat
        else:
            expo = sD_raw.clip(-1,1)            # long/short/flat
    else:
        expo = pd.Series(1.0, index=df.index)   # fallback buy&hold

    # Retornos de mercado
    r_mkt = _derive_returns(df).astype(float)

    # PnL bruto de Diamante (unidad de exposición)
    r_edge = expo * r_mkt
    eq_edge = (1 + r_edge.fillna(0.0)).cumprod()
    eq_bh   = (1 + r_mkt.fillna(0.0)).cumprod()

    # Métricas globales
    def kpis(r, eq):
        return {
            "NET": float(eq.iloc[-1] - 1) if len(eq) else float("nan"),
            "MDD": mdd_from_eq(eq),
            "VOL": ann_vol(r),
            "PF":  profit_factor(r),
            "WR":  float((r>0).mean()) if len(r) else float("nan"),
            "Sortino": sortino(r),
        }

    K_edge = kpis(r_edge, eq_edge)
    K_bh   = kpis(r_mkt, eq_bh)

    # Regímenes por volatilidad realizada del mercado (mediana)
    vol_ewm = r_mkt.ewm(span=60, adjust=False).std()
    med = float(vol_ewm.median())
    low_mask  = vol_ewm <= med
    high_mask = vol_ewm >  med

    def kpi_mask(mask):
        rr = r_edge.where(mask).fillna(0.0)
        ee = (1+rr).cumprod()
        return kpis(rr, ee)

    K_low  = kpi_mask(low_mask)
    K_high = kpi_mask(high_mask)

    # Turnover aproximado de la señal (cambios de estado)
    to_total = turnover_from_signal(df.get("sD", pd.Series(index=df.index)))

    # Guardar curvas y reporte
    eq_edge.to_frame("equity_diamante_raw").to_csv(out_dir/"eq_diamante_raw.csv",
                                                   date_format="%Y-%m-%d %H:%M:%S%z")
    eq_bh.to_frame("equity_buyhold").to_csv(out_dir/"eq_buyhold.csv",
                                            date_format="%Y-%m-%d %H:%M:%S%z")

    md = out_dir/"diamante_audit.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Auditoría de Edge — Diamante (bruto)\n\n")
        f.write("## Métricas globales\n\n")
        f.write("| Serie | NET | MDD | VOL | PF | WR | Sortino |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        f.write(f"| Diamante (bruto) | {K_edge['NET']:.4f} | {K_edge['MDD']:.2%} | {K_edge['VOL']:.4f} | {K_edge['PF']:.2f} | {K_edge['WR']:.2%} | {K_edge['Sortino']:.2f} |\n")
        f.write(f"| Buy&Hold         | {K_bh['NET']:.4f} | {K_bh['MDD']:.2%} | {K_bh['VOL']:.4f} | {K_bh['PF']:.2f} | {K_bh['WR']:.2%} | {K_bh['Sortino']:.2f} |\n\n")

        f.write("## Por regímenes de volatilidad (mediana del EWM std 60)\n\n")
        f.write("| Régimen | NET | MDD | VOL | PF | WR | Sortino |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        f.write(f"| Vol baja | {K_low['NET']:.4f} | {K_low['MDD']:.2%} | {K_low['VOL']:.4f} | {K_low['PF']:.2f} | {K_low['WR']:.2%} | {K_low['Sortino']:.2f} |\n")
        f.write(f"| Vol alta | {K_high['NET']:.4f} | {K_high['MDD']:.2%} | {K_high['VOL']:.4f} | {K_high['PF']:.2f} | {K_high['WR']:.2%} | {K_high['Sortino']:.2f} |\n\n")

        f.write("## Otros\n\n")
        f.write(f"- Cambios de estado de la señal (aprox turnover): {to_total:.0f}\n")
        f.write(f"- Barras totales: {len(df):d}\n")

    # Consola
    print("=== DIAMANTE RAW EDGE ===")
    print("--- Global ---")
    print(f"NET raw      : {K_edge['NET']:.6f}")
    print(f"MDD raw      : {K_edge['MDD']:.2%}")
    print(f"VOL raw      : {K_edge['VOL']:.4f}")
    print(f"PF / WR      : {K_edge['PF']:.2f} / {K_edge['WR']:.2%}")
    print(f"Sortino      : {K_edge['Sortino']:.2f}")
    print("--- Buy&Hold ---")
    print(f"NET BH       : {K_bh['NET']:.6f} | MDD {K_bh['MDD']:.2%} | VOL {K_bh['VOL']:.4f}")
    print("--- Regímenes (vol EWM60 mediana) ---")
    print(f"LOW-vol  NET : {K_low['NET']:.6f} | PF {K_low['PF']:.2f} | WR {K_low['WR']:.2%}")
    print(f"HIGH-vol NET : {K_high['NET']:.6f} | PF {K_high['PF']:.2f} | WR {K_high['WR']:.2%}")
    print(f"Turnover aprox (cambios estado sD): {to_total:.0f}")
    print("[MD] reports/heart/diamante_audit.md")
    print("[CSV] reports/heart/eq_diamante_raw.csv, reports/heart/eq_buyhold.csv")

if __name__ == "__main__":
    main()