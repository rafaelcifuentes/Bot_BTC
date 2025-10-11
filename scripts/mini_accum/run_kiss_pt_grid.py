#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import json
import re
import hashlib
from datetime import datetime
from subprocess import run, PIPE

# --- Rutas base ---
ROOT = Path(__file__).resolve().parents[2]
KISS = ROOT / "scripts/mini_accum/kiss_v1.py"
CFG  = ROOT / "configs/mini_accum/kiss_v1.yaml"
REPORTS_DIR = ROOT / "reports/mini_accum"
KISS_ALIAS  = REPORTS_DIR / "kiss_v1"
KISS_ALIAS.mkdir(parents=True, exist_ok=True)

# --- CSV de ventanas walkforward (si existe, sobreescribe WINDOWS) ---
WINDOWS_CSV = ROOT / "scripts/mini_accum/windows_walkforward.csv"

# --- Ventanas WF (ajústalas si quieres) ---
WINDOWS = [
    ("WF_2022", "2022-05-15", "2022-12-31"),
    ("WF_2023", "2023-01-01", "2023-12-31"),
    ("WF_2024", "2024-01-01", "2024-09-19"),  # pon aquí el último día disponible
]

# --- Helpers para cargar ventanas, reproducibilidad y parseo de sufijos ---
def load_windows():
    """Load windows either from CSV (if present) or fallback to the in-file WINDOWS list."""
    if WINDOWS_CSV.exists():
        df = pd.read_csv(WINDOWS_CSV)
        # normalize column names
        df.columns = [c.lower() for c in df.columns]
        required = {"window", "start", "end"}
        if not required.issubset(set(df.columns)):
            raise SystemExit(f"[run_kiss_pt_grid] WINDOWS CSV debe tener columnas {sorted(required)}. Archivo: {WINDOWS_CSV}")
        return [(str(r["window"]), str(r["start"]), str(r["end"])) for _, r in df.iterrows()]
    return WINDOWS

def git_commit_short() -> str:
    try:
        r = run(["git", "rev-parse", "--short", "HEAD"], stdout=PIPE, stderr=PIPE, text=True, cwd=str(ROOT))
        return r.stdout.strip() if r.returncode == 0 else "NA"
    except Exception:
        return "NA"

def sha256_file(p: Path) -> str:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "NA"

def parse_params_from_suffix(suffix: str) -> dict:
    m = re.search(r"PT_G(?P<g>\d+)_DD(?P<dd>\d+)_RB(?P<rb>\d+)_H(?P<h>\d+)_BULL(?P<bull>\d+)__", suffix or "")
    if not m:
        return {}
    d = m.groupdict()
    return {k: int(v) for k, v in d.items()}

# --- Grid de parámetros alrededor del TOP ---
DDS = [14, 15]        # drawdown desde pico para SELL
RBS = [1, 2]          # rebote desde valle para BUY
HS  = [30, 31, 32]    # dd_hard (hard stop)
GATES      = [200, 150]  # añadimos G150 al barrido
GATE_MODE  = "sell"
BULL_HOLD  = 0

def run_once(W, start, end, dd, rb, h, gate_sma) -> str:
    """Corre kiss_v1.py en modo PT para un combo de parámetros y ventana."""
    suffix = f"PT_G{gate_sma}_DD{dd}_RB{rb}_H{h}_BULL0__{W}"
    cmd = [
        sys.executable, str(KISS),
        "--config", str(CFG),
        "--mode", "pt",
        "--gate_sma", str(gate_sma), "--gate_mode", GATE_MODE,
        "--bull_hold_sma", str(BULL_HOLD),
        "--dd_pct", str(dd), "--rb_pct", str(rb), "--dd_hard_pct", str(h),
        "--start", start, "--end", end,
        "--suffix", suffix,
    ]
    res = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
    # Mostrar salida de kiss_v1 (paths [OK] y KPIs)
    out = (res.stdout or "").strip()
    if out:
        print(out)
    if res.returncode != 0:
        err = (res.stderr or "").strip()
        print(f"[ERR] {err}")
    return suffix

def latest_kpis_for_suffix(suffix: str) -> Path | None:
    patt = str(REPORTS_DIR / f"base_v0_1_*_kpis__{suffix}.csv")
    files = glob.glob(patt)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return Path(files[-1])

def kpis_to_metrics(kpis_path: Path) -> dict:
    df = pd.read_csv(kpis_path)
    r = df.iloc[0]
    return {
        "sats_mult": float(r.get("net_btc_vs_hodl", 0.0)),
        "mdd_vs_hodl": float(r.get("mdd_vs_hodl_ratio", 0.0)),
        "fpy": float(r.get("flips_per_year", 0.0)),
        "flips_total": int(r.get("flips_total", 0)),
        "file": str(kpis_path),
    }

def pick_best_for_window(W: str) -> tuple[dict | None, pd.DataFrame]:
    """Elige el mejor por sats_mult (tie-break: mdd_vs_hodl menor, luego fpy menor)."""
    rows = []
    for dd in DDS:
        for rb in RBS:
            for h in HS:
                for g in GATES:
                    suffix = f"PT_G{g}_DD{dd}_RB{rb}_H{h}_BULL0__{W}"
                    k = latest_kpis_for_suffix(suffix)
                    if not k:
                        continue
                    m = kpis_to_metrics(k)
                    m["suffix"] = suffix
                    rows.append(m)
    if not rows:
        return None, pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["sats_mult", "mdd_vs_hodl", "fpy"], ascending=[False, True, True]).reset_index(drop=True)
    return df.iloc[0].to_dict(), df

def write_alias(best: dict, W: str, table: pd.DataFrame, window_meta: tuple[str, str]) -> None:
    """Escribe alias KISS + candidatos + manifiesto reproducible."""
    k_src = Path(best["file"])
    # equity correspondiente al mismo run_id + suffix
    e_src = k_src.with_name(k_src.name.replace("_kpis__", "_equity__"))

    # Mapear columnas al formato KISS esperado por show_netbtc_summary.py
    src = pd.read_csv(k_src).iloc[0]
    fpy_val = float(src.get("flips_per_year", src.get("fpy", float("nan"))))
    out = pd.DataFrame([{
        "sats_mult": float(src.get("net_btc_vs_hodl", 0.0)),
        "mdd_vs_hodl": float(src.get("mdd_vs_hodl_ratio", 0.0)),
        "fpy": fpy_val,                   # compatibilidad con show_netbtc_summary
        "flips_per_year": fpy_val,        # duplicado para otras herramientas
        "flips_total": int(src.get("flips_total", 0)),
    }])

    # Destinos
    k_dst = KISS_ALIAS / f"{W}_kpis__v1_2.csv"
    e_dst = KISS_ALIAS / f"{W}_equity__v1_2.csv"
    cand_dst = KISS_ALIAS / f"{W}_candidates__v1_2.csv"
    man_dst = KISS_ALIAS / f"{W}_manifest__v1_2.json"

    # Alias KPIs
    out.to_csv(k_dst, index=False)

    # Equity
    if e_src.exists():
        shutil.copyfile(e_src, e_dst)
    else:
        print(f"[WARN] Equity faltante para {W}: {e_src.name}")

    # Candidatos
    if not table.empty:
        table.to_csv(cand_dst, index=False)

    # Manifiesto reproducible
    start, end = window_meta
    manifest = {
        "window": W,
        "start": start,
        "end": end,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit_short(),
        "config_yaml": str(CFG),
        "config_sha256": sha256_file(CFG),
        "alias_kpis": str(k_dst),
        "alias_equity": str(e_dst) if e_dst.exists() else None,
        "candidates_csv": str(cand_dst) if cand_dst.exists() else None,
        "best": {
            "suffix": best.get("suffix"),
            "params": parse_params_from_suffix(best.get("suffix")),
            "sats_mult": float(out["sats_mult"].iloc[0]),
            "mdd_vs_hodl": float(out["mdd_vs_hodl"].iloc[0]),
            "fpy": float(out["fpy"].iloc[0]),
            "flips_total": int(out["flips_total"].iloc[0]),
        },
        "grid": {
            "DDS": DDS,
            "RBS": RBS,
            "HS": HS,
            "GATES": GATES,
            "GATE_MODE": GATE_MODE,
            "BULL_HOLD": BULL_HOLD,
        },
    }
    with open(man_dst, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"[KISS] {W}: sats_mult={out['sats_mult'].iloc[0]:.6f}  "
        f"mdd_vs_hodl={out['mdd_vs_hodl'].iloc[0]:.3f}  "
        f"fpy={out['fpy'].iloc[0]:.2f}  flips={out['flips_total'].iloc[0]}"
    )
    print(f"      → {k_dst.name}  /  {e_dst.name if e_dst.exists() else 'equity missing'}")

def netbtc_product():
    prod = 1.0
    rows = []
    for W in ["WF_2022", "WF_2023", "WF_2024"]:
        k = KISS_ALIAS / f"{W}_kpis__v1_2.csv"
        if not k.exists():
            continue
        s = float(pd.read_csv(k)["sats_mult"].iloc[0])
        prod *= s
        rows.append((W, s))
    print("\n== NetBTC (producto por ventana) ==")
    for w, s in rows:
        print(f"{w}: {s:.6f}")
    print(f"Producto = {prod:.6f}")

def main():
    wfs = load_windows()

    # 1) Barrido por ventana
    for (W, start, end) in wfs:
        print(f"\n== GRID {W}  {start}..{end} ==")
        for dd in DDS:
            for rb in RBS:
                for h in HS:
                    for g in GATES:
                        run_once(W, start, end, dd, rb, h, g)

    # 2) Auto-pick y alias + artefactos
    for (W, start, end) in wfs:
        best, table = pick_best_for_window(W)
        if not best:
            print(f"[WARN] Sin resultados para {W}")
            continue
        write_alias(best, W, table, (start, end))

    # 3) NetBTC final
    netbtc_product()

if __name__ == "__main__":
    main()