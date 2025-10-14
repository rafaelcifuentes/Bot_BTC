#!/usr/bin/env python3
# scripts/mini_accum/dev/dev.py
from __future__ import annotations

import argparse, os, sys, glob, subprocess, datetime as dt, csv
import yaml

# =============================
# KPI loader estricto (anti-NaN)
# Acepta múltiples llaves de salida para compatibilidad entre versiones.
# =============================
_PF_SATS_KEYS = (
    "sats_mult", "netBTC", "net_btc_ratio", "net_btc",
    "final_equity_btc_ratio", "equity_btc_ratio", "equity_btc",
)
_PF_MDD_KEYS  = ("mdd_vs_hodl", "mdd_vs_hodl_ratio", "mdd_v_hodl", "mdd_model_vs_hodl")
_PF_FLIP_KEYS = ("flips_total", "flips", "total_flips")
_PF_FPY_KEYS  = ("flips_per_year", "fpy")

def _to_float_safe(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def read_kpi_csv_strict(path: str):
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        row = next(r, {}) or {}
    norm = {(k or "").strip(): (v or "").strip() for k, v in row.items()}

    def _pick(keys):
        for k in keys:
            if k in norm and norm[k] != "":
                v = _to_float_safe(norm[k])
                if v is not None:
                    return v, k
        return None, None

    sats,  sats_key   = _pick(_PF_SATS_KEYS)
    mdd,   mdd_key    = _pick(_PF_MDD_KEYS)
    flipsv, flips_key = _pick(_PF_FLIP_KEYS)
    fpyv,  fpy_key    = _pick(_PF_FPY_KEYS)

    try:
        flips = int(float(flipsv)) if flipsv is not None else 0
    except Exception:
        flips = 0

    return {
        "sats": sats, "sats_key": sats_key,
        "mdd": mdd,   "mdd_key":  mdd_key,
        "flips": flips, "flips_key": flips_key,
        "fpy": fpyv,  "fpy_key": fpy_key,
        "raw": norm, "path": path,
    }


def pick_latest(pattern: str) -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    # Ordenar por mtime para "último generado" (coincide con helpers actuales)
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]



def pct(x: float) -> str:
    return f"{x*100:.2f}%" if x is not None else "n/a"


def _parse_date(s: str) -> dt.date:
    return dt.datetime.fromisoformat(s).date()


def _years_between(start: str, end: str) -> float:
    d0, d1 = _parse_date(start), _parse_date(end)
    days = (d1 - d0).days
    return max(0.0, days) / 365.25


def _top_missing_keys(cfg: dict) -> list[str]:
    # TOP (v1) requerido por Superset: DD15 / RB1 / H30 / G200 / BULL0 + tag
    req = [
        ("risk", "hard_dd_pct"),
        ("rebalancing", "frequency"),
        ("horizon", "h_bars"),
        ("execution", "gamma_bps"),
        ("execution", "bull_bias_bps"),
        (None, "tag"),
    ]
    miss = []
    for parent, key in req:
        if parent is None:
            if key not in cfg or cfg.get(key) in (None, ""):
                miss.append(key)
        else:
            blk = cfg.get(parent) or {}
            if key not in blk or blk.get(key) in (None, ""):
                miss.append(f"{parent}.{key}")
    return miss


def main():
    ap = argparse.ArgumentParser(description="Mini-Accum v2 Gate & Contrato runner (opt-in)")
    ap.add_argument("--config", required=True, help="Ruta YAML v2.x")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--suffix", required=True, help="Sufijo para rename de artefactos")
    ap.add_argument("--strict", action="store_true", help="Evalúa checks estrictos si hay insumos")
    ap.add_argument("--no-run", action="store_true", help="No corre backtest; sólo evalúa gate")
    ap.add_argument("--write-docs", action="store_true", help="Append a docs/mini_accum/*.md")
    ap.add_argument("--base-kpi", default=os.environ.get("BASE_KPI", ""), help="Ruta explícita KPI BASE (opcional)")
    ap.add_argument("--base-glob", default=os.environ.get("BASE_KPI_GLOB", "reports/mini_accum/*_kpis__OOS_2025H1_G200_DD15_RB1_H30_BULL0.csv"),
                    help="Glob alternativo para encontrar BASE si no se pasa --base-kpi")
    args = ap.parse_args()

    # Carga YAML + verificación TOP (superset)
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    if args.strict:
        missing = _top_missing_keys(cfg)
        if missing:
            print(f"[TOP] FAIL: faltan claves requeridas en {args.config}: {', '.join(missing)}")
            sys.exit(2)

    rep_dir = (cfg.get("backtest") or {}).get("reports_dir", "reports/mini_accum")
    os.makedirs(rep_dir, exist_ok=True)

    # Corre el runner de mini_accum (opt-in por preset) si aplica
    if not args.no_run:
        cmd = [
            sys.executable, "-m", "mini_accum.cli",
            "--config", args.config,
            "--start", args.start, "--end", args.end,
            "--suffix", args.suffix,
        ]
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # KPI CANDIDATO: último *_kpis__<SUF>.csv en reports_dir
    cand_glob = os.path.join(rep_dir, f"*_kpis__{args.suffix}.csv")
    cand_path = pick_latest(cand_glob)

    # KPI BASE: prioridad a --base-kpi, luego --base-glob
    base_path = args.base_kpi if args.base_kpi else pick_latest(args.base_glob)

    if not cand_path:
        print(f"[GATE] cand no existe: {cand_glob}")
        sys.exit(1)
    if not base_path:
        print(f"[GATE] base no existe: {args.base_glob}")
        sys.exit(1)

    cand = read_kpi_csv_strict(cand_path)
    base = read_kpi_csv_strict(base_path)

    # 5) Anti-NaN en KPIs
    if cand.get("sats") is None or base.get("sats") is None:
        print("[ASSERT] FAIL: KPI sin sats en BASE o CAND.")
        print(f"  BASE={base_path} key={base.get('sats_key')}  CAND={cand_path} key={cand.get('sats_key')}")
        sys.exit(2)

    # Duración → FPY si no viene en KPI
    years = _years_between(args.start, args.end)
    fpy_base = base.get("fpy") if base.get("fpy") is not None else (base.get("flips", 0) / years if years > 0 else None)
    fpy_cand = cand.get("fpy") if cand.get("fpy") is not None else (cand.get("flips", 0) / years if years > 0 else None)

    # Aviso de alineación de ventanas: si la BASE es 2025H1 y el CAND usa un end > 2025-06-30
    try:
        base_name = os.path.basename(base_path)
        if "2025H1" in base_name:
            end_d = _parse_date(args.end)
            h1_end = dt.date(2025, 6, 30)
            if end_d > h1_end:
                print(f"[WARN] Comparando CAND (end={end_d}) contra BASE 2025H1 (end=2025-06-30). Para apples-to-apples, considera usar --end 2025-06-30.")
    except Exception:
        pass

    print(
        f"[BASE] sats={base['sats']:.6f} (key={base.get('sats_key')})  "
        f"mdd={(base.get('mdd') if base.get('mdd') is not None else float('nan')):.6f} (key={base.get('mdd_key')})  "
        f"flips={base.get('flips', 0)} (key={base.get('flips_key')})  "
        f"fpy={(fpy_base if fpy_base is not None else float('nan')):.2f} (key={base.get('fpy_key')})  "
        f"file={base_path}"
    )
    print(
        f"[CAND] sats={cand['sats']:.6f} (key={cand.get('sats_key')})  "
        f"mdd={(cand.get('mdd') if cand.get('mdd') is not None else float('nan')):.6f} (key={cand.get('mdd_key')})  "
        f"flips={cand.get('flips', 0)} (key={cand.get('flips_key')})  "
        f"fpy={(fpy_cand if fpy_cand is not None else float('nan')):.2f} (key={cand.get('fpy_key')})  "
        f"file={cand_path}"
    )

    # 3) Lift y 4) Riesgo
    lift = cand["sats"]/base["sats"] - 1.0
    risk_ok = (
        base.get("mdd") is not None and cand.get("mdd") is not None and cand["mdd"] <= base["mdd"]
    )

    # 7) Fricción operativa: si FPY_cand duplica BASE y no hay +5% lift ⇒ FAIL
    fpy_ok = True
    if fpy_base is not None and fpy_cand is not None:
        fpy_ok = not (fpy_cand > 2.0 * fpy_base and lift < 0.05)

    # 6) Estricto: Spearman / PBO por ENV (si están disponibles)
    spearman_ok = None
    pbo_ok = None
    if args.strict:
        try:
            rho = float(os.environ.get("SPEARMAN_RHO", "")) if os.environ.get("SPEARMAN_RHO") else None
            rho_min = float(os.environ.get("SPEARMAN_MIN", 0.95))
            if rho is not None:
                spearman_ok = (rho >= rho_min)
        except Exception:
            spearman_ok = False
        try:
            pbo_val = float(os.environ.get("PBO_VAL", "")) if os.environ.get("PBO_VAL") else None
            pbo_max = float(os.environ.get("PBO_MAX", 0.30))
            if pbo_val is not None:
                pbo_ok = (pbo_val <= pbo_max)
        except Exception:
            pbo_ok = False

    mdd_delta = (cand.get('mdd') - base.get('mdd')) if (cand.get('mdd') is not None and base.get('mdd') is not None) else float('nan')
    print(f"[DIFF] lift={pct(lift)}  mdd_delta={mdd_delta:+.6f}")
    print(f"[ROBUST] spearman {('OK' if spearman_ok else 'SKIP' if spearman_ok is None else 'FAIL')}  "
          f"PBO {('OK' if pbo_ok else 'SKIP' if pbo_ok is None else 'FAIL')}")

    pass_lift = (lift >= 0.05)
    ok = pass_lift and risk_ok and fpy_ok
    print(f"[GATE] {'PASS' if ok else 'FAIL'}: lift≥5%={pass_lift}  risk_ok={risk_ok}  fpy_ok={fpy_ok}")

    # 8) Docs y trazabilidad
    if args.write_docs:
        today = dt.date.today().isoformat()
        os.makedirs("docs/mini_accum", exist_ok=True)
        resumen = [
            f"## {today} — Gate {'PASS' if ok else 'FAIL'} {args.suffix}",
            f"- BASE: {base_path}",
            f"- CAND: {cand_path}",
            f"- Resultado: **{'PASS' if ok else 'FAIL'}** (lift {pct(lift)}; ΔMDD={mdd_delta:+.6f})",
            f"- Métricas: sats_BASE={base['sats']:.6f}, sats_CAND={cand['sats']:.6f}, "
            f"mdd_BASE={base.get('mdd') if base.get('mdd') is not None else float('nan'):.6f}, "
            f"mdd_CAND={cand.get('mdd') if cand.get('mdd') is not None else float('nan'):.6f}, "
            f"flips: base={base.get('flips', 0)} cand={cand.get('flips', 0)}; fpy: base={fpy_base if fpy_base is not None else float('nan'):.2f} cand={fpy_cand if fpy_cand is not None else float('nan'):.2f}",
            f"- Decisión: {'promover' if ok else 'mantener BASE; candidato OFF (opt-in)'}",
            "",
        ]
        for f in ("docs/mini_accum/Progreso.md", "docs/mini_accum/decisiones.md"):
            with open(f, "a", encoding="utf-8") as fh:
                fh.write("\n" + "\n".join(resumen))

    # 9) Exit code útil en CI
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()