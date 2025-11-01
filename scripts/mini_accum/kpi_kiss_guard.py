#!/usr/bin/env python3
# KISS Guard (C1 tolerant) — v1.0
# - Soporta faltantes de mdd_vs_hodl (deriva si puede; si no, WARN y sigue)
# - fpy opcional (WARN si no está)
# - Permite overrides por env: OOS_2025H1_KPIS / OOS_KPI_GLOB
import argparse, csv, glob, json, os, sys

def read_csv_first_row(path):
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        row = next(r)
        return row, (r.fieldnames or [])

def pick_numeric(row, prefer):
    for k in prefer:
        v = row.get(k)
        if v is not None and str(v).strip() != "":
            try: return float(str(v))
            except: pass
    for k,v in row.items():
        try: return float(str(v))
        except: pass
    return None

def repo_root_from_manifest(manifest_path):
    return os.path.abspath(os.path.join(os.path.dirname(manifest_path), "../../.."))

def resolve_path(root, p):
    return p if os.path.isabs(p) else os.path.join(root, p)

def find_oos_kpi(manifest_path):
    p = os.environ.get("OOS_2025H1_KPIS") or os.environ.get("OOS_KPI_GLOB")
    if p:
        cands = glob.glob(p) if any(ch in p for ch in "*?[]") else [p]
        for c in cands:
            if os.path.isfile(c): return c
    try:
        with open(manifest_path, "r") as f: m = json.load(f)
        ws = m.get("windows") or []
        root = os.environ.get("ROOT") or repo_root_from_manifest(manifest_path)
        for w in ws:
            k = w.get("kpis")
            if k and "OOS_2025H1" in k:
                kp = resolve_path(root, k)
                if os.path.isfile(kp): return kp
    except Exception:
        pass
    root = os.environ.get("ROOT") or repo_root_from_manifest(manifest_path)
    for c in glob.glob(os.path.join(root, "reports/mini_accum/*_kpis__OOS_2025H1_*.csv")):
        if os.path.isfile(c): return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--oos-cut", default=None)
    ap.add_argument("--min-sats", type=float, default=1.00)
    ap.add_argument("--max-fpy",  type=float, default=26)
    ap.add_argument("--require-mdd", action="store_true")
    args = ap.parse_args()

    kpi_path = find_oos_kpi(args.manifest)
    print(f"[KISS INFO] KPI={kpi_path}")
    if not kpi_path:
        print("[KISS FAIL] no pude localizar KPI OOS_2025H1; define $OOS_2025H1_KPIS o $OOS_KPI_GLOB", file=sys.stderr)
        sys.exit(2)

    row, hdr = read_csv_first_row(kpi_path)

    sats = pick_numeric(row, ["sats_mult","net_btc_mult","netbtc_mult","net_btc_ratio"])
    if sats is None:
        print(f"[KISS FAIL] no pude leer sats_mult en {kpi_path}; columnas={hdr}", file=sys.stderr); sys.exit(2)

    ok = True
    if sats >= args.min_sats:
        print(f"[KISS PASS] sats_mult={sats:.6f} ≥ min_sats={args.min_sats:.2f}")
    else:
        print(f"[KISS FAIL] sats_mult={sats:.6f} < min_sats={args.min_sats:.2f}")
        ok = False

    mdd = None
    for k in ["mdd_vs_hodl","mdd_vs_hodl_ratio","mdd_ratio_vs_hodl","mdd_model_vs_hodl"]:
        v = row.get(k)
        if v is not None and str(v).strip() != "":
            try: mdd = float(str(v)); break
            except: pass
    if mdd is None:
        model = None; hodl = None
        for k in ["mdd_model_btc","mdd_model_usd","mdd_model","mdd_model_pct"]:
            v = row.get(k)
            if v is not None and str(v).strip() != "":
                try: model = abs(float(str(v))); break
                except: pass
        for k in ["mdd_hodl_btc","mdd_hodl_usd","mdd_hodl","mdd_hodl_pct"]:
            v = row.get(k)
            if v is not None and str(v).strip() != "":
                try: hodl = abs(float(str(v))); break
                except: pass
        if model is not None and hodl not in (None, 0.0):
            mdd = model / hodl
            print(f"[KISS WARN] mdd_vs_hodl derivado = {mdd:.6f} (mdd_model/mdd_hodl); columnas directas ausentes ({hdr})")
        else:
            if args.require_mdd:
                print("[KISS FAIL] require-mdd activo y mdd_vs_hodl no disponible/derivable", file=sys.stderr)
                ok = False
            else:
                print("[KISS WARN] mdd_vs_hodl no disponible ni derivable; omito chequeo de MDD")

    if mdd is not None:
        print(f"[KISS INFO]  mdd_vs_hodl={mdd:.6f}")

    fpy = None
    if "fpy" in row and str(row["fpy"]).strip() != "":
        try: fpy = float(str(row["fpy"]))
        except: pass
    if fpy is None:
        print("[KISS WARN] 'fpy' no presente; omito chequeo de FPY")
    else:
        if fpy <= args.max_fpy:
            print(f"[KISS PASS] fpy={fpy:.2f} ≤ max_fpy={args.max_fpy:.2f}")
        else:
            print(f"[KISS FAIL] fpy={fpy:.2f} > max_fpy={args.max_fpy:.2f}")
            ok = False

    print("[KISS GUARD PASS]" if ok else "[KISS GUARD FAIL]")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
