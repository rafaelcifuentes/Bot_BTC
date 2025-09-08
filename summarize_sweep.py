#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
summarize_sweep.py

Genera un resumen (CSV, Markdown y JSON "best") a partir de:
  a) un índice CSV con filas: threshold,adx1d_min,adx4_min,summary_path
  b) o un patrón glob con los JSON de summary.

Robusto a distintos esquemas de summary:
- Si hay métricas numéricas (p.ej. "metrics180", "selected_180", etc.), las usa.
- Si no, extrae del texto (líneas tipo "90d  base → Net ..., PF ..., ...").
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from glob import glob
from typing import Any, Dict, List, Tuple, Optional

try:
    import pandas as pd
except Exception:
    print("ERROR: se requiere pandas para ejecutar este script.", file=sys.stderr)
    raise

# ---------- Utilidades ----------

# Acepta '-' normal y '−' unicode; números con signo y decimales.
NUM = r"[+\-\u2212]?\d+(?:\.\d+)?"

RE_BASE_90 = re.compile(
    rf"(?is)\b(?:90d\s*base|base\s*90d|90d\s*pick)\b.*?"
    rf"Net\s*({NUM})\s*,\s*PF\s*({NUM})\s*,\s*Win%.*?({NUM})\s*,\s*Trades\s*(\d+)\s*,\s*MDD\s*({NUM})\s*,\s*Score\s*({NUM})"
)

RE_BASE_180 = re.compile(
    rf"(?is)\b(?:180d\s*base|base\s*180d|180d\s*pick)\b.*?"
    rf"Net\s*({NUM})\s*,\s*PF\s*({NUM})\s*,\s*Win%.*?({NUM})\s*,\s*Trades\s*(\d+)\s*,\s*MDD\s*({NUM})\s*,\s*Score\s*({NUM})"
)

RE_HOLDOUT = re.compile(
    rf"(?is)\b(?:Holdout\s*\d+\s*d|Holdout)\b.*?"
    rf"Net\s*({NUM})\s*,\s*PF\s*({NUM})\s*,\s*Win%.*?({NUM})\s*,\s*Trades\s*(\d+)\s*,\s*MDD\s*({NUM})\s*,\s*Score\s*({NUM})"
)

def as_float(x: Any) -> float:
    try:
        if x is None:
            return math.nan
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        s = s.replace("−", "-").replace("%", "")
        return float(s)
    except Exception:
        return math.nan

def as_int(x: Any) -> float:
    try:
        if x is None:
            return math.nan
        if isinstance(x, int):
            return int(x)
        s = str(x).strip().replace("−", "-")
        return int(float(s))
    except Exception:
        return math.nan

def collect_strings(obj: Any) -> List[str]:
    """Recorre recursivamente el JSON y junta TODOS los strings."""
    out: List[str] = []
    if obj is None:
        return out
    if isinstance(obj, str):
        out.append(obj)
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(collect_strings(v))
        return out
    if isinstance(obj, list):
        for v in obj:
            out.extend(collect_strings(v))
        return out
    return out

def recursive_numeric_lookup(j: Any, key_candidates: List[str]) -> Optional[float]:
    """Busca recursivamente un valor numérico por lista de keys candidatas."""
    def walk(obj: Any) -> Optional[float]:
        if isinstance(obj, dict):
            # match directo
            for k, v in obj.items():
                kl = str(k).lower()
                for cand in key_candidates:
                    if cand in kl:
                        f = as_float(v)
                        if not math.isnan(f):
                            return f
            # navegación
            for v in obj.values():
                r = walk(v)
                if r is not None and not math.isnan(r):
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = walk(v)
                if r is not None and not math.isnan(r):
                    return r
        return None
    return walk(j)

def parse_from_text(blob: str) -> Dict[str, Any]:
    """Extrae métricas desde texto usando regex."""
    out: Dict[str, Any] = {}

    m90 = RE_BASE_90.search(blob)
    if m90:
        out.update({
            "net90": as_float(m90.group(1)),
            "pf90": as_float(m90.group(2)),
            "win90": as_float(m90.group(3)),
            "trades90": as_int(m90.group(4)),
            "mdd90": as_float(m90.group(5)),
            "score90": as_float(m90.group(6)),
        })

    m180 = RE_BASE_180.search(blob)
    if m180:
        out.update({
            "net180": as_float(m180.group(1)),
            "pf180": as_float(m180.group(2)),
            "win180": as_float(m180.group(3)),
            "trades180": as_int(m180.group(4)),
            "mdd180": as_float(m180.group(5)),
            "score180": as_float(m180.group(6)),
        })

    mh = RE_HOLDOUT.search(blob)
    if mh:
        out.update({
            "holdout_net": as_float(mh.group(1)),
            "holdout_pf": as_float(mh.group(2)),
            "holdout_win": as_float(mh.group(3)),
            "holdout_trades": as_int(mh.group(4)),
            "holdout_mdd": as_float(mh.group(5)),
            "holdout_score": as_float(mh.group(6)),
        })

    return out

def parse_summary_json(path: str) -> Dict[str, Any]:
    """Devuelve dict con métricas normalizadas, intentando:
       1) campos numéricos
       2) extracción por texto
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        # si el archivo no es JSON válido, devolvemos dict vacío (se filtrará luego)
        return {}

    result: Dict[str, Any] = {}

    # 1) lookup numérico por keys candidatas (por si el runner guardó estructura)
    # 90d
    result.setdefault("net90",  recursive_numeric_lookup(j, ["net90", "90d", "base90"]))
    result.setdefault("pf90",   recursive_numeric_lookup(j, ["pf90", "90d", "base90"]))
    result.setdefault("score90",recursive_numeric_lookup(j, ["score90", "score_90", "90d"]))
    result.setdefault("mdd90",  recursive_numeric_lookup(j, ["mdd90", "90d"]))
    result.setdefault("trades90", recursive_numeric_lookup(j, ["trades90", "trades_90", "90d"]))
    result.setdefault("win90",  recursive_numeric_lookup(j, ["win90", "win_90", "win%"]))

    # 180d
    result.setdefault("net180",  recursive_numeric_lookup(j, ["net180", "180d", "base180"]))
    result.setdefault("pf180",   recursive_numeric_lookup(j, ["pf180", "180d", "base180"]))
    result.setdefault("score180",recursive_numeric_lookup(j, ["score180", "score_180", "180d"]))
    result.setdefault("mdd180",  recursive_numeric_lookup(j, ["mdd180", "180d"]))
    result.setdefault("trades180", recursive_numeric_lookup(j, ["trades180", "trades_180", "180d"]))
    result.setdefault("win180",  recursive_numeric_lookup(j, ["win180", "win_180", "win%"]))

    # holdout
    result.setdefault("holdout_net", recursive_numeric_lookup(j, ["holdout_net", "oos", "holdout"]))
    result.setdefault("holdout_pf",  recursive_numeric_lookup(j, ["holdout_pf", "oos", "holdout"]))
    result.setdefault("holdout_score", recursive_numeric_lookup(j, ["holdout_score", "oos", "holdout"]))
    result.setdefault("holdout_mdd", recursive_numeric_lookup(j, ["holdout_mdd", "oos", "holdout"]))
    result.setdefault("holdout_trades", recursive_numeric_lookup(j, ["holdout_trades", "oos", "holdout"]))
    # timestamp textual no es fiable; se usa mtime del archivo si falta
    # result.setdefault("timestamp_utc", recursive_numeric_lookup(j, ["timestamp", "utc", "time"]))

    # 2) extracción desde texto si faltan campos o si no hubo nada
    need_text = (not result) or any(
        (k in result) and (result.get(k) is None or (isinstance(result.get(k), float) and math.isnan(result.get(k))))
        for k in ["net90","pf90","score90","mdd90","trades90","win90",
                  "net180","pf180","score180","mdd180","trades180","win180",
                  "holdout_net","holdout_pf","holdout_score","holdout_mdd","holdout_trades"]
    )

    if need_text:
        blob = "\n".join(collect_strings(j))
        parsed = parse_from_text(blob)
        # solo pisa si estaba vacío o NaN
        for k, v in parsed.items():
            if k not in result or result[k] is None or (isinstance(result[k], float) and math.isnan(result[k])):
                result[k] = v

    # Normaliza NaNs a float
    for k in ["net90","pf90","score90","mdd90","trades90","win90",
              "net180","pf180","score180","mdd180","trades180","win180",
              "holdout_net","holdout_pf","holdout_score","holdout_mdd","holdout_trades"]:
        result[k] = as_float(result.get(k))

    # timestamp: si no vino, toma mtime del archivo
    ts = result.get("timestamp_utc", None)
    if ts is None or (isinstance(ts, float) and math.isnan(ts)) or (isinstance(ts, str) and not ts.strip()):
        try:
            mtime = os.path.getmtime(path)
            result["timestamp_utc"] = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            result["timestamp_utc"] = ""

    return result

def read_index_csv(index_path: str) -> List[Tuple[float, int, int, str]]:
    """
    Lee index csv. Admite:
      - sin cabecera: threshold,adx1d_min,adx4_min,summary_path
      - con cabecera: mismas columnas nombradas
    """
    rows: List[Tuple[float, int, int, str]] = []
    with open(index_path, "r", encoding="utf-8") as f:
        sniff = f.read(512)
        f.seek(0)
        has_header = any(h in sniff.lower() for h in ["threshold", "adx1d", "adx4", "summary_path", "path"])
        reader = csv.reader(f)
        if has_header:
            header = [h.strip().lower() for h in next(reader, [])]
            def col_idx(name_candidates):
                for name in name_candidates:
                    if name in header:
                        return header.index(name)
                return None
            i_thr = col_idx(["threshold"])
            i_a1  = col_idx(["adx1d_min","adx1d"])
            i_a4  = col_idx(["adx4_min","adx4"])
            i_p   = col_idx(["summary_path","path","file"])
            for row in reader:
                if not row:
                    continue
                try:
                    thr = as_float(row[i_thr]) if i_thr is not None else math.nan
                    a1  = as_int(row[i_a1]) if i_a1 is not None else math.nan
                    a4  = as_int(row[i_a4]) if i_a4 is not None else math.nan
                    p   = row[i_p] if i_p is not None else ""
                    rows.append((thr, a1, a4, p))
                except Exception:
                    continue
        else:
            # sin cabecera: 4 columnas
            for row in reader:
                if not row or len(row) < 4:
                    continue
                thr = as_float(row[0])
                a1  = as_int(row[1])
                a4  = as_int(row[2])
                p   = row[3]
                rows.append((thr, a1, a4, p))
    return rows

def discover_from_pattern(pattern: str) -> List[Tuple[float, int, int, str]]:
    files = sorted(glob(pattern))
    # sin info de threshold/ADX en el nombre -> NaN (se muestra igualmente)
    return [(math.nan, math.nan, math.nan, p) for p in files]

def build_dataframe(rows: List[Tuple[float, int, int, str]], mode: str) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for thr, a1, a4, path in rows:
        metrics = parse_summary_json(path)
        rec = {
            "mode": mode,
            "threshold": as_float(thr),
            "adx1d_min": as_int(a1),
            "adx4_min": as_int(a4),
            "summary_path": os.path.abspath(path),
            "timestamp_utc": metrics.get("timestamp_utc", ""),
            "net90": metrics.get("net90", math.nan),
            "pf90": metrics.get("pf90", math.nan),
            "score90": metrics.get("score90", math.nan),
            "mdd90": metrics.get("mdd90", math.nan),
            "trades90": metrics.get("trades90", math.nan),
            "win90": metrics.get("win90", math.nan),
            "net180": metrics.get("net180", math.nan),
            "pf180": metrics.get("pf180", math.nan),
            "score180": metrics.get("score180", math.nan),
            "mdd180": metrics.get("mdd180", math.nan),
            "trades180": metrics.get("trades180", math.nan),
            "win180": metrics.get("win180", math.nan),
            "holdout_net": metrics.get("holdout_net", math.nan),
            "holdout_pf": metrics.get("holdout_pf", math.nan),
            "holdout_score": metrics.get("holdout_score", math.nan),
            "holdout_mdd": metrics.get("holdout_mdd", math.nan),
            "holdout_trades": metrics.get("holdout_trades", math.nan),
        }
        records.append(rec)

    df = pd.DataFrame.from_records(records)

    # Limpieza: quita filas sin métricas (todo NaN en métricas claves)
    metric_cols = ["score180","net180","pf180","mdd180","score90","net90","pf90","mdd90"]
    if len(df) == 0:
        return df
    df["has_any_metric"] = df[metric_cols].notna().any(axis=1)
    df = df[df["has_any_metric"]].drop(columns=["has_any_metric"])

    # Orden lógico
    order = [
        "mode","threshold","adx1d_min","adx4_min",
        "net90","pf90","score90","mdd90","trades90","win90",
        "net180","pf180","score180","mdd180","trades180","win180",
        "holdout_net","holdout_pf","holdout_score","holdout_mdd","holdout_trades",
        "timestamp_utc","summary_path"
    ]
    df = df.reindex(columns=order)
    return df

def _df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        # Fallback si falta 'tabulate'
        header = "| " + " | ".join(df.columns) + " |\n"
        sep    = "| " + " | ".join(["---"]*len(df.columns)) + " |\n"
        body   = ""
        for _, row in df.iterrows():
            body += "| " + " | ".join("" if (isinstance(v, float) and math.isnan(v)) else str(v) for v in row.tolist()) + " |\n"
        return header + sep + body

def write_outputs(df: pd.DataFrame, outdir: str) -> Tuple[str, str, str]:
    os.makedirs(outdir, exist_ok=True)
    csv_out = os.path.join(outdir, "sweep_summary.csv")
    md_out  = os.path.join(outdir, "sweep_summary.md")
    best_out= os.path.join(outdir, "sweep_best.json")

    # CSV
    df.to_csv(csv_out, index=False)

    # MD (Top 10)
    head = df.sort_values(by=["score180","net180"], ascending=[False, False]).head(10)
    md_tbl = _df_to_markdown(head)

    # Winners por threshold
    winners_md = "### Ganadores por threshold ({})\n\n".format(df["mode"].iloc[0] if len(df)>0 else "live")
    if len(df) > 0 and "threshold" in df.columns and df["threshold"].notna().any():
        grp = (df.sort_values(by=["threshold","score180","net180"], ascending=[True, False, False])
                 .groupby("threshold", as_index=False)
                 .first()[["threshold","adx1d_min","adx4_min","score180","net180","pf180","mdd180","summary_path"]])
        winners_md += _df_to_markdown(grp)
    else:
        winners_md += "\n*(sin datos)*\n"

    md_lines = []
    md_lines.append("# Sweep summary\n")
    mode = df["mode"].iloc[0] if len(df)>0 else "live"
    md_lines.append(f"## Modo: {mode}\n\n")
    md_lines.append(md_tbl + "\n\n")
    md_lines.append(winners_md + "\n")

    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # BEST
    if len(df) > 0:
        best = df.sort_values(by=["score180","net180"], ascending=[False, False]).iloc[0].to_dict()
    else:
        best = {}
    with open(best_out, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    return csv_out, md_out, best_out

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", help="CSV con: threshold,adx1d_min,adx4_min,summary_path")
    ap.add_argument("--pattern", help="Patrón glob (p.ej. reports/summary_rf_sentiment_EXP_*.json)")
    ap.add_argument("--outdir", default="reports", help="Directorio de salida (por defecto: reports)")
    ap.add_argument("--mode", default="live", choices=["live","freeze"], help="Etiqueta de modo para la tabla")
    args = ap.parse_args()

    if not args.index and not args.pattern:
        ap.error("Debes especificar --index o --pattern")

    if args.index:
        rows = read_index_csv(args.index)
    else:
        rows = discover_from_pattern(args.pattern)

    if not rows:
        print("⚠️  No se encontraron entradas para resumir.")
        sys.exit(0)

    df = build_dataframe(rows, mode=args.mode)

    if len(df) == 0:
        # Genera archivos vacíos pero válidos con encabezados
        empty_cols = [
            "mode","threshold","adx1d_min","adx4_min",
            "net90","pf90","score90","mdd90","trades90","win90",
            "net180","pf180","score180","mdd180","trades180","win180",
            "holdout_net","holdout_pf","holdout_score","holdout_mdd","holdout_trades",
            "timestamp_utc","summary_path"
        ]
        pd.DataFrame(columns=empty_cols).to_csv(os.path.join(args.outdir, "sweep_summary.csv"), index=False)
        with open(os.path.join(args.outdir, "sweep_summary.md"), "w", encoding="utf-8") as f:
            f.write("# Sweep summary\n\n*(sin datos)*\n")
        with open(os.path.join(args.outdir, "sweep_best.json"), "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        print("⚠️  No hubo filas válidas (todas sin métricas reconocibles).")
        sys.exit(0)

    csv_out, md_out, best_out = write_outputs(df, args.outdir)

    # Log en consola
    print("✅ Resumen generado:")
    print(f"  • CSV: {csv_out}")
    print(f"  • MD : {md_out}")
    print(f"  • BEST (json): {best_out}\n")

    head = df.sort_values(by=["score180","net180"], ascending=[False, False]).head(10)
    print("— Top 10 global —")
    with pd.option_context('display.max_colwidth', 120):
        print(head[["mode","threshold","adx1d_min","adx4_min","score180","net180","pf180","mdd180","summary_path"]].to_string(index=False))

    best = head.iloc[0].to_dict()
    print("\n💡 Sugerencia de pick global:")
    print(f"   mode={best.get('mode')} thr={best.get('threshold')} a1={best.get('adx1d_min')} a4={best.get('adx4_min')} | "
          f"score180={best.get('score180')} net180={best.get('net180')} pf180={best.get('pf180')} mdd180={best.get('mdd180')}")
    print(f"   summary={best.get('summary_path')}")

if __name__ == "__main__":
    main()