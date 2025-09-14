# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import glob
import argparse
import sys
from typing import Optional

import pandas as pd


# --------------------------- Parámetros de aceptación ---------------------------

ACCEPT_THRESHOLDS = {
    "net_btc_ratio_min": 1.05,
    "mdd_vs_hodl_ratio_max": 0.85,
    "flips_per_year_max": 26.0,
}


# --------------------------- Utilidades ---------------------------

def guess_window_from_suffix(s: str) -> str:
    """
    Intenta inferir ventana OOS a partir del suffix.
    Ejemplos: "...-oos23Q4" -> "2023Q4", "...-oos24H1" -> "2024H1"
    """
    if not isinstance(s, str):
        return ""
    m = re.search(r"-oos(\d{2})(Q[1-4]|H[12])$", s)
    if m:
        yy, bucket = m.groups()
        yyyy = f"20{yy}"
        return f"{yyyy}{bucket}"
    # fallback: busca secuencias tipo 2023Q4 / 2024H1 en el texto
    m2 = re.search(r"(20\d{2}(?:Q[1-4]|H[12]))", s)
    return m2.group(1) if m2 else ""


def infer_suffix_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si no existe 'suffix', intenta derivarlo desde alguna columna con rutas de archivos.
    Busca patrones '__<suffix>.csv' o '-oos..' en strings.
    """
    if "suffix" in df.columns:
        return df

    work = df.copy()
    suffix_col: Optional[str] = None

    # 1) Buscamos columnas que claramente tengan paths de reportes
    candidates = [c for c in work.columns if any(x in c.lower() for x in ("equity", "kpis", "summary", "path", "file"))]
    for c in candidates:
        s = work[c].astype(str)
        # patrón con doble underscore
        extr = s.str.extract(r"__([^/]+?)\.csv$", expand=False)
        if extr.notna().any():
            work["suffix"] = extr.fillna("")
            suffix_col = "suffix"
            break
        # patrón directo de sufijos (sin __)
        extr2 = s.str.extract(r"([A-Za-z0-9\-]+-oos(?:\d{2})(?:Q[1-4]|H[12]))", expand=False)
        if extr2.notna().any():
            work["suffix"] = extr2.fillna("")
            suffix_col = "suffix"
            break

    if suffix_col is None:
        # como último recurso, intenta desde un campo 'suffix*' si existe con otro nombre
        for c in df.columns:
            if c.lower().startswith("suffix"):
                work["suffix"] = df[c].astype(str)
                suffix_col = "suffix"
                break

    return work if suffix_col else work


def build_df_from_reports(reports_dir: str) -> pd.DataFrame:
    """
    Escanea reports_dir y reconstruye una tabla de KPIs a partir de los CSV:
      - *_kpis__<suffix>.csv
    Si no encuentra ninguno, devuelve DF vacío con columnas estándar.
    """
    cols = ["suffix", "net_btc_ratio", "mdd_model_usd", "mdd_hodl_usd",
            "mdd_vs_hodl_ratio", "flips_total", "flips_per_year"]
    rows = []

    if not reports_dir or not os.path.isdir(reports_dir):
        return pd.DataFrame(columns=cols)

    # Patrón moderno: *_kpis__<suffix>.csv
    pattern = os.path.join(reports_dir, "*_kpis__*.csv")
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"__([^/]+?)\.csv$", path)
        if not m:
            continue
        suffix = m.group(1)
        try:
            k = pd.read_csv(path)
        except Exception:
            continue
        if k.empty:
            continue
        row = {c: pd.to_numeric(k.iloc[0].get(c), errors="coerce") if c != "suffix" else suffix
               for c in cols}
        row["suffix"] = suffix
        rows.append(row)

    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# --------------------------- Carga + preparación ---------------------------

def load_and_prepare(log_path: str, reports_dir: str, include_dbg: bool = False) -> pd.DataFrame:
    """
    Carga experiments_log.csv y UNE con lo reconstruido desde reports/.
    - Si el log existe pero no trae 'suffix', se intenta inferir.
    - Se reconstruyen kpis desde reports_dir y se concatenan ambos.
    - Se hace dedupe por 'suffix' (keep='last').
    - Si include_dbg=False, se filtran sufijos 'dbg', 'dbg-*' y 'lo-que-uses'.
    """
    # --- 1) Leer LOG (opcional) ---
    df_log: pd.DataFrame | None = None
    if log_path and os.path.exists(log_path):
        try:
            df_log = pd.read_csv(log_path)
        except Exception as e:
            print(f"[WARN] No se pudo leer {log_path}: {e}")
            df_log = None
    else:
        print(f"[WARN] No existe el log: {log_path}")

    if df_log is not None and not df_log.empty:
        if "suffix" not in df_log.columns:
            df_log = infer_suffix_column(df_log)
        # Normaliza / filtra filas sin suffix
        if "suffix" in df_log.columns:
            df_log = df_log.copy()
            df_log["suffix"] = (
                df_log["suffix"].astype(str)
                .str.replace(r"\.csv$", "", regex=True)
                .str.strip()
            )
            df_log = df_log[df_log["suffix"] != ""]
        else:
            df_log = None

    # --- 2) Reconstruir desde REPORTS ---
    df_rep = build_df_from_reports(reports_dir)

    # --- 3) Unir fuentes disponibles ---
    frames = []
    if df_log is not None and not df_log.empty:
        frames.append(df_log)
    if df_rep is not None and not df_rep.empty:
        frames.append(df_rep)

    if not frames:
        raise RuntimeError(f"No se pudo preparar dataset; revisa {log_path} y/o {reports_dir}.")

    df = pd.concat(frames, ignore_index=True, sort=False)

    # --- 4) Normalización y enriquecimiento ---
    if "suffix" not in df.columns:
        df["suffix"] = ""
    df["suffix"] = df["suffix"].astype(str).str.replace(r"\.csv$", "", regex=True)
    df = df.drop_duplicates(subset=["suffix"], keep="last").reset_index(drop=True)
    df["window"] = df["suffix"].map(guess_window_from_suffix)

    # Filtra debug si no se pide explícitamente
    if not include_dbg and "suffix" in df.columns:
        mask_dbg = df["suffix"].astype(str).str.match(r"^(dbg(?:-.+)?|lo-que-uses)$")
        df = df[~mask_dbg].reset_index(drop=True)

    # Asegurar tipos numéricos clave
    for col in ["net_btc_ratio", "mdd_model_usd", "mdd_hodl_usd",
                "mdd_vs_hodl_ratio", "flips_total", "flips_per_year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Reordenar columnas si existen
    ordered = ["suffix", "window", "net_btc_ratio", "mdd_model_usd", "mdd_hodl_usd",
               "mdd_vs_hodl_ratio", "flips_total", "flips_per_year"]
    remain = [c for c in df.columns if c not in ordered]
    return df[[c for c in ordered if c in df.columns] + remain]


def apply_acceptance_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas PASS y FAIL_REASON según los umbrales ACCEPT_THRESHOLDS.
    - Si faltan métricas clave (NaN), marca como FAIL con 'missing_metrics'.
    - Ordena dejando arriba lo “mejorcito”.
    """
    df = df.copy()

    KEYS = ["net_btc_ratio", "mdd_vs_hodl_ratio", "flips_per_year"]

    def _fail(row) -> str:
        reasons = []
        v = row.get

        # 1) Falta de métricas -> FAIL
        missing = [k for k in KEYS if pd.isna(v(k))]
        if missing:
            reasons.append("missing_metrics")

        # 2) Reglas
        if pd.notna(v("net_btc_ratio")) and v("net_btc_ratio") < ACCEPT_THRESHOLDS["net_btc_ratio_min"]:
            reasons.append("net_btc_ratio<1.05")
        if pd.notna(v("mdd_vs_hodl_ratio")) and v("mdd_vs_hodl_ratio") > ACCEPT_THRESHOLDS["mdd_vs_hodl_ratio_max"]:
            reasons.append("mdd_vs_hodl_ratio>0.85")
        if pd.notna(v("flips_per_year")) and v("flips_per_year") > ACCEPT_THRESHOLDS["flips_per_year_max"]:
            reasons.append("flips_per_year>26.0")
        return "; ".join(reasons)

    df["FAIL_REASON"] = df.apply(_fail, axis=1)
    df["PASS"] = df["FAIL_REASON"].eq("")

    # Ordena para que lo “mejorcito” quede arriba
    sort_cols = [("PASS", False), ("window", True), ("net_btc_ratio", False)]
    sort_by = [c for c, _ in sort_cols if c in df.columns]
    ascending = [asc for c, asc in sort_cols if c in df.columns]
    if sort_by:
        df = df.sort_values(sort_by, ascending=ascending, na_position="last")

    return df


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="reports/mini_accum/experiments_log.csv")
    ap.add_argument("--reports-dir", default="reports/mini_accum")
    ap.add_argument("--include-dbg", action="store_true",
                    help="No filtrar sufijos 'dbg', 'dbg-*' y 'lo-que-uses'.")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Ruta de salida opcional. Si termina en .tsv escribe con tabulador; si no, CSV."
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="No imprimir a stdout (útil junto con --out)."
    )
    ap.add_argument(
        "--format",
        dest="out_format",
        choices=["csv", "tsv"],
        help="Forzar formato de salida (solo aplica con --out)."
    )
    ap.add_argument(
        "--only-pass",
        action="store_true",
        help="Mostrar solo filas con PASS=True."
    )
    ap.add_argument(
        "--strict-exit",
        action="store_true",
        help="Salir con código 1 si no hay ningún PASS."
    )
    args = ap.parse_args()

    df = load_and_prepare(args.log, args.reports_dir, include_dbg=args.include_dbg)
    df = apply_acceptance_rules(df)

    cols_show = ["suffix", "window", "net_btc_ratio", "mdd_model_usd", "mdd_hodl_usd",
                 "mdd_vs_hodl_ratio", "flips_total", "flips_per_year", "PASS", "FAIL_REASON"]
    cols_show = [c for c in cols_show if c in df.columns]
    out_df = df[cols_show]

    # Filtrar solo aprobados si se solicita
    if getattr(args, "only_pass", False) and "PASS" in out_df.columns:
        out_df = out_df[out_df["PASS"]].reset_index(drop=True)

    # Salida por pantalla (a menos que --quiet)
    if not args.quiet:
        try:
            print(out_df.to_string(index=False))
        except Exception:
            print(out_df)

    # Salida a archivo si se solicita
    if args.out:
        out_path = args.out
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Determinar formato de salida
        fmt = (args.out_format or "").lower() if hasattr(args, "out_format") else ""
        if fmt not in {"csv", "tsv"}:
            ext = os.path.splitext(out_path)[1].lower()
            fmt = "tsv" if ext == ".tsv" else "csv"

        sep = "\t" if fmt == "tsv" else ","
        out_df.to_csv(out_path, index=False, sep=sep)

    # Código de salida estricto (útil para CI)
    if getattr(args, "strict_exit", False):
        # Si se pidió only-pass, la condición se evalúa sobre lo filtrado; de lo contrario, sobre el total
        has_pass = bool(out_df["PASS"].any()) if getattr(args, "only_pass", False) and "PASS" in out_df.columns else (bool(df["PASS"].any()) if "PASS" in df.columns else False)
        sys.exit(0 if has_pass else 1)


if __name__ == "__main__":
    main()