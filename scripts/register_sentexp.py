#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Registra las DOS corridas más recientes y válidas de SentimentEXP
(en modo freeze y live) en registry_runs.csv evitando duplicados.

Criterios de validez (flags esperados, del plan):
- use_sentiment=True
- threshold≈0.59 (tolerancia 1e-6)
- adx_daily_source="resample"
- adx1d_len=14
- adx1d_min>=30
- adx4_min>=18

Heurística freeze/live:
- freeze: flags.freeze_end no nulo o repro_lock=True
- live:   sin freeze_end (nulo/ausente)
Si no se encuentra una de las dos, se toma la mejor alternativa
más reciente válida (ej. dos live recientes).
"""

from __future__ import annotations
import csv
import glob
import json
import os
import os.path as _p
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

REG = Path("registry_runs.csv")
REPORTS = Path("reports")
STRATEGY = "SentimentEXP"

# Ajusta aquí si cambian los parámetros del plan
EXPECTED = {
    "use_sentiment": True,
    "threshold": 0.59,
    "adx_daily_source": "resample",
    "adx1d_len": 14,
    "adx1d_min": 30.0,
    "adx4_min": 18.0,
}
THRESH_TOL = 1e-6


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def wf_path(summary_path: str) -> str:
    base = _p.basename(summary_path)
    return _p.join("reports", base.replace("summary_", "walkforward_").replace(".json", ".csv"))


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def ok_flags(js: Dict[str, Any]) -> bool:
    f = js.get("flags", {})
    # Campos obligatorios y tolerancias
    if f.get("use_sentiment") is not True:
        return False
    try:
        thr = float(f.get("threshold", -1))
    except Exception:
        return False
    if abs(thr - EXPECTED["threshold"]) > THRESH_TOL:
        return False
    if f.get("adx_daily_source") != EXPECTED["adx_daily_source"]:
        return False
    try:
        if int(f.get("adx1d_len", -1)) != EXPECTED["adx1d_len"]:
            return False
        if float(f.get("adx1d_min", -1)) < EXPECTED["adx1d_min"]:
            return False
        if float(f.get("adx4_min", -1)) < EXPECTED["adx4_min"]:
            return False
    except Exception:
        return False
    return True


def mode_from_flags(js: Dict[str, Any]) -> str:
    f = js.get("flags", {})
    freeze_end = f.get("freeze_end")
    repro_lock = f.get("repro_lock", False)
    if freeze_end not in (None, "", "null") or repro_lock is True:
        return "freeze"
    return "live"


def read_existing_rows(reg: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    if reg.exists():
        with reg.open() as f:
            for row in csv.reader(f):
                if not row:
                    continue
                rows.append(row)
    return rows


def registry_has(rows: List[List[str]], strategy: str, mode: str, summary_path: str) -> bool:
    for r in rows:
        # formato esperado: timestamp,strategy,mode,summary_path,walkforward_path
        if len(r) >= 4 and r[1] == strategy and r[2] == mode and r[3].strip() == summary_path:
            return True
    return False


def tail_preview(reg: Path, n: int = 6) -> None:
    try:
        with reg.open() as f:
            data = f.read().splitlines()
        for line in data[-n:]:
            print(line)
    except Exception:
        pass


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    REG.touch(exist_ok=True)

    # 1) Encuentra summaries de SentimentEXP (más recientes primero)
    paths = sorted(
        glob.glob("reports/summary_rf_sentiment_EXP_*.json"),
        key=os.path.getmtime,
        reverse=True,
    )

    if len(paths) < 1:
        print("⚠️ No hay summaries de SentimentEXP en reports/")
        return

    # 2) Filtra por flags válidos del plan
    valid: List[Tuple[str, Dict[str, Any]]] = []
    for p in paths:
        js = load_json(p)
        if not js:
            continue
        if ok_flags(js):
            valid.append((p, js))

    if len(valid) < 1:
        print("⚠️ No hay summaries de SentimentEXP que cumplan flags esperados.")
        return

    # 3) Detecta freeze y live más recientes (si existen)
    best_live: Optional[Tuple[str, Dict[str, Any]]] = None
    best_freeze: Optional[Tuple[str, Dict[str, Any]]] = None

    for p, js in valid:
        m = mode_from_flags(js)
        if m == "live" and best_live is None:
            best_live = (p, js)
        elif m == "freeze" and best_freeze is None:
            best_freeze = (p, js)
        if best_live and best_freeze:
            break

    # Fallbacks: si falta alguno, toma el siguiente válido por orden
    if not (best_live and best_freeze):
        # ya tenemos al menos uno; busca otro distinto
        for p, js in valid:
            if best_live and p == best_live[0]:
                continue
            if best_freeze and p == best_freeze[0]:
                continue
            if not best_live:
                best_live = (p, js)
            elif not best_freeze:
                best_freeze = (p, js)
            if best_live and best_freeze:
                break

    if not (best_live and best_freeze):
        print("⚠️ No se encontraron dos corridas válidas (freeze y live).")
        return

    # 4) Escribe solo las filas que no existan aún (idempotente)
    existing = read_existing_rows(REG)
    to_append: List[List[str]] = []

    for mode, (p, _js) in (("freeze", best_freeze), ("live", best_live)):
        sp = p
        wp = wf_path(sp)
        row = [now_iso_utc(), STRATEGY, mode, sp, wp]
        if not registry_has(existing, STRATEGY, mode, sp):
            to_append.append(row)

    if not to_append:
        print("ℹ️ Nada que registrar (ya existen las entradas más recientes).")
        tail_preview(REG, n=10)
        return

    # 5) Append atómico
    with REG.open("a", newline="") as f:
        w = csv.writer(f)
        for row in to_append:
            w.writerow(row)

    print("✅ Registradas SentimentEXP (freeze & live) en registry_runs.csv")
    # Muestra últimas líneas
    tail_preview(REG, n=10)


if __name__ == "__main__":
    main()