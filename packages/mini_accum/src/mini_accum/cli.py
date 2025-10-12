#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

import argparse
import os
from typing import Optional

import pandas as pd
import yaml

from .io import load_ohlc, merge_daily_into_4h
from .sim import simulate, TradeCosts
from dataclasses import dataclass  # adicionado para V2

__all__ = ["main"]


def _rename_with_suffix(path: str, suffix: Optional[str]) -> str:
    """
    Si hay suffix, renombra foo.csv -> foo__{suffix}.csv.
    Devuelve la ruta final (renombrada u original).
    """
    if not suffix:
        return path
    base, ext = os.path.splitext(path)
    new_path = f"{base}__{suffix}{ext}"
    try:
        os.replace(path, new_path)
        print(f"[RENAMED] {os.path.basename(path)} -> {os.path.basename(new_path)}")
        return new_path
    except Exception as e:
        print(f"[WARN] rename failed for {path}: {e}")
        return path

# --- guardias previos al rename ---
import csv, sys

def _kpi_netbtc_or_none(kpi_csv: str):
    """
    Lee el CSV de KPIs y devuelve el valor de 'netBTC' como float si existe.
    Fallback: intenta encontrar el primer valor numérico de la primera fila de datos.
    """
    try:
        with open(kpi_csv, newline='') as fh:
            r = csv.DictReader(fh)
            first = next(r, None)
            if not first:
                return None
            # Preferimos columna 'netBTC' explícita
            if "netBTC" in first and (first["netBTC"] or "").strip():
                try:
                    return float(first["netBTC"])
                except Exception:
                    return None
            # Fallback: primer valor numérico en la fila
            for v in first.values():
                s = (v or "").strip()
                if not s:
                    continue
                try:
                    return float(s)
                except Exception:
                    continue
            return None
    except Exception:
        return None

def _flips_has_executed(flips_csv: str):
    try:
        with open(flips_csv, newline='') as fh:
            r = csv.DictReader(fh)
            for row in r:
                if (row.get('executed') or '').strip():
                    return True
    except Exception:
        pass
    return False

def _print_summary_and_save_flips(res: pd.DataFrame, rep_dir: str, run_id: str, suffix: Optional[str]) -> None:
    """
    Guarda *_flips.csv y saca un mini-resumen al final.
    Requiere que res tenga columnas: ts, executed, open, close.
    """
    if res.empty or "executed" not in res.columns:
        print("[SUMMARY] flips_total=0 (sin filas ejecutadas)")
        return

    flips = res[res["executed"].notna()][["ts", "executed", "open", "close"]].copy()
    flips_path = os.path.join(rep_dir, f"{run_id}_flips.csv")
    flips.to_csv(flips_path, index=False)
    flips_path = _rename_with_suffix(flips_path, suffix)

    buy_count = int((flips["executed"] == "BUY").sum())
    sell_count = int((flips["executed"] == "SELL").sum())
    total = len(flips)
    print(f"[SUMMARY] flips_total={total} (BUY={buy_count}, SELL={sell_count}) | flips_csv={os.path.basename(flips_path)}")

    if total:
        print("\n== FLIPS ejecutados ==")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(flips.to_string(index=False))

        iso = pd.to_datetime(flips["ts"]).dt.isocalendar()
        weekly = flips.groupby([iso.year, iso.week])["executed"].count()
        print("\n== Flips por semana ==")
        print(weekly)


def main() -> None:
    ap = argparse.ArgumentParser(description="Mini-BOT BTC (mini_accum) — backtest CLI")
    ap.add_argument("--config", default="configs/mini_accum/config.yaml", help="Ruta al YAML de configuración")
    ap.add_argument("--start", default=None, help="ISO date (UTC). Ej: 2024-01-01")
    ap.add_argument("--end", default=None, help="ISO date (UTC). Ej: 2024-06-30")
    ap.add_argument("--suffix", default=None, help="Sufijo de reporte; si no se pasa, se usa $REPORT_SUFFIX si existe")
    args = ap.parse_args()

    # suffix precedence: CLI arg > env var
    suffix = args.suffix if args.suffix else os.environ.get("REPORT_SUFFIX")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        # --- alias: soporta 'modules.v2' como fuente de disciplina si no hay 'discipline' ---
        _mods_v2 = ((cfg.get("modules") or {}).get("v2") or {}) if isinstance(cfg, dict) else {}
        if _mods_v2 and not cfg.get("discipline"):
            cfg["discipline"] = {
                "bull_hold": (_mods_v2.get("bull_hold") or {}),
                "cooldown_after_loss": (_mods_v2.get("cooldown_after_loss") or {}),
                "hibernation_on_chop": (_mods_v2.get("hibernation_on_chop") or {}),
            }
        # --- v2.0 discipline: lectura segura de config (todo OFF por defecto) ---
        disc = (cfg.get("discipline") or {}) if isinstance(cfg, dict) else {}

        bull = disc.get("bull_hold", {}) or {}
        cool = disc.get("cooldown_after_loss", {}) or {}
        hib  = disc.get("hibernation_on_chop", {}) or {}

        D_PARAMS = {
            "bull_enabled":     bool(bull.get("enabled", False)),
            "bull_min_bars":    int(bull.get("min_bars_after_entry", 2)),
            "bull_adx_min":     float(bull.get("adx_min", 25)),

            "cool_enabled":     bool(cool.get("enabled", False)),
            "cool_bars":        int(cool.get("cooldown_bars", 12)),

            "hib_enabled":      bool(hib.get("enabled", False)),
            "hib_adx_max":      float(hib.get("adx_max", 20)),
            "hib_min_bars":     int(hib.get("min_bars", 6)),
        }

        # Lo pasamos al motor; si el motor no lo usa, todo queda NO-OP
        cfg["_discipline"] = D_PARAMS

    rep_dir = cfg["backtest"]["reports_dir"]
    os.makedirs(rep_dir, exist_ok=True)

    # Cargar datos
    df4 = load_ohlc(cfg["data"]["ohlc_4h_csv"], cfg["data"]["ts_col"], cfg["data"]["tz_input"])
    d1 = load_ohlc(cfg["data"]["ohlc_d1_csv"], cfg["data"]["ts_col"], cfg["data"]["tz_input"])

    # Merge D1→4h
    df = merge_daily_into_4h(df4, d1)

    # --- TOP v1: asegurar d_sma200 (SMA(200) sobre close diario) ---
    # Lo calculamos a partir de d_close ya mergeado y lo proyectamos a cada vela 4h.
    if 'd_sma200' not in df.columns and 'd_close' in df.columns:
        day = pd.to_datetime(df['ts']).dt.floor('D')
        # Último close del día (serie diaria)
        daily_last = pd.Series(df['d_close'].values, index=day).groupby(level=0).last()
        sma200 = daily_last.rolling(200, min_periods=200).mean()
        # Mapear por día a cada fila 4h y rellenar hacia delante
        df['d_sma200'] = day.map(sma200)
        df['d_sma200'] = df['d_sma200'].ffill()


    # --- TOP v1: asegurar d_sma200 diaria mergeada al 4h ---
    # Si el merge previo no la incluyó, la calculamos desde el diario y la unimos por asof (backward).
    if 'd_sma200' not in df.columns:
        try:
            d_daily = d1[['ts', 'close']].copy()
            # asegurar frecuencia diaria y último close del día
            d_daily = d_daily.set_index('ts').resample('1D').last().reset_index()
            d_daily.rename(columns={'close': '_d_close'}, inplace=True)
            # SMA200 estricta (no exponencial); min_periods=200 para evitar “precalentamiento” sesgado
            d_daily['d_sma200'] = d_daily['_d_close'].rolling(window=200, min_periods=200).mean()
            d_daily = d_daily[['ts', 'd_sma200']].sort_values('ts')

            # merge_asof para alinear cada vela 4h con el último valor diario disponible (backward)
            df = pd.merge_asof(df.sort_values('ts'), d_daily, on='ts', direction='backward')
        except Exception as e:
            print(f"[WARN] d_sma200 not merged: {e}")

    # Usar intervalo semiabierto [start, end):
    # tratamos 'end' como inclusivo a nivel de día, avanzando 1 día y filtrando con '<'
    if args.start:
        start = pd.Timestamp(args.start, tz="UTC")
        df = df[df["ts"] >= start]
    if args.end:
        end_excl = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
        df = df[df["ts"] < end_excl]

    # === ADX helper (usado por disciplina v2.0) ===
    def _compute_adx_inplace(df: pd.DataFrame, period: int = 14) -> None:
        # Requiere columnas high, low, close
        if not {'high', 'low', 'close'}.issubset(df.columns):
            return
        high, low, close = df['high'], df['low'], df['close']
        up = high.diff()
        down = -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = pd.concat([
            (high - low).abs(),
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100.0
        adx = dx.ewm(span=period, adjust=False).mean()
        df['adx'] = adx.bfill().fillna(0.0)

    # === Activación condicional: si disciplina v2.0 está ON y no hay 'adx', lo calculamos ===
    disc = (cfg.get('discipline') or {})
    need_adx = any(bool((disc.get(k) or {}).get('enabled', False)) for k in
                   ('bull_hold', 'cooldown_after_loss', 'hibernation_on_chop'))
    if need_adx and 'adx' not in df.columns:
        # Si hay periodo ADX definido en YAML de filtros, úsalo; si no, 14
        adx_period = int((cfg.get('filters', {}) or {}).get('adx', {}).get('period', 14))
        _compute_adx_inplace(df, period=adx_period)

    # Costes
    costs = TradeCosts(
        fee_bps_per_side=float(cfg["costs"]["fee_bps_per_side"]),
        slip_bps_per_side=float(cfg["costs"]["slip_bps_per_side"]),
    )

    # Simulación
    res, kpis = simulate(cfg, df, costs)

    # Guardar salidas
    run_id = pd.Timestamp.utcnow().strftime("base_v0_1_%Y%m%d_%H%M")
    eq_path = os.path.join(rep_dir, f"{run_id}_equity.csv")
    kpi_path = os.path.join(rep_dir, f"{run_id}_kpis.csv")
    md_path = os.path.join(rep_dir, f"{run_id}_summary.md")

    res.to_csv(eq_path, index=False)
    kpi_df = pd.DataFrame([kpis])
    # Reordenar columnas para mostrar primero métricas clave (si existen)
    cols = list(kpi_df.columns)
    priority = [
        c for c in [
            "netBTC", "net_btc_ratio",
            "mdd_vs_HODL", "mdd_vs_hodl_ratio",
            "fpy", "flips_per_year"
        ] if c in cols
    ]
    others = [c for c in cols if c not in priority]
    kpi_df = kpi_df[priority + others] if priority else kpi_df
    kpi_df.to_csv(kpi_path, index=False)
    with open(md_path, "w") as f:
        f.write(f"# Mini-BOT BTC v0.1 — Resumen {run_id}\n\n")
        f.write("## KPIs\n")
        for k, v in kpis.items():
            f.write(f"- **{k}**: {v}\n")
        # v2.0 Discipline (si aplica)
        dp = cfg.get("_discipline", {})
        f.write("\n## v2.0 Discipline (si aplica)\n")
        f.write(f"- bull_hold: enabled={dp.get('bull_enabled', False)}, "
                f"min_bars_after_entry={dp.get('bull_min_bars', 2)}, "
                f"adx_min={dp.get('bull_adx_min', 25)}\n")
        f.write(f"- cooldown_after_loss: enabled={dp.get('cool_enabled', False)}, "
                f"cooldown_bars={dp.get('cool_bars', 12)}\n")
        f.write(f"- hibernation_on_chop: enabled={dp.get('hib_enabled', False)}, "
                f"adx_max={dp.get('hib_adx_max', 20)}, "
                f"min_bars={dp.get('hib_min_bars', 6)}\n")

    print(f"[OK] {eq_path}")
    print(f"[OK] {kpi_path}")
    print(f"[OK] {md_path}")

    # --- Guardarraíl antes de renombrar ---
    # Toleramos variaciones de nombre en KPIs entre engines: netBTC | net_btc_ratio | net_btc | net
    net_candidate = (
        kpis.get("netBTC")
        or kpis.get("net_btc_ratio")
        or kpis.get("net_btc")
        or kpis.get("net")
    )
    try:
        net = float(net_candidate) if net_candidate is not None else float(_kpi_netbtc_or_none(kpi_path) or 0.0)
    except Exception:
        net = float(_kpi_netbtc_or_none(kpi_path) or 0.0)

    has_flips = ("executed" in res.columns) and res["executed"].notna().any()

    if not (net > 0.0 and has_flips):
        # Caso inválido: NO renombrar; guardar flips sin sufijo para inspección
        print(f"[SKIP] KPI/Flips inválidos (netBTC={net:.4f}, flips={has_flips}); no renombro artefactos.")
        _print_summary_and_save_flips(res, rep_dir, run_id, suffix=None)
        return

    # Caso válido: renombrar artefactos y flips con el sufijo
    eq_path = _rename_with_suffix(eq_path, suffix)
    kpi_path = _rename_with_suffix(kpi_path, suffix)
    md_path = _rename_with_suffix(md_path, suffix)
    _print_summary_and_save_flips(res, rep_dir, run_id, suffix)

if __name__ == "__main__":
    main()