#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import yaml


@dataclass
class Costs:
    fee_bps_per_side: float = 2.0
    slip_bps_per_side: float = 2.0


def load_ohlc(csv_path: str, ts_col: str, tz_input: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Parsear a UTC (los datos vienen con +00:00, pero forzamos utc=True por robustez)
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.rename(columns={ts_col: "ts"})
    # columnas esperadas: ts, open, high, low, close, volume
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak - 1.0).fillna(0.0)
    return float(-dd.min())  # positivo (ej. 0.25 = -25%)


def simulate_sma(
    df: pd.DataFrame,
    sma_fast: int,
    sma_slow: int,
    costs: Costs
) -> Tuple[pd.DataFrame, dict]:
    # Guardarraíl: si no hay datos suficientes
    if df is None or len(df) < max(sma_fast, sma_slow) + 1:
        empty = df[["ts","open","high","low","close"]].copy() if df is not None else pd.DataFrame(
            columns=["ts","open","high","low","close"]
        )
        empty["sma_fast"] = pd.NA
        empty["sma_slow"] = pd.NA
        empty["position"] = 0
        empty["executed"] = ""
        empty["model_equity"] = 1.0
        empty["hodl_equity"] = 1.0
        empty["model_equity_btc"] = 1.0
        kpis = {
            "net_btc_ratio": 1.0,
            "mdd_model_usd": 0.0,
            "mdd_hodl_usd": 0.0,
            "mdd_vs_hodl_ratio": 1.0,
            "flips_total": 0,
            "flips_per_year": 0.0,
            "net_btc_vs_hodl": 1.0,
            "run_ok": False,
            "skip_reason": "not_enough_data",
            "sma_fast": int(sma_fast),
            "sma_slow": int(sma_slow),
            "fee_bps_per_side": float(costs.fee_bps_per_side),
            "slip_bps_per_side": float(costs.slip_bps_per_side),
        }
        return empty, kpis

    px = df["close"].astype(float)
    fast = px.rolling(sma_fast, min_periods=sma_fast).mean()
    slow = px.rolling(sma_slow, min_periods=sma_slow).mean()

    # señal long-only (sin lookahead)
    pos = (fast > slow).astype(int)

    # cambios de posición → BUY/SELL
    pos_shift = pos.shift(1).fillna(0).astype(int)
    delta = pos - pos_shift
    executed = delta.map({1: "BUY", -1: "SELL"}).where(delta != 0, "")

    # rendimientos diarios (USD)
    ret = px.pct_change().fillna(0.0)

    # costes cuando hay trade
    trade_mask = executed.isin(["BUY", "SELL"])
    cost_bps = (costs.fee_bps_per_side + costs.slip_bps_per_side)
    cost_factor = 1.0 - (cost_bps / 10000.0)

    # equity modelo (USD)
    eq = pd.Series(1.0, index=df.index)
    for i in range(1, len(df)):
        eq.iloc[i] = eq.iloc[i-1] * (1.0 + ret.iloc[i] * pos.iloc[i])
        if trade_mask.iloc[i]:
            eq.iloc[i] *= cost_factor

    # HODL (USD)
    hodl = (1.0 + ret).cumprod()

    # equity en BTC (sats) = modelo / HODL
    equity_btc = eq / hodl

    # KPIs
    net_btc_ratio = float(eq.iloc[-1])   # multiplicador en USD
    mdd_model = float(max_drawdown(eq))
    mdd_hodl  = float(max_drawdown(hodl))
    mdd_vs_hodl_ratio = (mdd_model / mdd_hodl) if mdd_hodl > 0 else (1.0 if mdd_model == 0 else 999.0)

    flips = int(executed.isin(["BUY", "SELL"]).sum())
    days = int((df["ts"].iloc[-1] - df["ts"].iloc[0]).days or 1)
    fpy = flips / (days / 365.0)

    kpis = {
        "net_btc_ratio": net_btc_ratio,               # USD
        "mdd_model_usd": mdd_model,
        "mdd_hodl_usd": mdd_hodl,
        "mdd_vs_hodl_ratio": mdd_vs_hodl_ratio,
        "flips_total": flips,
        "flips_per_year": float(fpy),
        "net_btc_vs_hodl": float(equity_btc.iloc[-1]),  # ← SATS (modelo vs HODL)
        "run_ok": bool(float(equity_btc.iloc[-1]) > 1.0 and flips > 0),
        "skip_reason": "" if (float(equity_btc.iloc[-1]) > 1.0 and flips > 0) else "sats<=1 or no_flips",
        "sma_fast": int(sma_fast),
        "sma_slow": int(sma_slow),
        "fee_bps_per_side": float(costs.fee_bps_per_side),
        "slip_bps_per_side": float(costs.slip_bps_per_side),
    }

    out = df[["ts","open","high","low","close"]].copy()
    out["sma_fast"] = fast
    out["sma_slow"] = slow
    out["position"] = pos
    out["executed"] = executed
    out["model_equity"] = eq
    out["hodl_equity"]  = hodl
    out["model_equity_btc"] = equity_btc  # ← exportar la curva en sats

    return out, kpis


def simulate_peak_trough(
    df: pd.DataFrame,
    dd_pct: float,
    rb_pct: float,
    costs: Costs,
    sma_gate: Optional[pd.Series] = None,
    gate_mode: str = "sell",
    dd_hard_pct: Optional[float] = None,
    bull_hold_sma: int = 0,
) -> Tuple[pd.DataFrame, dict]:
    """
    Vende tras caída desde pico (DD) y recompra tras rebote desde valle (RB).

    Gate por SMA:
      - gate_mode="sell": permite SELL solo si precio < SMA (evita vender en bull).
      - gate_mode="buy" : permite BUY solo si precio > SMA (recompra confirmada).
      - gate_mode="both": aplica ambas.
      - gate_mode="off" : sin gate.

    dd_hard_pct: si se define, fuerza SELL al superar ese DD desde pico,
    aunque el gate bloqueara la venta (hard stop).
    bull_hold_sma: si >0, cuando precio >= SMA(bull_hold_sma) se fuerza estar
    largos y se suspenden ventas PT mientras dure esa condición.
    """
    if df is None or len(df) < 2:
        empty = df[["ts","open","high","low","close"]].copy() if df is not None else pd.DataFrame(
            columns=["ts","open","high","low","close"]
        )
        empty["sma_fast"] = pd.NA
        empty["sma_slow"] = pd.NA
        empty["position"] = 0
        empty["executed"] = ""
        empty["model_equity"] = 1.0
        empty["hodl_equity"] = 1.0
        empty["model_equity_btc"] = 1.0
        kpis = {
            "net_btc_ratio": 1.0,
            "mdd_model_usd": 0.0,
            "mdd_hodl_usd": 0.0,
            "mdd_vs_hodl_ratio": 1.0,
            "flips_total": 0,
            "flips_per_year": 0.0,
            "net_btc_vs_hodl": 1.0,
            "run_ok": False,
            "skip_reason": "not_enough_data",
            "sma_fast": 0,
            "sma_slow": 0,
            "fee_bps_per_side": float(costs.fee_bps_per_side),
            "slip_bps_per_side": float(costs.slip_bps_per_side),
        }
        return empty, kpis

    px = df["close"].astype(float).reset_index(drop=True)
    ts = df["ts"].reset_index(drop=True)
    open_ = df["open"].astype(float).reset_index(drop=True)
    close_ = df["close"].astype(float).reset_index(drop=True)
    high_ = df["high"].astype(float).reset_index(drop=True)
    low_  = df["low"].astype(float).reset_index(drop=True)

    gate = pd.Series(sma_gate).reset_index(drop=True) if sma_gate is not None else None

    bull = None
    if bull_hold_sma and bull_hold_sma > 0:
        bull = px.rolling(bull_hold_sma, min_periods=bull_hold_sma).mean()

    dd = dd_pct / 100.0
    rb = rb_pct / 100.0
    dd_hard = (dd_hard_pct / 100.0) if dd_hard_pct is not None else None

    n = len(px)
    pos = pd.Series(0, index=range(n), dtype=int)
    executed = pd.Series("", index=range(n), dtype=object)

    ret = px.pct_change().fillna(0.0)
    cost_bps = (costs.fee_bps_per_side + costs.slip_bps_per_side)
    cost_factor = 1.0 - (cost_bps / 10000.0)

    in_pos = False
    peak = px.iloc[0]
    trough = px.iloc[0]

    eq = pd.Series(1.0, index=range(n), dtype=float)

    def gate_sell_ok(i_: int, price_: float) -> bool:
        if gate is None or gate_mode in ("off", "buy"):
            return True
        gv = gate.iloc[i_]
        return True if pd.isna(gv) else (price_ < gv)

    def gate_buy_ok(i_: int, price_: float) -> bool:
        if gate is None or gate_mode in ("off", "sell"):
            return True
        gv = gate.iloc[i_]
        return True if pd.isna(gv) else (price_ > gv)

    for i in range(1, n):
        prev_close = close_.iloc[i-1]
        c = close_.iloc[i]
        h = high_.iloc[i]
        l = low_.iloc[i]

        # --- Bull-hold: si precio >= SMA(bull_hold_sma), estar largos y no vender
        in_bull = False
        if bull is not None:
            bv = bull.iloc[i]
            in_bull = (not pd.isna(bv)) and (c >= bv)
        if in_bull:
            if not in_pos:
                in_pos = True
                pos.iloc[i] = 1
                executed.iloc[i] = "BUY"
                # Entramos al precio de cierre (aprox) con coste
                eq.iloc[i] = eq.iloc[i-1] * cost_factor
                peak = c
                trough = c
            else:
                pos.iloc[i] = 1
                # Seguimos largos: evolución por close/prev_close
                eq.iloc[i] = eq.iloc[i-1] * (c / prev_close)
                if h > peak:
                    peak = h
            continue

        if in_pos:
            # Actualizar pico con el HIGH intradiario
            if h > peak:
                peak = h
            # Triggers por LOW intradiario
            sell_trigger = l <= peak * (1.0 - dd)
            sell_hard = (dd_hard is not None) and (l <= peak * (1.0 - dd_hard))
            if (sell_trigger and gate_sell_ok(i, c)) or sell_hard:
                dd_eff = dd_hard if sell_hard else dd
                exec_price = peak * (1.0 - dd_eff)
                in_pos = False
                pos.iloc[i] = 0
                executed.iloc[i] = "SELL"
                # Salimos al precio del trigger (aprox stop) y aplicamos coste
                eq.iloc[i] = eq.iloc[i-1] * (exec_price / prev_close) * cost_factor
                trough = exec_price
            else:
                pos.iloc[i] = 1
                # Mantener largos por close/prev_close
                eq.iloc[i] = eq.iloc[i-1] * (c / prev_close)
        else:
            # Actualizar valle con el LOW intradiario
            if l < trough:
                trough = l
            # Trigger de compra por HIGH intradiario
            buy_trigger = h >= trough * (1.0 + rb)
            if buy_trigger and gate_buy_ok(i, c):
                exec_price = trough * (1.0 + rb)
                in_pos = True
                pos.iloc[i] = 1
                executed.iloc[i] = "BUY"
                # Entramos al trigger y cerramos el día en c
                eq.iloc[i] = eq.iloc[i-1] * cost_factor * (c / exec_price)
                peak = exec_price
            else:
                pos.iloc[i] = 0
                eq.iloc[i] = eq.iloc[i-1]

    hodl = (1.0 + ret).cumprod()
    equity_btc = eq / hodl

    net_btc_ratio = float(eq.iloc[-1])
    mdd_model = float(max_drawdown(eq))
    mdd_hodl = float(max_drawdown(hodl))
    mdd_vs_hodl_ratio = (mdd_model / mdd_hodl) if mdd_hodl > 0 else (1.0 if mdd_model == 0 else 999.0)

    flips = int(executed.isin(["BUY","SELL"]).sum())
    days = int((ts.iloc[-1] - ts.iloc[0]).days or 1)
    fpy = flips / (days / 365.0)

    out = pd.DataFrame({
        "ts": ts,
        "open": open_,
        "high": high_,
        "low": low_,
        "close": close_,
        "sma_fast": pd.NA,
        "sma_slow": pd.NA,
        "position": pos,
        "executed": executed,
        "model_equity": eq,
        "hodl_equity": hodl,
        "model_equity_btc": equity_btc,
    })

    kpis = {
        "net_btc_ratio": net_btc_ratio,
        "mdd_model_usd": mdd_model,
        "mdd_hodl_usd": mdd_hodl,
        "mdd_vs_hodl_ratio": mdd_vs_hodl_ratio,
        "flips_total": flips,
        "flips_per_year": float(fpy),
        "net_btc_vs_hodl": float(equity_btc.iloc[-1]),
        "run_ok": bool(float(equity_btc.iloc[-1]) > 1.0 and flips > 0),
        "skip_reason": "" if (float(equity_btc.iloc[-1]) > 1.0 and flips > 0) else "sats<=1 or no_flips",
        "sma_fast": 0,
        "sma_slow": 0,
        "fee_bps_per_side": float(costs.fee_bps_per_side),
        "slip_bps_per_side": float(costs.slip_bps_per_side),
    }
    return out, kpis


def kpis_on_visible(res_slice: pd.DataFrame, costs: Costs, sma_fast: int, sma_slow: int) -> dict:
    """Recalcula KPIs sobre el tramo visible, normalizando equity al inicio del slice."""
    if res_slice.empty:
        return {
            "net_btc_ratio": 1.0,
            "mdd_model_usd": 0.0,
            "mdd_hodl_usd": 0.0,
            "mdd_vs_hodl_ratio": 1.0,
            "flips_total": 0,
            "flips_per_year": 0.0,
            "run_ok": False,
            "skip_reason": "empty_slice",
            "sma_fast": int(sma_fast),
            "sma_slow": int(sma_slow),
            "fee_bps_per_side": float(costs.fee_bps_per_side),
            "slip_bps_per_side": float(costs.slip_bps_per_side),
        }

    eq = res_slice["model_equity"].astype(float).copy()
    hd = res_slice["hodl_equity"].astype(float).copy()
    if eq.iloc[0] != 0:
        eq /= eq.iloc[0]
    if hd.iloc[0] != 0:
        hd /= hd.iloc[0]

    mdd_m = max_drawdown(eq)
    mdd_h = max_drawdown(hd)
    ratio = (mdd_m / mdd_h) if mdd_h > 0 else (1.0 if mdd_m == 0 else 999.0)

    flips = int(res_slice["executed"].isin(["BUY", "SELL"]).sum())
    days = (res_slice["ts"].iloc[-1] - res_slice["ts"].iloc[0]).days or 1
    fpy = flips / (days / 365.0)

    # Sats multiplier (modelo vs HODL) sobre el tramo visible
    sats_mult = float(eq.iloc[-1] / hd.iloc[-1]) if hd.iloc[-1] != 0 else 0.0

    return {
        "net_btc_ratio": float(eq.iloc[-1]),
        "mdd_model_usd": float(mdd_m),
        "mdd_hodl_usd": float(mdd_h),
        "mdd_vs_hodl_ratio": float(ratio),
        "flips_total": flips,
        "flips_per_year": float(fpy),
        "net_btc_vs_hodl": sats_mult,
        "run_ok": bool(float(sats_mult) > 1.0 and flips > 0),
        "skip_reason": "" if (float(sats_mult) > 1.0 and flips > 0) else "sats<=1 or no_flips",
        "sma_fast": int(sma_fast),
        "sma_slow": int(sma_slow),
        "fee_bps_per_side": float(costs.fee_bps_per_side),
        "slip_bps_per_side": float(costs.slip_bps_per_side),
    }


def main() -> None:
    ap = argparse.ArgumentParser("KISS v1 (SMA/PT, 1D) — SATS-first KPIs")
    ap.add_argument("--config", required=True)
    ap.add_argument("--start", default=None)  # ISO date (UTC)
    ap.add_argument("--end", default=None)    # ISO date (UTC)
    ap.add_argument("--suffix", default=None)
    ap.add_argument("--fast", type=int, default=None, help="override sma_fast")
    ap.add_argument("--slow", type=int, default=None, help="override sma_slow")
    ap.add_argument("--mode", choices=["sma","pt"], default="sma", help="sma: cruce de medias; pt: peak/trough % drop/rebound")
    ap.add_argument("--dd_pct", type=float, default=10.0, help="pt: drawdown % desde pico para SELL (ej. 10)")
    ap.add_argument("--rb_pct", type=float, default=5.0,  help="pt: rebound % desde valle para BUY (ej. 5)")
    ap.add_argument("--gate_sma", type=int, default=0,
                    help="pt: si >0, permite SELL solo si el precio < SMA(gate_sma), ej. 200 para filtrar bull markets")
    ap.add_argument("--gate_mode", choices=["off","sell","buy","both"], default="sell",
                    help="pt: aplica gate SMA a SELL, BUY, ambos o desactiva")
    ap.add_argument("--dd_hard_pct", type=float, default=None,
                    help="pt: hard stop: fuerza SELL al superar este DD desde pico, ignora el gate")
    ap.add_argument("--bull_hold_sma", type=int, default=0,
                    help="pt: si >0, precio>=SMA(bull_hold_sma) fuerza pos=1 y suspende ventas")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    rep_dir = cfg["backtest"]["reports_dir"]
    os.makedirs(rep_dir, exist_ok=True)

    d1 = load_ohlc(cfg["data"]["ohlc_d1_csv"], cfg["data"]["ts_col"], cfg["data"]["tz_input"])

    # --- WARM-UP + slice visible ---
    u_start = pd.Timestamp(args.start, tz="UTC") if args.start else None
    u_end = pd.Timestamp(args.end, tz="UTC") if args.end else None

    sma_fast = args.fast if args.fast is not None else int(cfg["params"]["sma_fast"])
    sma_slow = args.slow if args.slow is not None else int(cfg["params"]["sma_slow"])

    # Warm-up generoso para “madurar” la SMA lenta
    pad_days = max(250, sma_slow * 3)
    if u_start is not None:
        warm_start = u_start - pd.Timedelta(days=pad_days)
        end_excl = (u_end + pd.Timedelta(days=1)) if u_end is not None else (d1["ts"].max() + pd.Timedelta(days=1))
        d1_warm = d1[(d1["ts"] >= warm_start) & (d1["ts"] < end_excl)].copy()
    else:
        d1_warm = d1.copy()

    costs = Costs(
        fee_bps_per_side=float(cfg["costs"]["fee_bps_per_side"]),
        slip_bps_per_side=float(cfg["costs"]["slip_bps_per_side"]),
    )

    # Simula en warm-up y luego recorta al tramo visible
    if args.mode == "pt":
        gate_series = None
        if args.gate_sma and args.gate_sma > 0:
            gate_series = d1_warm["close"].rolling(args.gate_sma, min_periods=args.gate_sma).mean()
        res_full, _ = simulate_peak_trough(
            d1_warm,
            args.dd_pct,
            args.rb_pct,
            costs,
            sma_gate=gate_series,
            gate_mode=args.gate_mode,
            dd_hard_pct=args.dd_hard_pct,
            bull_hold_sma=int(args.bull_hold_sma),
        )
    else:
        res_full, _ = simulate_sma(d1_warm, sma_fast, sma_slow, costs)

    vis_start = u_start or res_full["ts"].iloc[0]
    vis_end_excl = (u_end + pd.Timedelta(days=1)) if u_end is not None else (res_full["ts"].max() + pd.Timedelta(days=1))
    res = res_full[(res_full["ts"] >= vis_start) & (res_full["ts"] < vis_end_excl)].copy()

    # Normalizar equities al inicio del slice para KPIs
    kpis = kpis_on_visible(res, costs, sma_fast, sma_slow)
    # --- Enforce SATS-first gate (robustness) ---
    # Asegura que run_ok/skip_reason dependan de sats (modelo vs HODL),
    # por si alguna ruta previa dejó valores heredados basados en USD.
    try:
        sats_ok = float(kpis.get("net_btc_vs_hodl", 1.0)) > 1.0
    except Exception:
        sats_ok = False
    flips_ok = int(kpis.get("flips_total", 0)) > 0
    kpis["run_ok"] = bool(sats_ok and flips_ok)
    kpis["skip_reason"] = "" if kpis["run_ok"] else "sats<=1 or no_flips"

    # Re-normalizar columnas en res para que el CSV refleje el arranque en 1.0
    if not res.empty:
        if res["model_equity"].iloc[0] != 0:
            res["model_equity"] = res["model_equity"] / float(res["model_equity"].iloc[0])
        if res["hodl_equity"].iloc[0] != 0:
            res["hodl_equity"] = res["hodl_equity"] / float(res["hodl_equity"].iloc[0])

        # Mantener la curva en sats coherente con las series normalizadas del slice
        res["model_equity_btc"] = res["model_equity"] / res["hodl_equity"]
        res["model_equity_btc"] = res["model_equity_btc"].fillna(0.0)

    # --- Guardado de artefactos ---
    run_id = pd.Timestamp.utcnow().strftime("base_v0_1_%Y%m%d_%H%M")
    suffix = args.suffix or os.environ.get("REPORT_SUFFIX") or "KISS_SMA"

    def out(name: str, ext: str = "csv") -> str:
        return os.path.join(rep_dir, f"{run_id}_{name}__{suffix}.{ext}")

    eq_path = out("equity")
    kpi_path = out("kpis")
    md_path = out("summary", "md")
    flips_path = out("flips")

    res.to_csv(eq_path, index=False)

    pd.DataFrame([kpis], columns=[
        "net_btc_ratio", "mdd_model_usd", "mdd_hodl_usd", "mdd_vs_hodl_ratio",
        "flips_total", "flips_per_year", "net_btc_vs_hodl", "run_ok", "skip_reason",
        "sma_fast", "sma_slow", "fee_bps_per_side", "slip_bps_per_side",
    ]).to_csv(kpi_path, index=False)

    flips = res.loc[res["executed"].isin(["BUY", "SELL"]), ["ts", "executed", "open", "close"]]
    flips.to_csv(flips_path, index=False)

    with open(md_path, "w") as f:
        f.write(f"# KISS v1 — Resumen {run_id}\n\n")
        for k, v in kpis.items():
            f.write(f"- **{k}**: {v}\n")

    print(f"[OK] {eq_path}")
    print(f"[OK] {kpi_path}")
    print(f"[OK] {md_path}")
    # Resumen corto en consola (sats-first)
    try:
        print(
            f"[KPIS] USD_net={float(kpis['net_btc_ratio']):.6f}  "
            f"sats_mult={float(kpis['net_btc_vs_hodl']):.6f}  "
            f"run_ok={kpis['run_ok']}  flips={int(kpis['flips_total'])}"
        )
    except Exception:
        pass
    if len(flips):
        print(f"[SUMMARY] flips_total={len(flips)} (BUY={(flips['executed']=='BUY').sum()}, SELL={(flips['executed']=='SELL').sum()}) | flips_csv={os.path.basename(flips_path)}")
        iso = pd.to_datetime(flips["ts"]).dt.isocalendar()
        print("== Flips por semana ==")
        print(flips.groupby([iso.year, iso.week])["executed"].count())


if __name__ == "__main__":
    main()
