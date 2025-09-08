#!/usr/bin/env python3
# === runner.py ===
# Coloca órdenes a partir de señales CSV (modo dry-run o live).
# Requiere: pandas, ccxt
#
# Ejemplos:
#   DRY:  python runner.py place --no-submit --log logs/wk1_dry.log \
#           --exchange binanceus --assets BTC-USD,ETH-USD \
#           --signals reports/diamante_btc_wf3_week1_plus.csv reports/diamante_eth_wf3_week1_plus.csv \
#           --threshold 0.60 --walk-k 3 --horizon 60 \
#           --capital 10000 --risk 0.0075 \
#           --weights BTC-USD:0.70,ETH-USD:0.30 \
#           --stops   BTC-USD:0.015,ETH-USD:0.030
#
#   LIVE: python runner.py place --submit --confirm --log logs/wk1_live.log \
#           (mismos flags que arriba)

import argparse
import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import ccxt  # type: ignore
except Exception as e:
    ccxt = None


# -------------------------------
# Utilidades generales
# -------------------------------

def ts_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%fZ")


def norm_asset(a: str) -> str:
    """Normaliza símbolo con guion, p.ej. 'BTC-USD'."""
    return a.strip().upper().replace("/", "-").replace("USDT", "USD")


def to_exchange_symbol(a: str) -> str:
    """Convierte 'BTC-USD' -> 'BTC/USD' para ccxt."""
    a = norm_asset(a)
    parts = a.split("-")
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return a.replace("-", "/")


def parse_mapping(s: str, what: str) -> Dict[str, float]:
    """
    Parsea mapas 'ASSET:VAL,...'
    Ej: 'BTC-USD:0.70,ETH-USD:0.30'
    """
    out: Dict[str, float] = {}
    if not s:
        return out
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Formato inválido para {what}: '{chunk}' (esperado ASSET:VAL)")
        k, v = chunk.split(":", 1)
        out[norm_asset(k)] = float(v)
    return out


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def load_signal_row(csv_path: str, asset: str) -> Tuple[Optional[pd.Series], Dict[str, str]]:
    """
    Carga la ÚLTIMA fila relevante para el activo desde un CSV de señales 'plus'.
    Devuelve (row, metainfo_cols_encontradas)
    """
    if not os.path.exists(csv_path):
        return None, {}

    df = pd.read_csv(csv_path)
    meta = {}

    # Si existe columna 'symbol', filtramos por activo
    sym_col = find_col(df, ["symbol", "asset", "ticker"])
    if sym_col:
        df_sym = df[df[sym_col].astype(str).str.upper().str.replace("/", "-") == norm_asset(asset)]
        if not df_sym.empty:
            df = df_sym

    if df.empty:
        return None, {}

    # Tomamos última fila
    row = df.tail(1).iloc[0]

    # Guardamos qué columnas relevantes hay:
    for key, candidates in {
        "proba": ["proba", "prob", "p_up", "prob_up", "prob_long", "y_proba", "yhat_proba"],
        "signal": ["signal", "yhat", "y_pred", "side"],
        "entry": ["entry", "entry_price", "price", "px_entry"],
        "sl": ["sl", "stop", "stop_price", "px_sl"],
        "tp1": ["tp1", "take1", "tp_1", "px_tp1"],
        "tp2": ["tp2", "take2", "tp_2", "px_tp2"],
        "horizon": ["days", "horizon", "H"],
        "walk_k": ["walk_k", "fold_k", "k"],
        "ts": ["ts", "timestamp", "time"],
    }.items():
        col = find_col(df, candidates)
        if col:
            meta[key] = col

    return row, meta


def fetch_entry_from_market(ex, symbol: str) -> float:
    """Obtiene un precio de referencia (best ask) si no viene en la señal."""
    try:
        ob = ex.fetch_order_book(symbol)
        if ob and ob.get("asks"):
            return float(ob["asks"][0][0])
    except Exception:
        pass
    t = ex.fetch_ticker(symbol)
    return float(t["last"] or t["ask"] or t["bid"])


def round_amount_price(ex, market, amount: float, price: float) -> Tuple[float, float]:
    """Ajusta a la precisión/steps del exchange."""
    try:
        amount = ex.amount_to_precision(market["symbol"], amount)
        price = ex.price_to_precision(market["symbol"], price)
        return float(amount), float(price)
    except Exception:
        return float(amount), float(price)


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def log_line(path: Optional[str], line: str):
    print(line)
    if path:
        ensure_dir(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")


# -------------------------------
# Lógica de sizing y órdenes
# -------------------------------

def compute_order_long(entry: float, stop: float, capital_usd: float, risk_pct: float, weight: float) -> Dict:
    if stop >= entry:
        raise ValueError(f"Stop {stop} >= entry {entry} (invalida para largo).")
    max_loss = capital_usd * risk_pct * weight
    risk_per_unit = entry - stop  # USD de pérdida por 1 unidad (base)
    if risk_per_unit <= 0:
        raise ValueError("risk_per_unit <= 0")
    units = max_loss / risk_per_unit
    notional = units * entry
    return {
        "max_loss": max_loss,
        "risk_per_unit": risk_per_unit,
        "units": units,
        "notional": notional,
    }


def should_go_long(row: pd.Series, meta: Dict[str, str], threshold: float) -> bool:
    # Preferimos probas si existen
    if "proba" in meta:
        try:
            return float(row[meta["proba"]]) >= threshold
        except Exception:
            pass
    # Si hay columna 'signal' o similar, interpretamos >0 como long
    if "signal" in meta:
        try:
            return float(row[meta["signal"]]) > 0
        except Exception:
            pass
    # Fallback: si no hay nada, no entramos
    return False


def derive_targets(entry: float, stop: float) -> Tuple[float, float]:
    """Por defecto TP1 = 1R, TP2 = 2R."""
    R = entry - stop
    return entry + R, entry + 2 * R


def place_market_buy(ex, symbol: str, market: dict, amount: float, price_ref: float) -> dict:
    """
    Intenta market buy usando quoteOrderQty (cost) y si falla, usa amount base.
    """
    cost = amount * price_ref
    params = {"quoteOrderQty": ex.cost_to_precision(symbol, cost)} if hasattr(ex, "cost_to_precision") else {"quoteOrderQty": cost}
    try:
        return ex.create_order(symbol, type="market", side="buy", amount=None, price=None, params=params)
    except Exception:
        # fallback a base amount
        return ex.create_order(symbol, type="market", side="buy", amount=amount, price=None, params={})


# -------------------------------
# CLI principal
# -------------------------------

def main():
    p = argparse.ArgumentParser(description="Runner de órdenes para Diamante (4h).")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("place", help="Evalúa señales y coloca órdenes (dry o live).")
    q.add_argument("--exchange", default=os.getenv("EXCHANGE", "binanceus"))
    q.add_argument("--assets", default=os.getenv("ASSETS", "BTC-USD,ETH-USD"), help="Lista separada por coma. Ej: BTC-USD,ETH-USD")
    q.add_argument("--signals", nargs="+", required=True, help="Rutas CSV de señales (+).")
    q.add_argument("--threshold", type=float, default=float(os.getenv("THRESH", "0.60")))
    q.add_argument("--walk-k", type=int, default=int(os.getenv("WALK_K", "3")))
    q.add_argument("--horizon", type=int, default=int(os.getenv("HORIZON", "60")))
    q.add_argument("--capital", type=float, required=True, help="Capital total del sistema (USD).")
    q.add_argument("--risk", type=float, default=float(os.getenv("RISK_PCT", "0.0075")), help="Riesgo por trade (0.005–0.01).")
    q.add_argument("--weights", default=os.getenv("WEIGHT_MAP", ""), help="Mapa ASSET:W,... (si vacío, reparte equitativo).")
    q.add_argument("--stops", default=os.getenv("STOP_MAP", ""), help="Mapa ASSET:STOP_PCT (ej 0.015).")
    q.add_argument("--log", default=os.getenv("RUN_LOG", ""), help="Archivo de log (opcional).")
    g = q.add_mutually_exclusive_group()
    g.add_argument("--submit", action="store_true", help="LIVE: envía órdenes.")
    g.add_argument("--no-submit", action="store_true", help="DRY: no envía (default).")
    q.add_argument("--confirm", action="store_true", help="Pedir confirmación antes de enviar.")
    q.add_argument("--debug", action="store_true")

    args = p.parse_args()

    if args.cmd == "place":
        assets = [norm_asset(x) for x in args.assets.split(",") if x.strip()]
        if not assets:
            print("No hay activos en --assets", file=sys.stderr)
            sys.exit(2)

        weight_map = parse_mapping(args.weights, "weights") if args.weights else {}
        stop_map = parse_mapping(args.stops, "stops") if args.stops else {}

        # Ponderación igual si no se definieron pesos
        if not weight_map:
            w = 1.0 / len(assets)
            weight_map = {a: w for a in assets}

        mode = "live" if args.submit and not args.no_submit else "dry"
        log_path = args.log or ""
        if log_path:
            ensure_dir(log_path)

        log_line(log_path, f"[{ts_utc()}] RUN start | mode={mode} | exch={args.exchange} | THRESH={args.threshold} | K={args.walk_k} | H={args.horizon}")
        log_line(log_path, f"[{ts_utc()}] assets={assets} | weights={weight_map} | stops={stop_map} | capital={args.capital} | risk={args.risk}")
        log_line(log_path, f"[{ts_utc()}] signals={args.signals}")

        # Prepara exchange si LIVE
        ex = None
        markets = {}
        if mode == "live":
            if ccxt is None:
                print("ccxt no disponible. Instala 'ccxt'.", file=sys.stderr)
                sys.exit(3)
            klass = getattr(ccxt, args.exchange, None)
            if not klass:
                print(f"Exchange '{args.exchange}' no soportado por ccxt.", file=sys.stderr)
                sys.exit(3)
            ex = klass({
                "apiKey": os.getenv("BINANCEUS_API_KEY", os.getenv("BINANCE_API_KEY", "")),
                "secret": os.getenv("BINANCEUS_SECRET", os.getenv("BINANCE_SECRET", "")),
                "enableRateLimit": True,
            })
            ex.load_markets()
            markets = ex.markets

        # Mapeo heurístico CSV por activo (por nombre o por columna 'symbol')
        csv_by_asset: Dict[str, str] = {}
        for a in assets:
            # preferimos CSV cuyo nombre contenga el base del activo
            base = a.split("-")[0].lower()
            chosen = None
            for path in args.signals:
                name = os.path.basename(path).lower()
                if base in name:
                    chosen = path
                    break
            # si no encontramos por nombre, usamos el primero como fallback
            csv_by_asset[a] = chosen or args.signals[0]

        results = []

        for a in assets:
            csv_path = csv_by_asset[a]
            row, meta = load_signal_row(csv_path, a)
            if row is None:
                log_line(log_path, f"[{ts_utc()}] WARN {a} | no se encontró fila en {csv_path} → SKIP")
                continue

            go_long = should_go_long(row, meta, args.threshold)
            proba_val = float(row[meta["proba"]]) if "proba" in meta else None
            walk_k_val = int(row[meta["walk_k"]]) if "walk_k" in meta and str(row[meta["walk_k"]]).isdigit() else None
            horizon_val = int(row[meta["horizon"]]) if "horizon" in meta and str(row[meta["horizon"]]).isdigit() else None

            if not go_long:
                log_line(log_path, f"[{ts_utc()}] SKIP {a} | proba={proba_val} < THRESH={args.threshold} (o signal<=0)")
                continue

            # Entry
            entry = None
            if "entry" in meta:
                try:
                    entry = float(row[meta["entry"]])
                except Exception:
                    entry = None

            exch_symbol = to_exchange_symbol(a)
            if entry is None:
                if mode == "live" and ex is not None:
                    entry = fetch_entry_from_market(ex, exch_symbol)
                else:
                    # en dry-run, usa el último 'price' si existe o fallback 0
                    if "entry" in meta:
                        entry = float(row[meta["entry"]])
                    else:
                        # no tenemos precio (difícil simular). Abortamos este activo.
                        log_line(log_path, f"[{ts_utc()}] WARN {a} | sin entry price en CSV (y sin mercado live) → SKIP")
                        continue

            # Stop
            stop = None
            if "sl" in meta:
                try:
                    stop = float(row[meta["sl"]])
                except Exception:
                    stop = None
            if stop is None:
                pct = float(stop_map.get(a, 0.02))  # default 2% si no se pasó
                stop = entry * (1.0 - pct)

            # Targets
            tp1 = tp2 = None
            if "tp1" in meta:
                try:
                    tp1 = float(row[meta["tp1"]])
                except Exception:
                    tp1 = None
            if "tp2" in meta:
                try:
                    tp2 = float(row[meta["tp2"]])
                except Exception:
                    tp2 = None
            if tp1 is None or tp2 is None:
                tp1, tp2 = derive_targets(entry, stop)

            # Sizing
            weight = float(weight_map.get(a, 1.0 / len(assets)))
            sizing = compute_order_long(entry, stop, args.capital, args.risk, weight)
            units = sizing["units"]
            notional = sizing["notional"]

            # Ajuste de precisión si live
            market = markets.get(exch_symbol) if markets else None
            if mode == "live" and ex is not None and market:
                units, entry = round_amount_price(ex, market, units, entry)

            # Log humano para grep
            log_line(log_path, f"[{ts_utc()}] {a} ENTRY={entry:.8f} STOP={stop:.8f} TARGET1={tp1:.8f} TARGET2={tp2:.8f}")
            log_line(log_path, f"[{ts_utc()}] {a} SIZE(units)={units:.8f} NOTIONAL(usd)={notional:.2f} RISK={args.risk:.4f} WEIGHT={weight:.2f} PROBA={proba_val}")

            order_resp = None
            if mode == "live" and ex is not None:
                if args.confirm:
                    ans = input(f"CONFIRMAR compra MARKET {a} (units≈{units:.8f}, entry≈{entry:.2f})? [y/N]: ").strip().lower()
                    if ans != "y":
                        log_line(log_path, f"[{ts_utc()}] {a} CANCELADO por usuario.")
                        continue
                try:
                    order_resp = place_market_buy(ex, exch_symbol, market, units, entry)
                    oid = order_resp.get("id")
                    log_line(log_path, f"[{ts_utc()}] {a} LIVE OK | order_id={oid}")
                except Exception as e:
                    log_line(log_path, f"[{ts_utc()}] {a} LIVE ERROR | {type(e).__name__}: {e}")
                    continue  # pasa al siguiente activo

            results.append({
                "ts": ts_utc(),
                "mode": mode,
                "asset": a,
                "exchange": args.exchange,
                "symbol": exch_symbol,
                "entry": entry,
                "stop": stop,
                "tp1": tp1,
                "tp2": tp2,
                "proba": proba_val,
                "threshold": args.threshold,
                "walk_k": walk_k_val or args.walk_k,
                "horizon": horizon_val or args.horizon,
                "weight": weight,
                "risk": args.risk,
                "max_loss": sizing["max_loss"],
                "units": units,
                "notional": notional,
                "order": order_resp or {},
                "csv": csv_path,
            })

        # Resumen final JSON (útil para auditoría)
        if results:
            line = json.dumps({"summary": results}, ensure_ascii=False)
            log_line(log_path, line)
        log_line(log_path, f"[{ts_utc()}] RUN end | placed={sum(1 for r in results if r['order'])} | evaluated={len(results)}")


if __name__ == "__main__":
    main()