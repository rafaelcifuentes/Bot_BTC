mkdir -p "$HOME/PycharmProjects/Bot_BTC/scripts/mini_accum"
cat > "$HOME/PycharmProjects/Bot_BTC/scripts/mini_accum/vision_cli.py" <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse, json
try:
    import ccxt
except Exception as e:
    print("ERROR: ccxt no instalado:", e, file=sys.stderr); sys.exit(1)

def ex_binance_vision():
    k = os.getenv("BINANCE_API_KEY")
    s = os.getenv("BINANCE_API_SECRET")
    if not k or not s:
        print("ERROR: BINANCE_API_KEY/BINANCE_API_SECRET vacíos", file=sys.stderr)
        sys.exit(2)
    ex = ccxt.binance({
        "apiKey": k, "secret": s,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}
    })
    ex.set_sandbox_mode(True)  # Spot Testnet (VISION)
    return ex

def cmd_balance(args):
    ex = ex_binance_vision()
    bal = ex.fetch_balance()
    totals = bal.get('total', {})
    if args.assets:
        want = [a.strip().upper() for a in args.assets.split(",")]
        filt = {a: totals.get(a, 0) for a in want}
    else:
        # top por valor no existe directo en testnet; muestra algunos
        filt = {k: v for k, v in totals.items() if v}
    print(json.dumps(filt, indent=2, sort_keys=True))

def cmd_trades(args):
    ex = ex_binance_vision()
    sym = args.symbol
    tr = ex.fetch_my_trades(sym, limit=args.limit)
    out = [
        {
            "time": t.get("datetime"),
            "symbol": t.get("symbol"),
            "side": t.get("side"),
            "amount": t.get("amount"),
            "price": t.get("price"),
            "cost": t.get("cost"),
        }
        for t in tr
    ]
    print(json.dumps(out, indent=2))

def cmd_open(args):
    ex = ex_binance_vision()
    sym = args.symbol
    oo = ex.fetch_open_orders(sym) if sym else ex.fetch_open_orders()
    out = [
        {
            "id": o.get("id") or o.get("orderId"),
            "symbol": o.get("symbol"),
            "side": o.get("side"),
            "type": o.get("type"),
            "status": o.get("status"),
            "amount": o.get("amount"),
            "price": o.get("price"),
            "filled": o.get("filled"),
        }
        for o in oo
    ]
    print(json.dumps(out, indent=2))

def cmd_orders(args):
    ex = ex_binance_vision()
    sym = args.symbol
    od = ex.fetch_orders(sym, limit=args.limit)
    out = [
        {
            "time": o.get("datetime"),
            "id": o.get("id") or o.get("orderId"),
            "symbol": o.get("symbol"),
            "side": o.get("side"),
            "type": o.get("type"),
            "status": o.get("status"),
            "amount": o.get("amount"),
            "price": o.get("price"),
            "filled": o.get("filled"),
        }
        for o in od
    ]
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="vision_cli", description="Binance Spot Testnet (VISION) – visor")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("balance", help="Ver balances (totales).")
    pb.add_argument("--assets", help="CSV de activos a filtrar (ej: USDT,BTC)")
    pb.set_defaults(func=cmd_balance)

    pt = sub.add_parser("trades", help="Mis trades recientes.")
    pt.add_argument("--symbol", default="BTC/USDT")
    pt.add_argument("--limit", type=int, default=10)
    pt.set_defaults(func=cmd_trades)

    po = sub.add_parser("open", help="Órdenes abiertas.")
    po.add_argument("--symbol", help="Filtrar por símbolo (opcional)")
    po.set_defaults(func=cmd_open)

    po2 = sub.add_parser("orders", help="Histórico de órdenes.")
    po2.add_argument("--symbol", default="BTC/USDT")
    po2.add_argument("--limit", type=int, default=20)
    po2.set_defaults(func=cmd_orders)

    args = p.parse_args()
    args.func(args)
PY
chmod +x "$HOME/PycharmProjects/Bot_BTC/scripts/mini_accum/vision_cli.py"