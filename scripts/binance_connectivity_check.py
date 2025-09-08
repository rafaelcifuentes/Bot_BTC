import socket
import sys

# Opcional: si tienes requests instalado, probamos HTTPS directo
try:
    import requests
except Exception:
    requests = None

def check_tcp(host: str, port: int = 443, timeout: int = 5):
    try:
        import socket as _socket
        with _socket.create_connection((host, port), timeout=timeout):
            return True, None
    except Exception as e:
        return False, repr(e)

def check_https_time(endpoint: str, timeout: int = 7):
    if requests is None:
        return None, "requests no instalado"
    try:
        r = requests.get(endpoint, timeout=timeout)
        ok = (r.status_code == 200) and ("serverTime" in r.text)
        return ok, f"HTTP {r.status_code} body={r.text[:120]!r}"
    except Exception as e:
        return False, repr(e)

def check_ccxt(exchange_id: str):
    try:
        import ccxt
        ex_class = getattr(ccxt, exchange_id)
        ex = ex_class({"enableRateLimit": True})
        t = ex.fetch_time()
        return True, f"{exchange_id}.fetch_time={t}"
    except Exception as e:
        return False, repr(e)

# Mapeo de endpoints/host/símbolo de ejemplo por exchange
EX_ENDPOINTS = {
    "binance":   ("api.binance.com",   "https://api.binance.com/api/v3/time",   "BTC/USDT"),
    "binanceus": ("api.binance.us",    "https://api.binance.us/api/v3/time",    "BTC/USD"),
}

def run_for(exchange_id: str):
    host, endpoint, symbol = EX_ENDPOINTS[exchange_id]

    print(f"== TCP 443 {host} ==")
    ok_tcp, info_tcp = check_tcp(host)
    print("OK" if ok_tcp else "FAIL", info_tcp)

    path = endpoint.replace("https://", "")
    print(f"\n== HTTPS GET /{path.split('/', 1)[1]} ==")
    ok_http, info_http = check_https_time(endpoint)
    print("OK" if ok_http else "FAIL", info_http)

    print(f"\n== ccxt.{exchange_id}() fetch_time ==")
    ok_ccxt, info_ccxt = check_ccxt(exchange_id)
    print("OK" if ok_ccxt else "FAIL", info_ccxt)

    print("\nSugerencias:")
    if not ok_tcp:
        print("- Si TCP FAIL → bloqueo de red/cortafuegos/DNS.")
    if ok_tcp and not ok_http:
        print("- Si TCP OK pero HTTPS FAIL con 403/451/1020 → bloqueo geográfico/Cloudflare.")
    print(f"- Exchange: {exchange_id} | Símbolo spot típico: {symbol}")

    return ok_tcp, ok_http, ok_ccxt

def parse_exchange(argv):
    # Valores aceptados: binance | binanceus | auto (por defecto)
    if not argv:
        return "auto"
    for i, a in enumerate(argv):
        a = a.lower()
        if a in ("--exchange", "-e") and i + 1 < len(argv):
            return argv[i + 1].lower()
        if a == "--binanceus":
            return "binanceus"
        if a == "--binance":
            return "binance"
    return "auto"

def main():
    ex = parse_exchange(sys.argv[1:])

    if ex == "auto":
        print("Modo AUTO: probando binance y si hay 451/403/1020, se probará binanceus…\n")
        _, ok_http, ok_ccxt = run_for("binance")
        # Si HTTPS o ccxt fallan, probamos binanceus como fallback
        if not ok_http or not ok_ccxt:
            print("\n——— Fallback a binanceus ———\n")
            run_for("binanceus")
    elif ex in EX_ENDPOINTS:
        run_for(ex)
    else:
        print("Uso: python binance_connectivity_check.py [--exchange binance|binanceus|auto] | [-e binanceus] | [--binance] | [--binanceus]")
        sys.exit(2)

if __name__ == "__main__":
    main()