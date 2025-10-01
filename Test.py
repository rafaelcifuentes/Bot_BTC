#echo ""
#python3 - "$ROOT" <<'PY'
import json, os, re, datetime, math

root = os.environ.get('ROOT', os.path.expanduser('~/PycharmProjects/Bot_BTC'))

def read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

# --- Estado (latest.json) ---
latest = read_json(os.path.join(root, 'signals', 'mini_accum', 'latest.json')) or {}
estado = latest.get('health') or latest.get('status') or "n/a"
ts = latest.get('ts_utc') or "n/a"
pos = latest.get('position_pct_btc')
guards = latest.get('guards', {}) if isinstance(latest.get('guards'), dict) else {}

# --- NetBTC vs HODL (perf_seal.json si existe) ---
perf = read_json(os.path.join(root, 'reports', 'mini_accum', 'perf_seal.json')) or {}
def pick(d, *keys):
    for k in keys:
        if k in d:
            return d[k]
    return None

netbtc = pick(perf, 'net_btc', 'netbtc', 'NetBTC', 'netBTC')
hodl   = pick(perf, 'hodl_btc', 'HODL_BTC', 'hodl')

# --- Flips y FPY estimado (de cron.log últimos 7 días) ---
cron_path = os.path.join(root, 'logs', 'cron.log')
flips_7 = 0
try:
    now = datetime.datetime.utcnow()
    cutoff = now - datetime.timedelta(days=7)
    pat = re.compile(r'flip', re.I)
    with open(cron_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z', line)
            if not m:
                continue
            t = datetime.datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%S')
            if t >= cutoff and pat.search(line):
                flips_7 += 1
    fpy = round(flips_7 * 365.0 / 7.0, 2)
except Exception:
    fpy = None

print("-- Dashboard simple --")
print(f"Estado: {estado} @ {ts}")
if isinstance(pos, (int, float)):
    print(f"Posición BTC: {pos:.4f}")
else:
    print("Posición BTC: n/a")

if isinstance(netbtc, (int, float)) and isinstance(hodl, (int, float)):
    diff = netbtc - hodl
    print(f"NetBTC vs HODL: {netbtc:.6f} vs {hodl:.6f} (Δ={diff:+.6f})")
else:
    print("NetBTC vs HODL: n/a (esperando perf_seal.json)")

if fpy is not None:
    print(f"Flips 7d: {flips_7}  |  FPY≈ {fpy}")
else:
    print("FPY: n/a (no se detectan flips en cron.log)")
#PY
