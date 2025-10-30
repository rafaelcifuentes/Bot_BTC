#!/bin/zsh
set -euo pipefail
: ${ROOT:="$HOME/PycharmProjects/Bot_BTC"}
DAY="dayN_$(date -u +%F)"; OUTD="$ROOT/evidence/$DAY"; mkdir -p "$OUTD"

# Busca OHLC diarios y semanales locales; si no hay, salimos en silencio (sombra)
DAILY=$(ls -1 $ROOT/data/snapshots/*/1d/BTC-USD.csv 2>/dev/null | tail -n1 || true)
WEEKLY=$(ls -1 $ROOT/data/snapshots/*/1w/BTC-USD.csv 2>/dev/null | tail -n1 || true)
[[ -n "$DAILY" && -n "$WEEKLY" ]] || { echo "[SKIP] lab4_shadow: no hay OHLC 1d/1w locales"; exit 0; }

$ROOT/.venv/bin/python - "$DAILY" "$WEEKLY" "$OUTD" <<'PY'
import sys, csv, statistics, json, pathlib
from datetime import datetime

dpath, wpath, outd = map(pathlib.Path, sys.argv[1:4])

def read_csv(p):
    with p.open() as f:
        r=csv.DictReader(f)
        rows=[{k: v for k,v in row.items()} for row in r]
    return rows

def sma(vals, n):
    if len(vals)<n: return None
    return sum(vals[-n:])/n

daily=read_csv(dpath)
weekly=read_csv(wpath)

close_d=[float(r.get('close') or r.get('Close') or r.get('close_price') or 0.0) for r in daily]
close_w=[float(r.get('close') or r.get('Close') or r.get('close_price') or 0.0) for r in weekly]

SMA200_d=sma(close_d,200)
SMA200_w=sma(close_w,200)
last_d=close_d[-1] if close_d else None

# Heurística sombra Bull-guard:
# - Doble confirmación:
#   a) daily < SMA200_d
#   b) dos semanas cerradas sin recuperar >= SMA200_w (aprox: las últimas 10 sesiones diarias por debajo de SMA200_w)
below_w = [c for c in close_d[-10:] if SMA200_w and c < SMA200_w]
two_weeks_below = (len(below_w) >= 10)
shadow_block_sell = (last_d is not None and SMA200_d and last_d < SMA200_d and two_weeks_below)

out = {
  "ts_utc": datetime.utcnow().isoformat()+"Z",
  "last_close_d": last_d,
  "SMA200_d": SMA200_d,
  "SMA200_w": SMA200_w,
  "two_weeks_below_w": two_weeks_below,
  "shadow_block_sell": bool(shadow_block_sell),
  "note": "LAB4 Bull-guard en sombra; no afecta lógica. Si true, habría filtrado SELL."
}
(pathlib.Path(outd)/"LAB4_BG2w.shadow.json").write_text(json.dumps(out, indent=2))
(pathlib.Path(outd)/"LAB4_BG2w.shadow.md").write_text(
  "# LAB4 Bull-guard (sombra)\n\n"
  f"- last_close_d: {last_d}\n"
  f"- SMA200_d: {SMA200_d}\n"
  f"- SMA200_w: {SMA200_w}\n"
  f"- two_weeks_below_w: {two_weeks_below}\n"
  f"- shadow_block_sell: {bool(shadow_block_sell)}\n"
)
print("[OK] lab4_shadow: evidencia generada")
PY
