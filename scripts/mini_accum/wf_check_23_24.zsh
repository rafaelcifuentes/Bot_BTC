#!/usr/bin/env zsh
set -euo pipefail
ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
VENV="${VENV:-$ROOT/.venv}"
PRESET="${PRESET:-$ROOT/configs/mini_accum/presets/CORE_2025.yaml}"
EPS="${EPS:-0.000002}"

# Valores canónicos del Santo Grial (CORE_2025)
typeset -A EXP_SATS
EXP_SATS[2023]=2.641397
EXP_SATS[2024]=1.613240

. "$VENV/bin/activate"

run_year() {
  local Y="$1"; local SUF="WF_${Y}_core_recheck"
  echo "[RUN] mini_accum.cli ${Y} (${Y}-01-01..${Y}-12-31)"
  python -m mini_accum.cli --config "$PRESET" --start "${Y}-01-01" --end "${Y}-12-31" --suffix "$SUF"

  # Localiza KPI recién creado
  local KPI_CSV
  KPI_CSV="$(ls -1t "$ROOT"/reports/mini_accum/*_kpis__${SUF}.csv | head -n1 || true)"
  [[ -s "$KPI_CSV" ]] || { echo "[ERR] KPI vacío para $Y"; return 1; }
  echo "[KPI] $KPI_CSV"

  # Lee campos con tolerancia (corrige el bug de comillas rotas)
  python - "$KPI_CSV" "$Y" "${EXP_SATS[$Y]}" "$EPS" <<'PY'
import sys, csv, math
kpi, year, exp_s, eps = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])

def F(x):
    try: return float(x)
    except: return math.nan

with open(kpi, newline='') as f:
    row = next(csv.DictReader(f), {})

sats = row.get("sats_mult") or row.get("net_btc_ratio") or ""
mddv = row.get("mdd_vs_hodl")
if not mddv:
    mm = row.get("mdd_model_usd") or row.get("mdd_model_btc") or row.get("mdd_model")
    mh = row.get("mdd_hodl_usd")  or row.get("mdd_hodl_btc")  or row.get("mdd_hodl")
    if mm and mh and F(mh) > 0: mddv = str(F(mm)/F(mh))
flips = row.get("flips_total") or row.get("flips") or ""

print(f"[RESULT] {year} CORE_2025 → sats_mult={sats} | mdd_vs_hodl={mddv} | flips={flips}")

got = F(sats)
if math.isnan(got):
    print(f"[CHECK] {year}: sats_mult inválido ({sats})")
else:
    ok = abs(got - exp_s) <= eps
    print(f"[CHECK] {year}: got={got:.6f} vs exp={exp_s:.6f} (eps={eps:.0e}) → {'PASS' if ok else 'FAIL'}")
PY
}

echo "[DATA HASH] 4h: $(shasum -a 256 "$ROOT/data/ohlc/4h/BTC-USD.csv" 2>/dev/null | awk '{print $1}')"
echo "[DATA HASH] 1d: $(shasum -a 256 "$ROOT/data/ohlc/1d/BTC-USD.csv" 2>/dev/null | awk '{print $1}')"

# Cabeceras para reconfirmar OHLCV
echo "== HEADERS 4h =="
head -n1 "$ROOT/data/ohlc/4h/BTC-USD.csv" | tr ',' '\n' | nl -ba
echo "== HEADERS 1d =="
head -n1 "$ROOT/data/ohlc/1d/BTC-USD.csv" | tr ',' '\n' | nl -ba

run_year 2023
run_year 2024
echo "[DONE] Recheck 2023/2024"
