
#!/usr/bin/env zsh
set -euo pipefail

### =======================
### Config y defaults (zsh)
### =======================
ROOT="${ROOT:-$HOME/PycharmProjects/Bot_BTC}"
VENV="${VENV:-$ROOT/.venv}"
PRESET="${PRESET:-$ROOT/configs/mini_accum/presets/CORE_2025.yaml}"   # KISS v1 TOP
EX="${EX:-binanceus}"            # coinbase|binanceus|kraken|...
SYMBOL="${SYMBOL:-BTC/USD}"
TF4H="${TF4H:-4h}"               # Para coinbase usa segundos: 21600
TF1D="${TF1D:-1d}"               # Para coinbase usa segundos: 86400
START_ISO="${START_ISO:-2025-01-01T00:00:00Z}"
END_ISO="${END_ISO:-2025-10-31T23:59:59Z}"

# Rutas de datos que usa el motor
DST4H="$ROOT/data/ohlc/4h/BTC-USD.csv"
DST1D="$ROOT/data/ohlc/1d/BTC-USD.csv"

# Ingesta + backups
ING_DIR="$ROOT/data/_ingest"
BK_DIR="$ROOT/data/_bk"

mkdir -p "$ING_DIR"/{4h,1d} "$BK_DIR" "$ROOT/reports/mini_accum"

export ROOT VENV PRESET EX SYMBOL TF4H TF1D START_ISO END_ISO DST4H DST1D

echo "[DEBUG] ROOT=$ROOT"
echo "[DEBUG] PRESET=$PRESET"
echo "[DEBUG] EX=$EX SYMBOL=$SYMBOL RANGE=$START_ISO → $END_ISO"
echo "[DEBUG] 4H dst=$DST4H | 1D dst=$DST1D"

# Activa venv
. "$VENV/bin/activate"

### =======================
### Ingesta + Merge tolerante
### =======================
python - <<'PY'
import os, sys, time, hashlib
from datetime import datetime, timezone
import pandas as pd

try:
    import ccxt  # noqa
except Exception as e:
    sys.exit(f"[ERR] Necesitas ccxt en el venv: {e}")

ROOT      = os.environ["ROOT"]
EX_ID     = os.environ["EX"]
SYMBOL    = os.environ["SYMBOL"]
TF4H      = os.environ["TF4H"]
TF1D      = os.environ["TF1D"]
START_ISO = os.environ["START_ISO"]
END_ISO   = os.environ["END_ISO"]

ING_DIR = os.path.join(ROOT, "data/_ingest")
DST4H   = os.environ["DST4H"]
DST1D   = os.environ["DST1D"]
BK_DIR  = os.path.join(ROOT, "data/_bk")

def ms(dt: str) -> int:
    # Soporta "Z"
    return int(datetime.fromisoformat(dt.replace("Z", "+00:00")).timestamp() * 1000)

def iso(ms_: int) -> str:
    return datetime.utcfromtimestamp(ms_/1000).replace(tzinfo=timezone.utc).isoformat().replace("+00:00","Z")

def fetch_ohlcv_full(ex, symbol, timeframe, start_ms, end_ms, limit=500, sleep_s=0.2):
    out = []
    since = start_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        out.extend(batch)
        last_ts = batch[-1][0]
        next_since = last_ts + 1
        if next_since >= end_ms:
            break
        since = next_since
        time.sleep(sleep_s)
    return out

def to_df(rows):
    # ccxt OHLCV: [timestamp, open, high, low, close, volume]
    cols = ["timestamp","open","high","low","close","volume"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return df[cols]

def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def backup_if_exists(path, bk_dir):
    import shutil, time
    if os.path.exists(path) and os.path.getsize(path) > 0:
        s = sha256(path)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        base = os.path.basename(path)
        bk = os.path.join(bk_dir, f"{base}.{ts}__{s}.csv")
        shutil.copy2(path, bk)
        print(f"[BKUP] {path} -> {bk}")

def read_csv_tolerant(path):
    """
    Lee CSV con alta tolerancia:
    - Acepta encabezados variados: timestamp|ts|date|datetime|time (case-insensitive)
    - Si no hay encabezado, intenta header=None y asume primera columna es timestamp
    - Devuelve df con al menos 'timestamp' y, si hay, 'open','high','low','close','volume'
    - Normaliza timestamp a 'YYYY-MM-DDTHH:MM:SS+00:00'
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    try:
        df = pd.read_csv(path, dtype=str, engine="python")
    except Exception:
        # Último recurso: sin encabezado
        df = pd.read_csv(path, dtype=str, header=None, engine="python")

    # Normaliza nombres
    cols = {c.lower().strip(): c for c in df.columns}
    ts_col = None
    for cand in ("timestamp","ts","date","datetime","time"):
        if cand in cols:
            ts_col = cols[cand]
            break
    if ts_col is None:
        # Puede venir sin encabezado: asumimos primera
        ts_col = df.columns[0]

    # Reconstruye DataFrame con columnas estándar si existen
    std = {}
    std["timestamp"] = df[ts_col].astype(str)
    # Intentar mapear OHLCV si están
    name_map = {}
    for k in ("open","high","low","close","volume"):
        for cand in (k, k.upper(), k.capitalize()):
            if cand in df.columns:
                name_map[k] = cand
                break
    for k in ("open","high","low","close","volume"):
        std[k] = df[name_map[k]].astype(str) if k in name_map else None

    out = pd.DataFrame(std)
    # Parseo seguro de timestamp
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    # Si no hay OHLCV, quizás era (timestamp,close) únicamente
    if out.get("open") is None and out.get("close") is None:
        if df.shape[1] >= 2 and df.columns[1] != ts_col:
            try:
                close_series = pd.to_numeric(df[df.columns[1]], errors="coerce")
                out["close"] = close_series
            except Exception:
                pass

    # Filtra columnas válidas
    keep = ["timestamp"]
    for k in ("open","high","low","close","volume"):
        if k in out.columns:
            keep.append(k)
    out = out[keep].copy()
    return out

def align_columns(target_like: pd.DataFrame, new_df: pd.DataFrame):
    """
    Alinea columnas del nuevo DF para mergear con el existente:
    - Si el target es (timestamp,close), reduce new_df a ese esquema.
    - Si el target es OHLCV, asegura ese esquema (rellena close si falta).
    """
    tcols = [c for c in target_like.columns if c != "timestamp"]
    if tcols == ["close"] or (len(tcols)==1 and tcols[0].lower()=="close"):
        # Reducir a timestamp,close
        df = new_df.copy()
        if "close" not in df.columns:
            # Deriva close desde OHLCV si existe
            if all(k in df.columns for k in ("open","high","low","close","volume")):
                pass
            else:
                raise ValueError("[ALIGN] No hay columna 'close' en ingesta para esquema (timestamp,close).")
        return df[["timestamp","close"]].copy()
    else:
        # Esquema rico (OHLCV). Si faltan algunas, intenta completar mínimo 'close'
        need = ["open","high","low","close","volume"]
        df = new_df.copy()
        has_ohlcv = all(k in df.columns for k in need)
        if not has_ohlcv and "close" in df.columns and len(tcols)==5:
            out = pd.merge(
                target_like[["timestamp"]].copy(),
                df[["timestamp","close"]],
                on="timestamp",
                how="left"
            )
            for k in need:
                if k not in out.columns:
                    out[k] = pd.NA
            return out[["timestamp"]+need].copy()

        for k in need:
            if k not in df.columns:
                df[k] = pd.NA
        return df[["timestamp"]+need].copy()

def safe_merge_write(dst_path, ingested_df):
    """
    Merge tolerante por 'timestamp':
    - Lee existente de forma tolerante
    - Alinea columnas
    - De-dup + sort
    - Si algo falla, fallback: backup y overwrite seguro
    """
    try:
        existing = read_csv_tolerant(dst_path)
        if existing.empty:
            out = ingested_df.copy()
        else:
            aligned_new = align_columns(existing, ingested_df)
            for k in aligned_new.columns:
                if k != "timestamp":
                    aligned_new[k] = pd.to_numeric(aligned_new[k], errors="coerce")
            for k in existing.columns:
                if k != "timestamp":
                    existing[k] = pd.to_numeric(existing[k], errors="coerce")
            out = pd.concat([existing, aligned_new], ignore_index=True)
    except Exception as e:
        print(f"[WARN] Merge tolerante falló: {e} -> fallback overwrite.")
        backup_if_exists(dst_path, BK_DIR)
        if "close" in ingested_df.columns:
            out = ingested_df[["timestamp","close"]].copy()
        else:
            out = ingested_df.copy()

    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out.to_csv(dst_path, index=False)
    s = sha256(dst_path)
    return len(ingested_df), len(out), s

# Construye exchange
ex_kwargs = {"enableRateLimit": True}
ex = getattr(ccxt, EX_ID)(ex_kwargs)

start_ms, end_ms = ms(START_ISO), ms(END_ISO)
print(f"[RANGE] {iso(start_ms)} → {iso(end_ms)} UTC")

# 1) 4H
rows4 = fetch_ohlcv_full(ex, SYMBOL, TF4H, start_ms, end_ms)
df4   = to_df(rows4)
ing4  = os.path.join(ING_DIR, "4h", f"BTC-USD__{TF4H}__{iso(start_ms)}_{iso(end_ms)}.csv".replace(":",""))
df4.to_csv(ing4, index=False)
print(f"[INGEST] 4H → {ing4} ({len(df4)} filas)")

# 2) 1D
rows1 = fetch_ohlcv_full(ex, SYMBOL, TF1D, start_ms, end_ms)
df1   = to_df(rows1)
ing1  = os.path.join(ING_DIR, "1d", f"BTC-USD__{TF1D}__{iso(start_ms)}_{iso(end_ms)}.csv".replace(":",""))
df1.to_csv(ing1, index=False)
print(f"[INGEST] 1D → {ing1} ({len(df1)} filas)")

# 3) Backups (del archivo destino real) y merge tolerante
backup_if_exists(DST4H, BK_DIR)
backup_if_exists(DST1D, BK_DIR)

added4h, total4h, sha4h = safe_merge_write(DST4H, df4)
added1d, total1d, sha1d = safe_merge_write(DST1D, df1)

print(f"[WRITE] {DST4H} (+{added4h} → {total4h}) sha256={sha4h}")
print(f"[WRITE] {DST1D} (+{added1d} → {total1d}) sha256={sha1d}")
PY

### =======================
### Ejecuta motor KISS v1
### =======================
echo "[RUN] mini_accum.cli con CORE_2025 sobre 2025-01-01..2025-10-31"
python -m mini_accum.cli \
  --config "$PRESET" \
  --start "${START_ISO%%T*}" --end "${END_ISO%%T*}" \
  --suffix OOS_2025H1_core_from_newdata

# Renombra artefactos a sufijo consistente si hiciera falta
last="$(ls -1t "$ROOT"/reports/mini_accum/*_kpis*.csv | head -n1 || true)"
if [[ -n "${last:-}" ]]; then
  base="${last%_kpis*}"
  dir="$(dirname "$last")"
  stamp="$(basename "$base" | sed 's/_kpis$//')"
  if [[ "$last" != *"__OOS_2025H1_core_from_newdata.csv" ]]; then
    for ext in equity kpis summary flips; do
      f="$dir/${stamp}_${ext}.csv"
      [[ "$ext" == "summary" ]] && f="$dir/${stamp}_${ext}.md"
      if [[ -s "$f" ]]; then
        ext="${f##*.}"
        nf="${f%.$ext}__OOS_2025H1_core_from_newdata.${ext}"
        mv "$f" "$nf"
        echo "[RENAMED] $(basename "$f") -> $(basename "$nf")"
      fi
    done
  fi
fi

### =======================
### KPIs (tolerantes)
### =======================
KPI_CSV="$(ls -1t "$ROOT"/reports/mini_accum/*_kpis__OOS_2025H1_core_from_newdata.csv | head -n1 || true)"
if [[ -z "${KPI_CSV:-}" || ! -s "$KPI_CSV" ]]; then
  echo "[ERR] No encontré KPI nuevo"
  exit 1
fi
echo "[KPI] $KPI_CSV"

python - "$KPI_CSV" <<'PY'
import sys, csv, math
kpi = sys.argv[1]
with open(kpi, newline='') as f:
    r = next(csv.DictReader(f), {})
def F(x):
    try: return float(x)
    except: return math.nan
sats  = r.get("sats_mult") or r.get("net_btc_ratio")
mddv  = r.get("mdd_vs_hodl
")
if not mddv:
    mm = r.get("mdd_model_usd") or r.get("mdd_model_btc") or r.get("mdd_model")
    mh = r.get("mdd_hodl_usd")  or r.get("mdd_hodl_btc")  or r.get("mdd_hodl")
    if mm and mh and F(mh) > 0:
        mddv = str(F(mm) / F(mh))
flips = r.get("flips_total") or r.get("flips") or ""
print(f"[RESULT] 2025H1 CORE_2025 → sats_mult={sats} | mdd_vs_hodl={mddv} | flips={flips}")
PY
