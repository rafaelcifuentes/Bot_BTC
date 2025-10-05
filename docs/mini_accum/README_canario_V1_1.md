# Mini-Accum V1.1 — Canario (SL/TP defensivo ATR)

**Método:** ATR(14), **SL=2×**, **TP=3×**, `fix_on_entry=true`  
**Ámbito:** despliegue canario con **10–20%** del capital. V1.0 queda **en sombra** para A/B.  

**Guardrails 30d:**  
- **ΔMDD ≤ 0**  
- **ΔFPY ≤ +2**  
- **ΔROI_anual ≥ −4%**  
→ cualquier violación ⇒ **rollback** a CORE.

```
> **Estado SPA/RC (2025-10-05)**  
> - Ago-2023: `p_consistent=0.545` → **FAIL**  
> - Q3-2024: `p_consistent=0.545` → **FAIL** *(curvas idénticas → neutral)*  
> - Q2-2025: `p_consistent=0.545` → **FAIL** *(curvas idénticas → neutral)*  
>
> **Decisión:** mantener **canario** (10–20%) con guardrails activos; **promoción bloqueada** hasta PASS ≥ 0.60.  
> **Siguiente paso:** medir **ATR 2.5×** y/o **`reentry_ttl=8–12`** en Ago-2023 y Q3-2024 y repetir SPA/RC.
```

---

## 1) Qué incluye V1.1

- Preset/pipeline con **SL/TP defensivo por ATR** (14) activado.
- **No** cambia lógica de señales CORE fuera de SL/TP; objetivo: “seguro barato” ante mini-crash.
- Overlays/presets de canario (congélalos en `configs/mini_accum/presets/canary/`):
  - `CORE_2025_ATR14x2_0.yaml`  (SL=2×, TP=3×)
  - (opcional estudio) `CORE_2025_ATR14x2_5.yaml` o `reentry_ttl=8–12`

> En los comandos de abajo se usa el preset **ATR 2×3**:  
> `configs/mini_accum/presets/_kt_tmp/CORE_2025_ATR14x2_0.yaml`  
> (Si ya lo congelaste en `.../canary/`, cambia la ruta.)

---

## 2) Ventanas de validación (resultado esperado)

- **Ago-2023 (mini-crash):** `Δmult≈−0.0068` ✅, `ΔROI_anual≈−3.41%` (❌ SLO estricto −3% pero **> −4% guardrail** ✅), **ΔMDD ↓** ✅, **ΔFPY ≤ +2** ✅  
- **Q3-2024 (limpia):** `Δmult=0`, `ΔROI=0`, **MDD igual**, **FPY igual** ✅  
- **Q2-2025 (limpia):** `Δmult=0`, `ΔROI=0`, **MDD igual**, **FPY igual** ✅

**SPA/RC requerido:** **PASS ≥ 0.60** antes de promover.

---

## 3) Repro OOS (neutras) — Q3-2024 y Q2-2025

> Usa **histórico largo** y **D1 close-only**. Si aún no tienes el CSV D1 close, créalo (ver §6).

```zsh
# Presets
CORE="configs/mini_accum/presets/CORE_2025.yaml"
ATR20="configs/mini_accum/presets/_kt_tmp/CORE_2025_ATR14x2_0.yaml"

# Datos (overrides por ENV)
export OHLC_4H_CSV="data/snapshots/full_history/4h/BTC-USD.csv"
export OHLC_D1_CSV="data/snapshots/full_history/1d/BTC-USD_close.csv"

# CORE — Q3-2024
START=2024-06-15 END=2024-08-15 PRESET="$CORE" SUF="Q3_2024" \
bash scripts/mini_accum/run_oos.sh

# ATR 2×3 — Q3-2024
START=2024-06-15 END=2024-08-15 PRESET="$ATR20" SUF="Q3_2024_ATR2x3" \
bash scripts/mini_accum/run_oos.sh

# CORE — Q2-2025
START=2025-03-15 END=2025-05-31 PRESET="$CORE" SUF="Q2_2025" \
bash scripts/mini_accum/run_oos.sh

# ATR 2×3 — Q2-2025
START=2025-03-15 END=2025-05-31 PRESET="$ATR20" SUF="Q2_2025_ATR2x3" \
bash scripts/mini_accum/run_oos.sh
```

**Artefactos (ejemplos actuales):**
- Q3-2024 CORE:  
  - `reports/mini_accum/base_v0_1_20251005_0151_equity__Q3_2024.csv`  
  - `reports/mini_accum/base_v0_1_20251005_0151_flips__Q3_2024.csv`
- Q3-2024 ATR 2×3:  
  - `reports/mini_accum/base_v0_1_20251005_0153_equity__Q3_2024_ATR2x3.csv`  
  - `reports/mini_accum/base_v0_1_20251005_0153_flips__Q3_2024_ATR2x3.csv`
- Q2-2025 CORE:  
  - `reports/mini_accum/base_v0_1_20251005_0151_equity__Q2_2025.csv`  
  - `reports/mini_accum/base_v0_1_20251005_0151_flips__Q2_2025.csv`
- Q2-2025 ATR 2×3:  
  - `reports/mini_accum/base_v0_1_20251005_0153_equity__Q2_2025_ATR2x3.csv`  
  - `reports/mini_accum/base_v0_1_20251005_0153_flips__Q2_2025_ATR2x3.csv`

> Si los nombres rotan por fecha/hora, usa:  
> `ls -1t reports/mini_accum/*_equity__Q3_2024.csv | head -n1`

---

## 4) Verificación de KPIs (Δmult, ΔROI, MDD, FPY)

### 4.1 Deltas (equity_metrics)
```zsh
# Q3-2024
BASE_EQ=(reports/mini_accum/*_equity__Q3_2024.csv(Nom[1]))
CAND_EQ=(reports/mini_accum/*_equity__Q3_2024_ATR2x3.csv(Nom[1]))

python3 scripts/mini_accum/equity_metrics.py \
  --equity-base "$BASE_EQ" --equity-cand "$CAND_EQ" \
  --start 2024-06-15 --end 2024-08-15 --unit btc
```

### 4.2 MDD (snippet seguro)
```zsh
python3 - <<'PY' "$BASE_EQ" "$CAND_EQ" 2024-06-15 2024-08-15
import sys,pandas as pd
def s(p):
    df=pd.read_csv(p); ts=pd.to_datetime(df.get('timestamp',df.get('ts')), utc=True)
    eq=df.get('equity_btc', df.get('equity')); return pd.Series(eq.values, index=ts).dropna()
b=s(sys.argv[1]).loc[sys.argv[3]:sys.argv[4]]
c=s(sys.argv[2]).loc[sys.argv[3]:sys.argv[4]]
m=lambda x:(x/x.cummax()-1).min()
print(f"[MDD] base={m(b):.2%} cand={m(c):.2%} Δ={m(c)-m(b):+.2%}")
PY
```

### 4.3 FPY (conteo BUY/SELL/SELL_SLTP)
```zsh
BASE_FLIPS=(reports/mini_accum/*_flips__Q3_2024.csv(Nom[1]))
CAND_FLIPS=(reports/mini_accum/*_flips__Q3_2024_ATR2x3.csv(Nom[1]))
python3 - <<'PY' "$BASE_FLIPS" "$CAND_FLIPS" 2024-06-15 2024-08-15
import sys,datetime,csv
def count(p,pat):
    n=0
    with open(p) as f:
        r=csv.DictReader(f)
        for row in r:
            if (row.get('executed') or '') in pat: n+=1
    return n
nb=count(sys.argv[1],{'BUY','SELL'})
np=count(sys.argv[2],{'BUY','SELL','SELL_SLTP'})
s=datetime.date.fromisoformat(sys.argv[3]); e=datetime.date.fromisoformat(sys.argv[4])
days=(e-s).days or 1; f=lambda n:n*365/days
print(f"FPY_base≈{f(nb):.2f}/a | FPY_cand≈{f(np):.2f}/a | Δ≈{f(np)-f(nb):+.2f}/a")
PY
```

---

## 5) SPA / Reality-Check (PASS ≥ 0.60)

```zsh
# Q3-2024
python3 scripts/mini_accum/spa_reality_check.py \
  --equity-a "$BASE_EQ" --equity-b "$CAND_EQ" \
  --start 2024-06-15 --end 2024-08-15 \
  --trials 1000 --seed 42 --json > reports/mini_accum/spa_Q3_2024_ATR2x3.json

# Q2-2025
BASE_EQ2=(reports/mini_accum/*_equity__Q2_2025.csv(Nom[1]))
CAND_EQ2=(reports/mini_accum/*_equity__Q2_2025_ATR2x3.csv(Nom[1]))
python3 scripts/mini_accum/spa_reality_check.py \
  --equity-a "$BASE_EQ2" --equity-b "$CAND_EQ2" \
  --start 2025-03-15 --end 2025-05-31 \
  --trials 1000 --seed 42 --json > reports/mini_accum/spa_Q2_2025_ATR2x3.json
```

**Ago-2023 (activo, post-sim):**
- Base: `reports/mini_accum/*_equity__CORE_2025.csv`
- Candidato: `reports/mini_accum/post_*_equity____ATR2x3_post.csv`

```zsh
python3 scripts/mini_accum/spa_reality_check.py \
  --equity-a "$(ls -1t reports/mini_accum/*_equity__CORE_2025.csv | head -n1)" \
  --equity-b "$(ls -1t reports/mini_accum/post_*_equity____ATR2x3_post.csv | head -n1)" \
  --start 2023-08-01 --end 2023-09-30 \
  --trials 1000 --seed 42 --json > reports/mini_accum/spa_Ago2023_ATR2x3.json
```

---

## 6) Datos (CSV) y errores típicos

- **Recomendado para canario:**  
  - `OHLC_4H_CSV="data/snapshots/full_history/4h/BTC-USD.csv"` (tiene `open,high,low,close,volume`)  
  - `OHLC_D1_CSV="data/snapshots/full_history/1d/BTC-USD_close.csv"` (solo `timestamp,close`)

- **Crear D1 close-only a partir de 4h:**
```zsh
python3 - <<'PY'
import pandas as pd, os
src="data/snapshots/full_history/4h/BTC-USD.csv"
dst="data/snapshots/full_history/1d/BTC-USD_close.csv"
os.makedirs(os.path.dirname(dst), exist_ok=True)
df=pd.read_csv(src, comment='#')
ts=pd.to_datetime(df['timestamp'], utc=True)
df=df.assign(ts=ts).sort_values('ts')
d1 = df.set_index('ts')['close'].resample('1D').last().dropna().reset_index()
d1.rename(columns={'ts':'timestamp'}, inplace=True)
d1.to_csv(dst, index=False)
print("Wrote",dst,"rows",len(d1))
PY
```

- **KeyError `'high'`**  
  Ocurre si algún snippet intenta indexar columnas OHLC sobre archivos que **no son** OHLC (p.ej., `*_equity*.csv` o D1 close). Usa siempre los bloques de verificación de §4.

---

## 7) Playbook de operación canario

**Monitoreo diario (últimos 30 días):**  
- **Rollback inmediato** si **cualquiera**:
  - `ΔMDD_30d > 0`
  - `ΔFPY_30d > +2/año`
  - `ΔROI_anual_30d < −4%`

Snippet para calcular deltas 30d (ajusta rutas si cambian los sufijos):

```zsh
BASE_30=(reports/mini_accum/*_equity__CORE_2025.csv(Nom[1]))
CAND_30=(reports/mini_accum/*_equity__CORE_2025_ATR14x2_0.csv(Nom[1]))
python3 - <<'PY' "$BASE_30" "$CAND_30"
import sys,pandas as pd, datetime as dt
def s(p):
    df=pd.read_csv(p); ts=pd.to_datetime(df.get('timestamp',df.get('ts')), utc=True)
    eq=df.get('equity_btc', df.get('equity')); return pd.Series(eq.values, index=ts).dropna()
b=s(sys.argv[1]); c=s(sys.argv[2])
end=b.index.max().to_pydatetime().date()
start=(end - dt.timedelta(days=30)).isoformat()
b=b.loc[start:]; c=c.loc[start:]
m=lambda x:(x/x.cummax()-1).min()
roi=lambda x:(x.iloc[-1]/x.iloc[0]-1)
print(f"[30d] ΔMDD={m(c)-m(b):+.2%} | ΔROI={roi(c)-roi(b):+.2%}")
PY
```

---

## 8) Checklist PR canario

- [ ] Overlays/presets **congelados** en `configs/mini_accum/presets/canary/`  
- [ ] **Guardrails** documentados en este README  
- [ ] **Validación**: Ago-2023 + Q3-2024 + Q2-2025 (artefactos enlazados)  
- [ ] **SPA/RC** JSONs (PASS ≥ 0.60) guardados en `reports/mini_accum/`  
- [ ] **Tag**: `mini-accum-v1.1-canary`  
- [ ] **Changelog** breve

**Changelog sugerido:**
```md
### V1.1 (Canary) — SL/TP ATR
- SL/TP defensivo: ATR(14), SL=2×, TP=3×, fix_on_entry.
- Resultados: Ago-2023 mejora MDD y Δmult≈−0.0068; ΔROI_anual≈−3.41% (dentro de guardrail −4%).
- Q3-2024 & Q2-2025 neutras: Δmult=0, ΔROI=0, ΔMDD=0, ΔFPY=0.
- Guardrails 30d y SPA/RC ≥ 0.60 requeridos para promover.
```

---

## 9) Enlaces rápidos a artefactos (ejemplos actuales)

- **Q3-2024 CORE:**  
  `reports/mini_accum/base_v0_1_20251005_0151_equity__Q3_2024.csv` · `reports/mini_accum/base_v0_1_20251005_0151_flips__Q3_2024.csv`
- **Q3-2024 ATR 2×3:**  
  `reports/mini_accum/base_v0_1_20251005_0153_equity__Q3_2024_ATR2x3.csv` · `reports/mini_accum/base_v0_1_20251005_0153_flips__Q3_2024_ATR2x3.csv`
- **Q2-2025 CORE:**  
  `reports/mini_accum/base_v0_1_20251005_0151_equity__Q2_2025.csv` · `reports/mini_accum/base_v0_1_20251005_0151_flips__Q2_2025.csv`
- **Q2-2025 ATR 2×3:**  
  `reports/mini_accum/base_v0_1_20251005_0153_equity__Q2_2025_ATR2x3.csv` · `reports/mini_accum/base_v0_1_20251005_0153_flips__Q2_2025_ATR2x3.csv`
- **Ago-2023 post (ATR 2×3):**  
  `reports/mini_accum/post_20251004_032956_equity____ATR2x3_post.csv`

> Si los nombres rotan, usa `ls -1t` para capturar el último archivo con el sufijo.

---

## 10) Notas

- El CLI ya soporta overrides por ENV: `OHLC_4H_CSV` (OHLC completo) y `OHLC_D1_CSV` (close-only).  
- Los snippets de MDD/FPY trabajan **solo** con columnas de equity/flips (evitan `KeyError: 'high'`).  
- Para **2.5×** o `reentry_ttl`, repite la mecánica cambiando `PRESET` y el sufijo `SUF`.
