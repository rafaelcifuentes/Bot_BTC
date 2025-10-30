# 🧪 Gate & Contrato — Mini-Accum (aplica a v2.0 y sucesivas)

## 1) Cláusulas generales (recordatorio, aplican también a v3.0 vs v2.0)
1. **Superset**: v2.0 no elimina palancas de v1 que aportan sats. (Válido para v3.0 vs v2.0, etc.)
2. **NetBTC por año ≥ versión previa** (tolerancia técnica ε = +0.10%).
3. **Lift OOS ≥ +5% vs BASE**.
4. **MDD no peor que BASE**.
5. **Anti-NaN en KPIs** (si hay NaN crítico → FAIL inmediato).
6. **(Estricto)**: Spearman ≥ 0.95 y PBO ≤ 0.30 cuando toque.
7. **Fricción**: si FPY_cand > 2× FPY_base **y** lift < +5% ⇒ FAIL (sobre-operar sin beneficio).
8. **Trazabilidad**: sufijos claros, documentación, tag y plan de rollback.
9. **Nuevas versiones**: **opt-in** (se activan explícitamente tras PASS).

---

## Gate de Aprobación a Producción — Mini-Accum

### A. Cláusulas de Contrato (obligatorias)
1. **Superset de lógica ganadora.**  
   Toda nueva versión debe conservar (o extender de forma ortogonal) las palancas y lógica que hicieron que la versión anterior acumule sats.  
   - No se permite eliminar módulos que aportan sats sin evidencia de que el reemplazo mejora en **todos** los años.  
   - Para **KISS v1**: EMA21/55, filtro macro (SMA200), ADX gate si está ON, TTL (h_bars), rebalanceo semanal (RB1), costos, sin leverage ni shorts, bias=0.

2. **Mejor NetBTC en todos los años (WF/OOS).**  
   La nueva versión debe superar el NetBTC por año de la versión anterior en todas las ventanas disponibles (tolerancia ε = +0.10%).  
   **Base de referencia (v1 KISS):**
   - 2022: **1.018661**
   - 2023: **2.641397**
   - 2024: **1.613240** *(si/cuando esté disponible en tus datos)*
   - 2025H1: **1.138462**  
   **Cumulativos orientativos ya documentados:**
   - 2022→2024: ×**4.340726**
   - 2022→2025H1: ×**4.941751**  
   *Si falta 2024 en datos actuales, se omite hoy, pero será obligatorio cuando se incorpore.*

### B. Métrica clave y riesgo (obligatorio)
3. **Lift OOS actual ≥ +5% vs BASE (NetBTC/SATS).**  
   - `kiss_gate_lift BASE CAND 5 0`  
   - Si \< +5% → **FAIL**.

4. **Riesgo: MDD no peor que la BASE.**  
   - Si `mdd_vs_hodl` empeora → **FAIL**, aunque el lift sea positivo.

5. **Salud del KPI (anti-NaN).**  
   - `assert_kpi_has_sats BASE.csv` y `assert_kpi_has_sats CAND.csv`.  
   - Si falta métrica de sats → **FAIL** inmediato.

### C. Robustez (recomendado/estricto)
6. **Modo estricto para cambios relevantes:**
   - Spearman ≥ **0.95** (ranks de stress/costos).
   - PBO ≤ **0.30**.
   - Ejemplo:  
     ```zsh
     SPEARMAN_CSV=... PBO_VAL=0.27 PBO_MAX=0.30 \
     kiss_gate_lift BASE CAND 5 1
     ```

### D. Fricción operativa y trazabilidad
7. **Fricción**: flips/año no debe duplicar a la BASE sin aportar lift; si `FPY_cand > 2× FPY_base` y `lift < +5%` → **FAIL** (señal de sobre-operar).

8. **Trazabilidad y naming**: candidato con sufijo claro (`*_SLTP_*`, `*_v1_2*`); usa `rename_last_reports "__<SUF>"`.  
   `data/tmp_wf/*.yaml` en `.gitignore`.

9. **Documentación y release**: registra en `docs/mini_accum/decisiones.md` y `Progreso.md` (fecha, versión, lift, mdd_delta, FPY, PASS/FAIL y motivo), crea **tag** y **release notes** si pasa, con plan de rollback.

---

## Snippet — Chequeo “Mejor en todos los años”
Úsalo además del gate de +5% OOS para verificar el contrato multi-año.

```zsh
# uso: check_multi_year BASE_MAP_JSON CAND_GLOB_PATTERN
# BASE_MAP_JSON: mapping año->factor NetBTC de la versión previa (la “mejor conocida”)
# CAND_GLOB_PATTERN: patrón que encuentra los *_kpis__WF_<año>_* o *_OOS_<año>* de la nueva versión
check_multi_year () {
  local BASE_JSON="$1"; local GLOB="$2"; local EPS=0.001  # 0.10%
  python3 - <<PY "$BASE_JSON" "$GLOB" "$EPS"
import sys, json, glob, re, pandas as pd, numpy as np

base_map = json.loads(sys.argv[1])   # {"2022":1.018661, "2023":2.641397, "2024":1.613240, "2025H1":1.138462}
pattern  = sys.argv[2]
eps      = float(sys.argv[3])

def first_num(r, keys):
    for k in keys:
        if k in r and pd.notna(r[k]):
            try: return float(str(r[k]).replace(',',''))
            except: pass
    return np.nan

def year_from_name(p):
    m = re.search(r'WF_(\d{4})|OOS_(\d{4}H1)', p)
    return (m.group(1) if m and m.group(1) else (m.group(2) if m else None))

errs=[]
for y, base in base_map.items():
    files = sorted([f for f in glob.glob(pattern) if (y in f)])
    if not files:
        print(f"[SKIP] {y}: no hay KPI candidato que matchee patrón"); continue
    f = files[-1]
    df = pd.read_csv(f, nrows=1); r = df.iloc[0].to_dict()
    cand = first_num(r, ['sats_mult','net_btc_vs_hodl','net_btc_ratio','net_btc','netBTC','net_btc_oos'])
    if not np.isfinite(cand):
        print(f"[FAIL] {y}: KPI sin sats en {f}"); errs.append(y); continue
    need = base*(1.0+eps)
    status = "PASS" if cand >= need else "FAIL"
    print(f"[{status}] {y}: cand={cand:.6f}  base={base:.6f}  req≥{need:.6f}  file={f}")
    if status=="FAIL": errs.append(y)

sys.exit(1 if errs else 0)