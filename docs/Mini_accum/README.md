# Mini Accum (KISS) — Baseline

**Fecha del PASS:** 2025-09-13  
**Ventana:** 2025-05-10 → 2025-09-07 (BTC-USD 4h, macro D1)

**Config única:** `configs/mini_accum/config.yaml`  
**Run diag:** `bash scripts/mini_accum/diag_gate_mix.sh`

## Parámetros (KISSv4)
- Entrada: `ema21 > ema55 and macro_green`
- Salida activa: `close < ema21 and macro_red` (confirm=2) + `exit_margin=0.003`
- Filtros: `ADX14 ≥ 22`, `macro_filter.use=true`, `ATR regime p65 (on)`
- Anti-whipsaw: `dwell_min_between_flips=96`
- Costes: fee/slip = 6/6 bps por lado
- Reports: `reports/mini_accum/`

## KPIs del PASS
- `netBTC=1.0130`, `mdd_vs_HODL=0.716`, `fpy=18.42` (6 flips)
- Ver detalle en `docs/runs/*_PASS.md` (autogenerable abajo).

## Próximos pasos (en foco)
1) Afinar `exit_margin` 30→35 bps si FPY ≤ 26.  
2) ADX 20 vs 22 con ATR p65 y medir netBTC/MDD/FPY.  
3) (Opcional) `dwell` 96↔72 si vemos re-entradas tardías.

## Experimento: Cross-buffer fijo vs adaptativo (ATR)

**Slice**: CORE_2025 (4H), 2025-05-10 → 2025-09-06  
**Costes**: fee 8 bps + slip 2 bps (round-trip ≈ 20 bps)

**Resumen BASE vs XB fijo**

| Exp        | netBTC | Δnet vs BASE |  mdd  |  Δmdd |  fpy  | Archivo |
|:-----------|------:|-------------:|:-----:|------:|------:|:--|
| BASE (PIN) | 0.9069 | —            | 0.810 | —     | 21.67 | reports/mini_accum/base_v0_1_20250914_0309_kpis__KISSv4_CORE_PIN.csv |
| XB=15      | 0.9930 | +0.0861      | 0.716 | −0.094| 18.57 | reports/mini_accum/base_v0_1_20250914_0532_kpis__KISSv4_CORE_XB15.csv |
| XB=20      | **1.0029** | **+0.0960** | **0.716** | **−0.094** | **18.57** | reports/mini_accum/base_v0_1_20250914_0532_kpis__KISSv4_CORE_XB20.csv |
| XB=25      | 0.9824 | +0.0755      | 0.782 | −0.028| 18.57 | reports/mini_accum/base_v0_1_20250914_0506_kpis__KISSv4_CORE_XB25.csv |
| XB=30      | 0.9493 | +0.0424      | 0.773 | −0.037| 18.57 | reports/mini_accum/base_v0_1_20250914_0532_kpis__KISSv4_CORE_XB30.csv |
| XB=35      | 0.9189 | +0.0120      | 0.919 | **+0.109** | 18.57 | reports/mini_accum/base_v0_1_20250914_0532_kpis__KISSv4_CORE_XB35.csv |

**Lecturas clave**
- XB **15–25 bps** mejora netBTC y baja **FPY** (~−3.1) sin empeorar MDD; **XB=20** es el mejor en este slice.
- XB≥30 empieza a degradar netBTC y, a 35 bps, empeora MDD.

### Estado de validación (XB=20)

**CORE_2025 (slice 4H, 2025‑05‑10→2025‑09‑06):** netBTC=**1.0029**, MDD=**0.716**, FPY=**18.57**  
Archivo: `reports/mini_accum/base_v0_1_20250914_0611_kpis__KISSv4_CORE_XB20.csv`

**OOS H2‑2024 (4H):** netBTC=**0.8865**, MDD=**0.625**, FPY=**20.07**  
Archivo: `reports/mini_accum/base_v0_1_20250914_0611_kpis__KISSv4_H2_2024_XB20.csv`

**OOS Q1‑2025 (4H):** netBTC=**1.0291**, MDD=**0.885**, FPY=**20.75**  
Archivo: `reports/mini_accum/base_v0_1_20250914_0611_kpis__KISSv4_Q1_2025_XB20.csv`

(ver comandos en este README)

### Baseline adoptado (XB=20) — OOS PASS

✅ Con los resultados anteriores, **adoptamos XB fijo = 20 bps** como baseline del preset **CORE_2025** (pasa OOS en H2‑2024 y Q1‑2025 sin empeorar MDD ni FPY).

**Cómo etiquetar el baseline (comandos):**
```sh
# Asegura el valor en el preset (si no está ya)
chflags nouchg configs/mini_accum/presets/CORE_2025.yaml 2>/dev/null || true
yq -i '.signals.cross_buffer_bps=20' configs/mini_accum/presets/CORE_2025.yaml

# Commit + tag del baseline
git add -A
git commit -m "Mini_accum: baseline CORE_2025 XB=20 (OOS PASS)"
git tag -a baseline-core2025-xb20 -m "Baseline CORE_2025 XB=20 (OOS PASS)"
```

---

### Experimento A/B siguiente — XB **adaptativo** (ATR)

**Objetivo:** comparar **XB fijo=20** vs **XB adaptativo por ATR** con bandas `quiet=20 / yellow=30 / loud=40` sin subir FPY ni empeorar MDD.

**Config (feature flag, encender solo para el experimento):**
```yaml
modules:
  xb_adaptive:
    enabled: true      # OFF por defecto en el preset; ON aquí para A/B
    quiet_bps: 20
    yellow_bps: 30
    loud_bps: 40
```

**Cómo correr el A/B (slice CORE_2025 + OOS):**
```sh
# Ventana CORE_2025 fijada
S="2025-05-10"; E="2025-09-06"
CFG="configs/mini_accum/presets/CORE_2025.yaml"

# A) BASE (XB=20 fijo)
REPORT_SUFFIX="KISSv4_CORE_XB20" mini-accum-backtest --config "$CFG" --start "$S" --end "$E"

# B) ADAPTIVO por ATR
TMP=$(mktemp) && cp "$CFG" "$TMP"
yq -i '
  .modules.xb_adaptive.enabled=true |
  .modules.xb_adaptive.quiet_bps=20 |
  .modules.xb_adaptive.yellow_bps=30 |
  .modules.xb_adaptive.loud_bps=40
' "$TMP"
REPORT_SUFFIX="KISSv4_CORE_XB_ADAPT" mini-accum-backtest --config "$TMP" --start "$S" --end "$E"

# KPIs rápidos (último archivo de cada patrón)
setopt NULL_GLOB
for pat in \
  reports/mini_accum/*_kpis__KISSv4_CORE_XB20.csv \
  reports/mini_accum/*_kpis__KISSv4_CORE_XB_ADAPT.csv
do
  f=(${~pat}(Om[1])); [[ -z $f ]] && continue
  awk -F, -v F="$f" 'NR==2{printf "%-68s  netBTC=%.4f  mdd=%.3f  fpy=%.2f\n",F,$1,$4,$7}' "$f"
done

# OOS para ambos
bash scripts/mini_accum/run_oos.sh 2024-07-01 2024-12-31 KISSv4_H2_2024_XB20
bash scripts/mini_accum/run_oos.sh 2025-01-01 2025-03-31 KISSv4_Q1_2025_XB20
bash scripts/mini_accum/run_oos.sh 2024-07-01 2024-12-31 KISSv4_H2_2024_XB_ADAPT
bash scripts/mini_accum/run_oos.sh 2025-01-01 2025-03-31 KISSv4_Q1_2025_XB_ADAPT
```

**Criterio de adopción:** mantener el de “Reglas de paro” más arriba (ΔnetBTC ≥ +0.02 en slice, OOS ≥ −0.01 vs BASE, FPY ±2, MDD ≤ BASE+0.05 abs).

### Próximo paso A — Barrido 40/50
(ver comandos en este README)

### Próximo paso B — XB adaptativo por ATR
**Config recomendada**:
```yaml
modules:
  xb_adaptive:
    enabled: true
    quiet_bps: 20
    yellow_bps: 30
    loud_bps: 40
```

## Roadmap PDCA (foco: subir NetBTC sin subir FPY ni empeorar MDD)

| Palanca | Cómo impacta | Señales de éxito | Riesgo a FPY/MDD | Dificultad | Éxito esperado |
|---|---|---|---|---|---|
| **Cross‑buffer adaptativo por ATR (xb_adaptive)** | Sincroniza entradas con la volatilidad; evita cruces “al ras” en ruido y deja respirar en alta vol. | ↑netBTC, FPY igual o ↓ levemente, MDD igual o ↓ | **Bajo** si rangos (quiet/yellow/loud) están bien calibrados | Media | **Alta (★★★★☆)** |
| **Guardarraíl de salida por ATR (exit‑ATR guardrail)** | Bloquea ventas si el rebote aún está dentro de *k·ATR* tras confirmación | netBTC igual o ↑, sin subir FPY | **Medio**: si *k* muy alto, se atrapan retrocesos | Media | Media‑baja (★★☆☆☆) |
| **Confirmación con “age valve” (TTL de confirmación)** | Evita salidas rezagadas si no hay follow‑through en ≤N velas | FPY igual o ↓, MDD igual o ↓ | **Bajo** | Baja | Media‑baja (★★☆☆☆) |
| **Pausa tras flip que también afecta SELL** | Reduce ping‑pong justo después de un giro | FPY ↓, MDD igual o ↓ | **Medio**: puede perder alguna venta buena | Baja | Media‑baja (★★☆☆☆) |
| **ADX dinámico (percentiles)** | Endurece el umbral en rangos chop y lo relaja en tendencia nítida | ↑netBTC levemente, FPY estable | **Medio** si baja demasiado el umbral | Media‑alta | Media (★★★☆☆) |
| **Macro filter con pendiente/“distancia a EMA200”** | Evita operar con macro verde “débil” | MDD ↓ leve, FPY ↓ leve | **Medio**: puede filtrar trades válidos | Media | Media (★★★☆☆) |

> Nota: En el slice **CORE_2025** fijado, **XB=20 bps** ya mostró la mejor mejora neta con FPY ↓ y MDD no peor; sirve de pivote para validar las variantes adaptativas.

## Qué haría ya (orden sugerido)

1. **Fijar “XB fijo=20 bps” como baseline del preset** `CORE_2025` y etiquetar commit.  
   - `signals.cross_buffer_bps: 20` (documentado en este README).
2. **Validación OOS rápida (sin tocar más knobs)** en **H2‑2024** y **Q1‑2025** con los OHLC “pinned”.  
   - Criterio rápido de adopción: OOS no peor que BASE por más de **−0.01** netBTC y **FPY** dentro de **±2**.
3. **Implementar “xb_adaptive”** con bandas `quiet=20 / yellow=30 / loud=40` (feature flag, apagado por defecto).  
   - Comparar **XB fijo 20** vs **adaptativo**; conservar el que gane **netBTC** sin subir **FPY** ni **MDD**.
4. **Mantener “exit‑ATR guardrail” en código pero OFF por defecto**.  
   - Ya vimos que **k=1.5/2.0** no mejora este slice; se re‑evalúa con más historia.
5. **Dejar “age valve” y “pausa SELL” documentadas** (tests hechos) y **OFF** por defecto hasta tener más datos OOS.
6. **Automatizar el reporte A/B** (tabla de BASE vs A/B) en `scripts/mini_accum/make_run_report.sh` para cada experimento.

## Reglas de paro (para no sobre‑ajustar)

- **Una palanca por commit** y siempre con etiqueta + reporte (`docs/runs/`).
- **Criterio de adopción** (cada cambio debe cumplir todos):
  - Δ**netBTC ≥ +0.02** vs BASE en el slice fijado **y** no peor que BASE por más de **−0.01** en ambos OOS.
  - **FPY** dentro de **±2** vs BASE.
  - **MDD_vs_HODL** ≤ BASE + **0.05** absoluto.
- **Si dos iteraciones seguidas no mejoran** bajo los criterios anteriores, **archivar la palanca** y volver a otra hipótesis.
- **Datos “pinned” para todos los tests** y **regresión OOS** mínima (H2‑2024 y Q1‑2025).
- **No mezclar cambios**: si un experimento gana, se vuelve el nuevo BASE; si no, **revert** inmediato.
- **Costes fijos y explícitos** en el preset durante toda la serie de pruebas.

### Experimento: Umbral ADX (18 vs 20 vs 24)

**Slice CORE_2025 (4H, pinned):**  
| ADX min | netBTC | mdd  | fpy  | Archivo (último) |
|---------|-------:|:----:|:----:|:-----------------|
| 18      | 1.0029 | 0.716| 18.57| reports/mini_accum/*_kpis__KISSv4_CORE_ADX18.csv |
| 20 (BASE)| 1.0029| 0.716| 18.57| reports/mini_accum/*_kpis__KISSv4_CORE_ADX20.csv |
| 24      | 1.0029 | 0.716| 18.57| reports/mini_accum/*_kpis__KISSv4_CORE_ADX24.csv |

**OOS H2-2024 y Q1-2025:** mismos KPIs (invariancia).  
**Decisión:** mantener **ADX=20** (robustez > complejidad).
