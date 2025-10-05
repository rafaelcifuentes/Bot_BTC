# Mini-Accum — Canario V1.1 (SL/TP ATR) · Guardrails 30d

> **Importante:** Este README **no ejecuta código**. El guardado y la evaluación de guardrails viven en el script
> `scripts/mini_accum/canary_guardrails.sh`. Aquí solo se documenta **cómo** usarlo.

---

## Qué hace el canario

- Estrategia candidata: ATR(14) con **SL=2×**, **TP=3×** (`fix_on_entry=true`).
- **Guardrails (ventana móvil 30d)**:
  - **ΔMDD ≤ 0**
  - **ΔFPY ≤ +2/año**
  - **ΔROI_anual ≥ −4%**
- Si faltan datos para MDD/ROI en la ventana (pocos puntos), el script reporta **WARN** y **no marca violación**.

---

## Script

- Archivo: `scripts/mini_accum/canary_guardrails.sh`
- Descubre automáticamente los artefactos más recientes por convención de nombres:
  - Base (CORE): `*_equity__CORE_2025.csv`, `*_flips__CORE_2025.csv`
  - Candidato (ATR 2×3): `*_equity__CORE_2025_ATR14x2_0.csv`, `*_flips__CORE_2025_ATR14x2_0.csv`
- Puedes **forzar** rutas específicas exportando variables antes de llamar al script:
  - `BASE_EQ`, `CAND_EQ`, `BASE_FLIPS`, `CAND_FLIPS`

### Umbrales (por defecto)

- Dentro del script:
  - `MDD_MAX_DELTA=0`
  - `FPY_MAX_DELTA=2`
  - `ROI_MIN_DELTA_ANNUAL=-0.04`
- Para un override rápido en shell:
  ```bash
  MDD_MAX_DELTA=0 FPY_MAX_DELTA=2 ROI_MIN_DELTA_ANNUAL=-0.04 bash scripts/mini_accum/canary_guardrails.sh
  ```

---

## Ejecución manual (con venv)

```bash
# Ubícate en la raíz del repo
cd /path/to/Bot_BTC

# Asegura permisos
chmod +x scripts/mini_accum/canary_guardrails.sh

# Ejecuta con el venv en PATH y log persistente
PATH="$PWD/.venv/bin:$PATH" \
  bash scripts/mini_accum/canary_guardrails.sh | tee -a reports/mini_accum/guardrails.log
```

### Salida esperada (ejemplo)

```text
[DEBUG] BASE_EQ=reports/mini_accum/base_v0_1_20251004_1435_equity__CORE_2025.csv
[DEBUG] CAND_EQ=reports/mini_accum/base_v0_1_20251004_0713_equity__CORE_2025_ATR14x2_0.csv
[DEBUG] BASE_FLIPS=reports/mini_accum/base_v0_1_20251004_1435_flips__CORE_2025.csv
[DEBUG] CAND_FLIPS=reports/mini_accum/base_v0_1_20251004_0713_flips__CORE_2025_ATR14x2_0.csv
[DEBUG] Window=últimos 30d desde min(ts_max)
[GUARDRAILS] ΔMDD=N/A | ΔFPY=-12.17/a | ΔROI_anual=N/A
[WARN] ΔMDD no disponible (datos insuficientes)
[WARN] ΔROI_anual no disponible (datos insuficientes)
[PASS] Guardrails dentro de umbrales
```

---

## Programación (cron)

Edítalo con `crontab -e` y pega una línea similar (ajusta la ruta del repo):

```cron
10 2 * * * cd /Users/rafaelcifuentes/PycharmProjects/Bot_BTC && PATH="/Users/rafaelcifuentes/PycharmProjects/Bot_BTC/.venv/bin:$PATH" /bin/bash scripts/mini_accum/canary_guardrails.sh >> reports/mini_accum/guardrails.log 2>&1
```

Verifica que quedó instalado:

```bash
crontab -l | grep canary_guardrails.sh
```

---

## Artefactos usados

- **Equity/Flips (CORE):** `reports/mini_accum/*_equity__CORE_2025.csv`, `reports/mini_accum/*_flips__CORE_2025.csv`
- **Equity/Flips (ATR 2×3):** `reports/mini_accum/*_equity__CORE_2025_ATR14x2_0.csv`, `reports/mini_accum/*_flips__CORE_2025_ATR14x2_0.csv`
- Si alguno falta, define las variables:
  ```bash
  BASE_EQ=reports/mini_accum/...csv \
  CAND_EQ=reports/mini_accum/...csv \
  BASE_FLIPS=reports/mini_accum/...csv \
  CAND_FLIPS=reports/mini_accum/...csv \
  PATH="$PWD/.venv/bin:$PATH" bash scripts/mini_accum/canary_guardrails.sh
  ```

---

## Estado SPA/RC (2025-10-05)

- Ago-2023: `p_consistent=0.545` → **FAIL**
- Q3-2024: `p_consistent=0.545` → **FAIL** *(curvas idénticas → neutral)*
- Q2-2025: `p_consistent=0.545` → **FAIL** *(curvas idénticas → neutral)*

**Decisión:** mantener **canario** (10–20%) con guardrails activos; **promoción bloqueada** hasta PASS ≥ 0.60.  
**Siguiente paso:** medir **ATR 2.5×** y/o **`reentry_ttl=8–12`** en Ago-2023 y Q3-2024 y repetir SPA/RC.

---

## Troubleshooting

- `N/A` en ΔMDD/ΔROI: ventana común de 30d con pocos puntos → **esperado** en periodos sin operaciones; no es violación.
- `ΔFPY` grande: revisa que el *flips canditado* cuente `SELL_SLTP` y el baseline **no**.
- Forzar ventanas/artefactos: usa las variables `BASE_EQ`/`CAND_EQ`/`BASE_FLIPS`/`CAND_FLIPS`.
