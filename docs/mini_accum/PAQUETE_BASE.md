# Mini-Accum KISS — Paquete Base (KISS)
**Última actualización (UTC): 2025-10-29**

Este paquete resume las reglas operativas y de estrategia que rigen el proyecto. Mantenerlo versionado evita olvidos entre sesiones.

## 1) Rutas y artefactos
- **Repo raíz:** `~/PycharmProjects/Bot_BTC`
- **Canary (DRYRUN, hourly :07 UTC):** `evidence/dayN_<YYYY-MM-DD>/canary_live.*.log`
  - **GREEN** = el log del ciclo contiene **ambas** líneas:
    - `ready (signal fresh)`
    - `canary_live: done`
- **Reporte diario:** `scripts/mini_accum/bb_dailyreport.zsh` → `evidence/dayN_<DATE>/REPORT.md`
  - Cabecera imprime `- ATTEST OK: 1` cuando aplica.
- **Health:** `health/mini_accum.status`
- **Cron logs:** `logs/cron.log`
- **Paquetes diarios:** `artifacts/canary_pack_<DATE>.tgz`
- **Tag diario (freeze):** `FREEZE_DAILY_YYYYMMDD`

## 2) ATTEST (robusto, KISS)
- Acepta en `logs/cron.log` la línea `"[OK] write_status:"` **con o sin** prefijo ISO (p.ej. `2025-...Z [OK] write_status:` o `[OK] write_status:`).
- Si no hay fecha en el log, **fallback** a:
  - `mtime` de `health/mini_accum.status` = hoy UTC **y**
  - `"health":"ok"`.
- **Gates:** `scripts/mini_accum/gates_pilot_live`
  - **GREEN** = gate **duro**
  - **ATTEST** = **warning** por defecto (`ATTEST_REQUIRE=0`), configurable a **hard** con `ATTEST_REQUIRE=1`.

## 3) Freeze nocturno (automatizado)
- **Horario:** 18:59 América/Bogotá = **23:59 UTC**.
- Al disparo, ejecutar en este orden (append-only, idempotente):
  1. `scripts/mini_accum/bb_dailyreport.zsh <DAY_UTC>`
  2. `scripts/mini_accum/pack_canary.zsh <DAY_UTC>` (o fallback `tar -czf`)
  3. `scripts/mini_accum/gates_pilot_live` (solo para dejar traza)
  4. `git tag -f FREEZE_DAILY_YYYYMMDD && git push --force origin refs/tags/FREEZE_DAILY_YYYYMMDD`
- Dejar sello en `logs/cron.log`:  
  `YYYY-MM-DDThh:mm:ssZ [OK] nightly freeze completed`

## 4) Estrategia por ciclo (KISS-estacional)
- **Año +2 post-halving** (2014, 2018, 2022, 2026, …) ⇒ **E1_Y2 (táctico 1D)**  
  - EMA 12/26, RSI(14) con bandas 35/65, ADX(14) ≥ 22, **dwell=3**, `macro_sma200=ON`.
- **Otros años** ⇒ **KISS v1 TOP (core 1D)**  
  - EMA 21/55 + macro G200 + ADX, **DD15**, **RB1**, **H30**, **BULL0**.
- Referencia expandida: `docs/mini_accum/Mini-Accum_KISS_Executive_Summary_and_Reconstruction_Blueprint_EN.md`

## 5) Reglas de operación
- **Sesgo:** long/flat (spot; sin leverage; sin shorts).
- **Entradas:** consenso de tendencia + filtro macro **ON**.
- **Salidas:** pérdida de consenso o guardarraíl de riesgo (DD15) / TTL (H30).
- **Objetivo:** maximizar **NetBTC** vs HODL con **MDD_vs_HODL ≤ 1** y **FPY** contenido.

## 6) Chequeos exprés (diarios)
- **GREEN hoy:** último `canary_live.*.log` del día con `ready (signal fresh)` + `done`.
- **ATTEST hoy:** `ATTEST_OK=1` por línea válida en `cron.log` **o** fallback por `mtime + health=ok`.
- **Freeze:** existe / se actualizó `FREEZE_DAILY_YYYYMMDD`.

> Este documento es el “piso” operativo del proyecto. Si cambia la política de presets, cron o gates, **actualizar aquí primero** y comitear.
