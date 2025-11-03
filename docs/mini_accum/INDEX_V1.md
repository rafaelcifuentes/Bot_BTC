# Mini-Accum — Índice V1 (docs + código + gobernanza)

> Esta página referencia **todo lo previo a V2** (KISS v1: CORE_2025, E1_Y2, semáforo/selector, gates y wrappers).
> Si algún item aparece con ⚠️, crea el archivo o ajusta la ruta cuando lo tengas.

## Documentación canónica y operativa

| Nombre | Ruta | ¿Para qué sirve? | Estado |
|---|---|---|---|
| SANTO_GRIAL.md | docs/mini_accum/SANTO_GRIAL.md | One-pager canónico del TOP (contrato v1). | ✅ |
| 🧪 Gate & Contrato | docs/mini_accum/gate_contrato_v2.md | Reglas de gate/contrato (PF, MDD, FPY, PBO/CSCV, DSR, no-regresión). | ✅ |
| README_MARCHAS.md | docs/mini_accum/README_MARCHAS.md | Semáforo y marchas M1/M2/M3; freeze M3 H1-2025; cómo promover M3. | ✅ |
| PRESETS.md | docs/mini_accum/PRESETS.md | Índice de presets y overlays; runners y ejemplos. | ✅ |
| RECETA_REPRO.md | docs/mini_accum/RECETA_REPRO.md | Receta KISS de reproducción punta a punta. | ✅ |
| RECOVERY_TOP.md | docs/mini_accum/RECOVERY_TOP.md | Recuperación express del “Santo Grial”. | ✅ |
| RESULTADOS_MARCHAS.md | docs/mini_accum/RESULTADOS_MARCHAS.md | Tabla canónica de resultados M1/M2/M3 (2022–2025). | ✅ |
| Onepagers/OOS_2025H1_KISSv1.md | docs/mini_accum/onepagers/OOS_2025H1_KISSv1.md | One-pager OOS 2025H1 (CORE). | ✅ |
| BULL_HOLD.md | docs/mini_accum/BULL_HOLD.md | Notas/criterios overlay bull_hold (M3). | ✅ |
| cooldown_after_loss.md | docs/mini_accum/cooldown_after_loss.md | Módulo opt-in cooldown. | ✅ |
| hibernation_on_chop.md | docs/mini_accum/hibernation_on_chop.md | Módulo opt-in hibernación lateral. | ✅ |
| SL_tp_defensivo.md | docs/mini_accum/SL_tp_defensivo.md | Variante defensiva SL/TP v1.1. | ✅ |
| roadmap.md | docs/mini_accum/roadmap.md | Roadmap vivo (PDCA). | ✅ |
| Roadmap_PDCA.md | docs/mini_accum/Roadmap_PDCA.md | **Alias** al roadmap vivo. | ✅ |
| CONTRACT.lock.json | docs/mini_accum/CONTRACT.lock.json | Lock versionado de contrato TOP. | ✅ |
| ARTIFACTS.md | docs/mini_accum/ARTIFACTS.md | Índice de artefactos + sha256 (opcional). | ✅ |

## Presets y overlays V1

| Nombre | Ruta | Estado |
|---|---|---|
| CORE_2025 (TOP v1) | configs/mini_accum/presets/CORE_2025.yaml | ✅ |
| E1_Y2 (Año+2) | configs/mini_accum/presets/E1_Y2.yaml | ✅ |
| bull_hold (levered) | configs/mini_accum/overlays/bull_hold_levered.yaml | ✅ |
| bull_hold (puro) | configs/mini_accum/overlays/bull_hold_pure.yaml | ✅ |

## Motor (Python) y semáforo/selector

| Componente | Ruta esperada | Propósito | Estado |
|---|---|---|---|
| Motor en vivo | scripts/mini_accum/live_wrapper.py | Ejecuta la lógica de trading (lee env, ccxt, señales). | ✅ |
| Selector por ciclo | scripts/mini_accum/dev/run_regime_year.sh | Elegir preset por año (E1 en +2; CORE el resto). | ⚠️ |
| Semáforo (regime) | scripts/mini_accum/get_regime.py | EMA200/ADX14 para determinar macro verde/rojo. | ⚠️ |

## “Plomería” (wrappers ZSH) y gobernanza

| Script | Ruta | ¿Para qué sirve? | Estado |
|---|---|---|---|
| bb_day.zsh | scripts/mini_accum/bb_day.zsh | Runner principal (cron). | ✅ |
| bb_dailyreport.zsh | scripts/mini_accum/bb_dailyreport.zsh | Reporte diario (KPIs/estado). | ✅ |
| pack_canary.zsh | scripts/mini_accum/pack_canary.zsh | Empaqueta evidencia diaria. | ✅ |
| write_status.zsh | scripts/mini_accum/write_status.zsh | Heartbeat/health. | ✅ |
| run_with_overlay.zsh | scripts/mini_accum/run_with_overlay.zsh | Fusiona overlay + preset y corre ventana. | ✅ |
| ab_m3_check.zsh | scripts/mini_accum/ab_m3_check.zsh | **Gate** A/B para M3 (net ≥ 1.05; bull_pct ≥ 0.90). | ✅ |
| build_contract_lock.zsh | scripts/mini_accum/build_contract_lock.zsh | Genera CONTRACT.lock.json. | ✅ |
| yaml_sanity.zsh | scripts/mini_accum/yaml_sanity.zsh | Saneamiento YAML (tabs, CRLF, BOM...). | ✅ |

## Orquestación (Cron)

- **bb_day.zsh** cada hora .  
- Registros: , , .  
- Política: M3 solo si **PASS** A/B; por defecto **M2 (CORE_2025)**; shock → **M1 (E1_Y2)**.

