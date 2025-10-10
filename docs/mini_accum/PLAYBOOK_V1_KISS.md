

# Playbook — Reproducción & Rollback (V1_KISS)

Este playbook permite **rehidratar resultados**, **verificar integridad** y **hacer rollback** al freeze estable de KISS v1. Úsalo cuando quieras dejar el alias `kiss_v1` exactamente como el snapshot dorado, validar hashes o volver al tag de seguridad.

---

## 0) Variables (ajusta si cambias de snapshot)
```bash
export REPO="$HOME/PycharmProjects/Bot_BTC"
export ALIAS="$REPO/reports/mini_accum/kiss_v1"
export SNAP="$REPO/reports/mini_accum/kiss_v1/_snapshots/20251010_171051__NETBTC_4p340727__DD15_RB1_H30_G200_BULL0"
export TAG="KISSv1_BASE_20251010_freeze_NETBTC_4p340727"
```

> Tips de shell (zsh):
> - Si pegas líneas con `!` (p.ej. excepciones de `.gitignore`), desactiva historia: `set +H` (solo esa sesión).
> - Para que los `cp "$SNAP"/WF_* ...` no fallen si no hay match: `setopt nonomatch`.

---

## 1) Rehidratar alias legacy (`kiss_v1`)
Restaura los CSVs del snapshot para tener el alias **contractual** e inmutable sobre el que calculamos `NetBTC`.

```bash
# Evita error por globs en zsh si no hay match
setopt nonomatch 2>/dev/null || true

# Copia KPIs y equities del snapshot al alias estable
cp "$SNAP"/WF_*_kpis__v1_2.csv   "$ALIAS"/
cp "$SNAP"/WF_*_equity__v1_2.csv "$ALIAS"/

# Comprueba el resumen y el NetBTC (producto por ventanas)
python scripts/mini_accum/show_netbtc_summary.py
```

Esperado (ejemplo):
```
WF_2022 1.018661 ...
WF_2023 2.641397 ...
WF_2024 1.613240 ...
NetBTC = 4.340726883639296
```

---

## 2) Verificar integridad (hashes del `manifest.json`)
Confirma que los archivos restaurados coinciden **bit a bit** con el freeze.

```bash
python - <<'PY'
import os, json, hashlib, sys
REPO = os.environ["REPO"]
SNAP = os.environ["SNAP"]
m = json.load(open(os.path.join(SNAP, "manifest.json")))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()
ok=True
for w in m["windows"]:
    for k in ("kpis","equity"):
        p = os.path.join(REPO, w[k])
        got = sha(p); exp = w[f"sha256_{k}"]
        print(("OK  " if got==exp else "FAIL"), w["window"], k, got, ("=" if got==exp else "!="), exp)
        ok &= (got==exp)
print("RESULT:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
```

Salida esperada: `RESULT: OK`.

---

## 3) Rollback completo al **tag dorado** (opcional)
Si además quieres volver el repo a la **instantánea exacta** del freeze:

```bash
git fetch --all --tags
git checkout "$TAG"   # cambia HEAD al tag del freeze

# Para regresar a tu rama anterior:
# git checkout -
```

---

## 4) Reproducibilidad (WF + freeze 1‑click)
Ejecuta el pipeline con freeze automático (usa la versión inferida o fija una explícita):

```bash
# Opción A: versión implícita con fecha
bash scripts/mini_accum/kiss_v1_wf_pipeline.sh

# Opción B: fijar la etiqueta del freeze
FREEZE_VERSION="KISSv1_BASE_20251010_freeze" DO_FREEZE=1 \
bash scripts/mini_accum/kiss_v1_wf_pipeline.sh
```

El pipeline debe:
- generar/validar KPIs por ventana,
- añadir stress de costes (Spearman ±bps),
- correr CSCV/PBO,
- crear snapshot (`_snapshots/.../`),
- escribir `manifest.json`,
- y etiquetar: `KISSv1_BASE_20251010_freeze_NETBTC_4p340727`.

---

## 5) Checklist de FREEZE (6 líneas)
1. **QA datos OK** (ohlc/orders sin duplicados; schemas válidos).
2. **WF completo** con columnas clave: `sats_mult, mdd_vs_hodl, fpy, fail_rate`.
3. **Stress de costes OK** (mediana por Δbps y Spearman estable en −20…+20 bps).
4. **CSCV/PBO OK** (p̂ razonable documentado).
5. **Snapshot + manifest + tag** creados y **commiteados**.
6. **One‑pager actualizado** en `docs/mini_accum/freezes/...` y linkado en Roadmap.

> Nota: Este checklist ya está automatizado en `kiss_v1_wf_pipeline.sh`. Úsalo como verificación manual rápida.

---

## 6) Dónde mirar los artefactos
- **Snapshot** (CSV + manifest):  
  `reports/mini_accum/kiss_v1/_snapshots/20251010_171051__NETBTC_4p340727__DD15_RB1_H30_G200_BULL0/`
- **Tag**: `KISSv1_BASE_20251010_freeze_NETBTC_4p340727`
- **One‑pager del freeze**:  
  `docs/mini_accum/freezes/KISSv1_BASE_20251010_freeze.md`
- **Roadmap PDCA** (resumen continuo):  
  `reports/mini_accum/walkforward/Roadmap_PDCA.md`

---

## 7) Troubleshooting rápido
- `heredoc>` en terminal: cierra el bloque pegando **solo** la línea final `MD`.
- Error zsh con `!`: ejecuta `set +H` antes de pegar líneas con exclamaciones.
- `KeyError: 'REPO'` al verificar hashes: exporta variables del **§0**.
- `tag not found`: `git fetch --all --tags`.
- `cp "$SNAP"/WF_* ...` falla por globs: `setopt nonomatch`.

---

## 8) ¿Por qué este playbook?
- **Reproducibilidad**: devuelve el alias `kiss_v1` a un estado **medible y auditable**.
- **Trazabilidad**: `manifest.json` con SHA‑256 por ventana evita drift.
- **Seguridad operativa**: rollback atómico con **tag dorado** y snapshot versionado.