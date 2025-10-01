[PEGA AQUÍ TODO EL MANUAL DE 1 HOJA]
# Manual operativo semanal — mini_accum KISS v1

**Botón semanal:** `/bin/bash scripts/mini_accum/run_once_paper.sh`  
**Pre-run:** VPN activo → fetch D1/4h OK.  
**Aceptación checkpoint:** Datos frescos (D1=día actual UTC, 4h ≤4h), `Health=OK`, A/B sin promoción (Δ < +0.02 o empeora MDD/FPY), flips razonables.

---

## Registro de checkpoints

### Semana 2025-09-15 → 2025-09-21
- [x] **Miércoles 2025-09-17 23:11Z (~19:11 ET)** — pos=**1.0**, flips=**3**, **Health=OK**, **A/B Δ=+0.019** → baseline se mantiene.
- [x] **Jueves 2025-09-18 12:18Z (08:18 ET)** — D1=**2025-09-18**, 4h=**12:00Z**, pos=**1.0**, flips=**3**, **Health=OK**, **A/B Δ=+0.019** → baseline se mantiene.

> Nota: A/B “Δ median(sats_mult)” < **+0.02** y sin mejora en MDD/FPY ⇒ **seguir con baseline**.
> 
> # Guardar checkpoint y tag
TS="$(date -u +'%F_%H%MUTC')"
SNAP="docs/Mini_accum/checkpoints/${TS}"
mkdir -p "$SNAP" || true
cp reports/mini_accum/ab_latest.md              "$SNAP/ab_latest.md" || true
cp reports/mini_accum/live_kpis.csv             "$SNAP/live_kpis_${TS}.csv" || true
cp reports/mini_accum/flips_log.csv             "$SNAP/flips_log_${TS}.csv" || true
cp health/mini_accum.status                     "$SNAP/health_${TS}.status" || true
echo "[OK] Snapshot guardado en $SNAP"
git add "$SNAP" && git commit -m "paper checkpoint — ${TS}" && git push
git tag -a "PAPER_${TS}" -m "Paper checkpoint — ${TS}" && git push origin "PAPER_${TS}"