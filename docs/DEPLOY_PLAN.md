# Deploy Plan — Mini-Accum KISS
- Selector fricción viva: 2/1–2/2 => SPORT_2024; ≥2/3 => DRIVE_2023
- Pasos: preflight → deploy-select → deploy-status → verificación health
- Rollback: si health>0 o alerta dura => set DRIVE_2023 y tag estable
- Registros: deploy/ACTIVE.mode, deploy/ACTIVE.tag, logs/deploy.log
