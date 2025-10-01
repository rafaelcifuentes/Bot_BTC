#!/usr/bin/env bash
ENFORCE_FRESH=1
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
. .venv/bin/activate
/bin/bash scripts/mini_accum/kiss_v1_wf_pipeline.sh || true
python3 scripts/mini_accum/signal_emitter.py
RUN_MODE=paper python3 scripts/mini_accum/live_wrapper.py
echo "---- A/B ----"; sed -n '1,160p' reports/mini_accum/ab_latest.md || true
echo "---- LIVE KPIs ----"; tail -n 5 reports/mini_accum/live_kpis.csv || true
echo "---- FLIPS ----"; tail -n 5 reports/mini_accum/flips_log.csv || true
echo "---- HEALTH ----"; cat health/mini_accum.status || true
