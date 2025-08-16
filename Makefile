SHELL := /bin/bash

# =========================
# Python por defecto: usa venv si existe
# =========================
PYRUN ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || command -v python)

# =========================
# Corte histórico (para freeze)
# =========================
FREEZE ?= 2025-08-10

# =========================
# Parámetros overridables
# =========================
THRESHOLD ?= 0.61
ADX1D_MIN ?= 30
ADX4_MIN  ?= 18

# -------------------------
# Flags base (SIN adx1d_min/--adx4_min)
# -------------------------
COMMON_FLAGS_BASE := --use_sentiment \
  --fg_csv ./data/sentiment/fear_greed.csv \
  --funding_csv ./data/sentiment/funding_rates.csv \
  --fg_long_min -0.18 --fg_short_max 0.18 --funding_bias 0.01 \
  --pt_mode pct --sl_pct 0.02 --tp_pct 0.10 --trail_pct 0.015 \
  --adx_daily_source resample --adx1d_len 14

GATE_FLAGS := --adx1d_min $(ADX1D_MIN) --adx4_min $(ADX4_MIN)

# Si COMMON_FLAGS viene del entorno (export), lo ignoramos para recomponer
ifeq ($(origin COMMON_FLAGS), environment)
  override COMMON_FLAGS :=
endif

# Composición por defecto
COMMON_FLAGS ?= --threshold $(THRESHOLD) $(COMMON_FLAGS_BASE) $(GATE_FLAGS)

# Sanitiza (por si alguien mete "..." literal)
override COMMON_FLAGS := $(filter-out ...,$(COMMON_FLAGS))

# =========================
# Helpers
# =========================
print-%: ; @echo '$*=$($*)'
print-orig-%: ; @echo '$* origin=$(origin $*) value=$($*)'

# =========================
# Runners / scripts
# =========================
RUNNER_V2    := runner_profileA_V2.py
RUNNER_PERLA := runner_profileA_PerlaNegra.py
RUNNER_SENT  := runner_profileA_RF_sentiment_EXP.py
FLAGS_PRIMARY := --primary_only

REGISTRY := registry_runs.csv

.PHONY: \
  env semaforo open-semaforo \
  register-sentexp registry-dedupe register-v2 register-perla register-all \
  run-v2-freeze run-v2-live run-perla-freeze run-perla-live run-sentexp-freeze run-sentexp-live run-all \
  weekly-full weekly weekly-clean snapshot-semaforo weekly-61 weekly-59 \
  sweep-sentexp sweep-sentexp-freeze summarize-sweep summarize-sweep-index summarize-sweep-pattern \
  open-sweep open-sweep-index open-sweep-pattern \
  print-% print-orig-%

# =========================
# Utilidades
# =========================
env:
	@echo "PYRUN = $(PYRUN)"
	@echo "Python: $$($(PYRUN) -c 'import sys; print(sys.executable)')"

# =========================
# Semáforo y registro
# =========================
semaforo:
	@$(PYRUN) scripts/semaforo.py
	@echo "✅ Semáforo regenerado → reports/semaforo.csv"

open-semaforo:
	@open reports/semaforo.csv || true

register-sentexp:
	@mkdir -p reports
	@touch $(REGISTRY)
	@$(PYRUN) scripts/register_sentexp.py
	@tail -n 6 $(REGISTRY) || true

registry-dedupe:
	@$(PYRUN) scripts/registry_dedupe.py

register-v2:
	@mkdir -p reports
	@touch $(REGISTRY)
	@$(PYRUN) scripts/register_generic.py --strategy V2 --pattern 'reports/summary_v2_*.json'
	@tail -n 6 $(REGISTRY) || true

register-perla:
	@mkdir -p reports
	@touch $(REGISTRY)
	@$(PYRUN) scripts/register_generic.py --strategy PerlaNegra --pattern 'reports/summary_perla_negra_*.json'
	@tail -n 6 $(REGISTRY) || true

register-all: register-v2 register-perla register-sentexp

# =========================
# Runners
# =========================
run-v2-freeze:
	@echo "▶️  V2 (freeze hasta $(FREEZE))"
	@$(PYRUN) $(RUNNER_V2) $(FLAGS_PRIMARY) --freeze_end $(FREEZE) --repro_lock $(COMMON_FLAGS)

run-v2-live:
	@echo "▶️  V2 (live)"
	@$(PYRUN) $(RUNNER_V2) $(FLAGS_PRIMARY) $(COMMON_FLAGS)

run-perla-freeze:
	@echo "▶️  PerlaNegra (freeze hasta $(FREEZE))"
	@$(PYRUN) $(RUNNER_PERLA) $(FLAGS_PRIMARY) --freeze_end $(FREEZE) --repro_lock $(COMMON_FLAGS)

run-perla-live:
	@echo "▶️  PerlaNegra (live)"
	@$(PYRUN) $(RUNNER_PERLA) $(FLAGS_PRIMARY) $(COMMON_FLAGS)

run-sentexp-freeze:
	@echo "▶️  SentimentEXP (freeze hasta $(FREEZE))"
	@$(PYRUN) $(RUNNER_SENT) $(FLAGS_PRIMARY) --freeze_end $(FREEZE) --repro_lock $(COMMON_FLAGS)

run-sentexp-live:
	@echo "▶️  SentimentEXP (live)"
	@$(PYRUN) $(RUNNER_SENT) $(FLAGS_PRIMARY) $(COMMON_FLAGS)

run-all: run-v2-freeze run-v2-live run-perla-freeze run-perla-live run-sentexp-freeze run-sentexp-live
	@echo "✅ Runners completados (V2, PerlaNegra, SentimentEXP; freeze + live)"

# =========================
# Pipelines
# =========================
weekly-full: run-all register-all registry-dedupe semaforo open-semaforo
	@echo "✅ Weekly full listo"

weekly: register-sentexp semaforo
	@true

weekly-clean: register-sentexp registry-dedupe semaforo open-semaforo
	@true

snapshot-semaforo:
	@mkdir -p reports/snapshots
	@cp reports/semaforo.csv "reports/snapshots/semaforo_$$(date -u +%Y%m%dT%H%M%SZ).csv"
	@echo "🗂️ Snapshot guardado en reports/snapshots/"

# =========================
# Presets convenientes (incluyen gates)
# =========================
weekly-61:
	@$(MAKE) weekly-full COMMON_FLAGS="--threshold 0.61 $(COMMON_FLAGS_BASE) $(GATE_FLAGS)" PYRUN="$(PYRUN)"

weekly-59:
	@$(MAKE) weekly-full COMMON_FLAGS="--threshold 0.59 $(COMMON_FLAGS_BASE) $(GATE_FLAGS)" PYRUN="$(PYRUN)"

# =========================
# Sweep (barridos) — genera índice con combinaciones y usa el ÚLTIMO summary disponible
# =========================
SWEEP_THRESHOLDS := 0.57 0.59 0.61 0.63
SWEEP_A1 := 25 28
SWEEP_A4 := 16 17
SWEEP_INDEX := reports/sweep_index.csv

sweep-sentexp:
	@echo ">>> Generando sweep INDEX (live)…"
	@rm -f $(SWEEP_INDEX)
	@touch $(SWEEP_INDEX)
	@for T in $(SWEEP_THRESHOLDS); do \
	  for A1 in $(SWEEP_A1); do \
	    for A4 in $(SWEEP_A4); do \
	      echo ">>> th=$$T a1=$$A1 a4=$$A4 (live)"; \
	      LAST=$$(ls -t reports/summary_rf_sentiment_EXP_*.json 2>/dev/null | head -1); \
	      if [ -z "$$LAST" ]; then \
	        $(PYRUN) $(RUNNER_SENT) $(FLAGS_PRIMARY) $(COMMON_FLAGS) >/dev/null 2>&1 || true; \
	        LAST=$$(ls -t reports/summary_rf_sentiment_EXP_*.json 2>/dev/null | head -1); \
	      fi; \
	      echo "$$T,$$A1,$$A4,$$LAST" >> $(SWEEP_INDEX); \
	    done; \
	  done; \
	done
	@echo "✅ sweep-sentexp (live) completo → $(SWEEP_INDEX)"

sweep-sentexp-freeze:
	@echo ">>> Generando sweep INDEX (freeze hasta $(FREEZE))…"
	@rm -f $(SWEEP_INDEX)
	@touch $(SWEEP_INDEX)
	@# hacemos una corrida freeze rápida para asegurar un summary reciente
	@$(PYRUN) $(RUNNER_SENT) $(FLAGS_PRIMARY) --freeze_end $(FREEZE) --repro_lock $(COMMON_FLAGS) >/dev/null 2>&1 || true
	@for T in $(SWEEP_THRESHOLDS); do \
	  for A1 in $(SWEEP_A1); do \
	    for A4 in $(SWEEP_A4); do \
	      echo ">>> th=$$T a1=$$A1 a4=$$A4 (freeze hasta $(FREEZE))"; \
	      LAST=$$(ls -t reports/summary_rf_sentiment_EXP_*.json 2>/dev/null | head -1); \
	      echo "$$T,$$A1,$$A4,$$LAST" >> $(SWEEP_INDEX); \
	    done; \
	  done; \
	done
	@echo "✅ sweep-sentexp-freeze completo → $(SWEEP_INDEX)"

# =========================
# Resúmenes de sweep (INDEX & PATTERN)
# =========================
SWEEP_INDEX_FILE := reports/sweep_index.csv

SWEEP_INDEX_DIR  := reports/sweep_index_out
SWEEP_INDEX_MD   := $(SWEEP_INDEX_DIR)/sweep_summary.md

SWEEP_PATTERN_GLOB := 'reports/summary_rf_sentiment_EXP_*.json'
SWEEP_PATTERN_DIR  := reports/sweep_pattern_out
SWEEP_PATTERN_MD   := $(SWEEP_PATTERN_DIR)/sweep_summary.md

.PHONY: summarize-sweep-index open-sweep-index summarize-sweep-pattern open-sweep-pattern

## Resumen SOLO del barrido actual (usa el índice)
summarize-sweep-index:
	@echo "Resumiendo sweep (INDEX) → $(SWEEP_INDEX_DIR)"
	@mkdir -p "$(SWEEP_INDEX_DIR)"
	@$(PYRUN) scripts/summarize_sweep.py --index "$(SWEEP_INDEX_FILE)" --outdir "$(SWEEP_INDEX_DIR)"
	@echo "✅ MD: $(SWEEP_INDEX_MD)"

## Abre el MD del resumen por índice
open-sweep-index: summarize-sweep-index
	@open "$(SWEEP_INDEX_MD)" 2>/dev/null || cat "$(SWEEP_INDEX_MD)"

## Resumen GLOBAL con TODO el histórico que matchee el patrón
summarize-sweep-pattern:
	@echo "Resumiendo sweep (PATTERN) → $(SWEEP_PATTERN_DIR)"
	@mkdir -p "$(SWEEP_PATTERN_DIR)"
	@$(PYRUN) scripts/summarize_sweep.py --pattern $(SWEEP_PATTERN_GLOB) --outdir "$(SWEEP_PATTERN_DIR)"
	@echo "✅ MD: $(SWEEP_PATTERN_MD)"

## Abre el MD del resumen por patrón
open-sweep-pattern: summarize-sweep-pattern
	@open "$(SWEEP_PATTERN_MD)" 2>/dev/null || cat "$(SWEEP_PATTERN_MD)"

# Back-compat (alias)
summarize-sweep: summarize-sweep-index
open-sweep: open-sweep-index