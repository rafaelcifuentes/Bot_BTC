SHELL := /bin/zsh

ROOT := $(HOME)/PycharmProjects/Bot_BTC
MANIFEST := $(ROOT)/reports/mini_accum/kiss_v1/_snapshots/PROD_TOP/manifest.json

.PHONY: contract-check kiss-guard daily-guards

contract-check:
	# Carga contrato y corre guardián del Santo Grial
	source "$(ROOT)/env/kiss_contract.env" || true; \
	"$(ROOT)/scripts/mini_accum/contract_check.zsh"

kiss-guard:
	# KPI guard con KPI pinneado (OOS_2025H1_KPIS) si existe
	source "$(ROOT)/env/kiss_contract.env" || true; \
	. "$(ROOT)/.venv/bin/activate"; \
	if [[ -n "$$OOS_2025H1_KPIS" && -s "$$OOS_2025H1_KPIS" ]]; then \
	  python "$(ROOT)/scripts/mini_accum/kpi_kiss_guard.py" \
	    --min-sats 1.00 --max-fpy 26 \
	    --manifest "$(MANIFEST)" \
	    --oos-kpi "$$OOS_2025H1_KPIS"; \
	else \
	  python "$(ROOT)/scripts/mini_accum/kpi_kiss_guard.py" \
	    --min-sats 1.00 --max-fpy 26 \
	    --manifest "$(MANIFEST)"; \
	fi

daily-guards:
	# Ejecuta ambos guardianes y registra en logs/contract.log
	ROOT="$(ROOT)" MANIFEST="$(MANIFEST)" \
	"$(ROOT)/scripts/mini_accum/daily_guards.zsh"
