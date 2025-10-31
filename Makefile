SHELL := /bin/zsh
.ONESHELL:

# Exporta ROOT/MANIFEST al entorno de la receta (shell ve $$ROOT)
export ROOT := $(HOME)/PycharmProjects/Bot_BTC
export MANIFEST := $(ROOT)/reports/mini_accum/kiss_v1/_snapshots/PROD_TOP/manifest.json

.PHONY: contract-check kiss-guard daily-guards

contract-check:
	# Carga contrato y corre guardián del Santo Grial
	source "$$ROOT/env/kiss_contract.env" || true
	"$$ROOT/scripts/mini_accum/contract_check.zsh"

kiss-guard:
	# KPI guard con KPI pinneado (OOS_2025H1_KPIS) desde el env
	source "$$ROOT/env/kiss_contract.env" || true
	. "$$ROOT/.venv/bin/activate"
	OOS_KPI_GLOB="$$OOS_2025H1_KPIS" \
	python "$$ROOT/scripts/mini_accum/kpi_kiss_guard.py" \
		--min-sats 1.00 --max-fpy 26 \
		--manifest "$$MANIFEST"

daily-guards:
	# Ejecuta ambos guardianes y loguea (sin paréntesis: usamos { ...; }):
	mkdir -p "$$ROOT/logs"
	{
		date -u +"==== %FT%TZ ====";
		source "$$ROOT/env/kiss_contract.env" || true;
		"$$ROOT/scripts/mini_accum/contract_check.zsh";
		. "$$ROOT/.venv/bin/activate";
		OOS_KPI_GLOB="$$OOS_2025H1_KPIS" \
		python "$$ROOT/scripts/mini_accum/kpi_kiss_guard.py" \
			--min-sats 1.00 --max-fpy 26 \
			--manifest "$$MANIFEST";
	} 2>&1 | tee -a "$$ROOT/logs/contract.log"
