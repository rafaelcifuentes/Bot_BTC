SHELL := /bin/zsh
.SHELLFLAGS := -eu -o pipefail -c

ROOT := $(HOME)/PycharmProjects/Bot_BTC
MANIFEST := $(ROOT)/reports/mini_accum/kiss_v1/_snapshots/PROD_TOP/manifest.json

.PHONY: engine-switch-check engine-core-2025H1 contract-check kiss-guard daily-guards

engine-switch-check:
	@ROOT="$(ROOT)" "$(ROOT)/scripts/mini_accum/engine_switch_check.zsh"

engine-core-2025H1:
	@ROOT="$(ROOT)" "$(ROOT)/scripts/mini_accum/engine_core_2025H1_from_source.zsh"

contract-check:
	@source "$(ROOT)/env/mini_accum/kiss_contract.env" || true; \
	"$(ROOT)/scripts/mini_accum/contract_check.zsh"

kiss-guard:
	@source "$(ROOT)/env/mini_accum/kiss_contract.env" || true; \
	. "$(ROOT)/.venv/bin/activate"; \
	python "$(ROOT)/scripts/mini_accum/kpi_kiss_guard.py" \
		--min-sats 1.00 --max-fpy 26 \
		--manifest "$(MANIFEST)"

daily-guards:
	@mkdir -p "$(ROOT)/logs"; \
	ROOT="$(ROOT)" MANIFEST="$(MANIFEST)" \
	  "$(ROOT)/scripts/mini_accum/daily_guards.zsh"
