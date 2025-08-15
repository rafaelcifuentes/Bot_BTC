SHELL := /bin/bash
.ONESHELL:

REGISTRY := registry_runs.csv
PYRUN := $(if $(BOTBTC_PY),$(BOTBTC_PY),python)

.PHONY: register-sentexp semaforo default
default: semaforo

register-sentexp:
	@mkdir -p reports
	@touch $(REGISTRY)
	@$(PYRUN) scripts/register_sentexp.py
	@tail -n 6 $(REGISTRY) || true

semaforo:
	@$(PYRUN) scripts/semaforo.py
	@echo "👉 Abre reports/semaforo.csv"

sentexp: register-sentexp semaforo

open-semaforo:
	@open reports/semaforo.csv

open-semaforo:
	@open reports/semaforo.csv

open-semaforo:
	@open reports/semaforo.csv

open-semaforo:
	@open reports/semaforo.csv

open-semaforo:
	@open reports/semaforo.csv
