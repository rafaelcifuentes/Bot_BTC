.DEFAULT_GOAL := deploy-help
SHELL := /bin/bash

.PHONY: deploy-help
deploy-help:
	@printf '%s\n' 'Uso:' \
	'  make -f mk/deploy.mk -s deploy-plan' \
	'  make -f mk/deploy.mk -s deploy-select' \
	'  make -f mk/deploy.mk -s deploy-status' ''
	@printf '%s\n' 'Conceptos:' \
	'  DRIVE  = núcleo estable (2023: PROD_KISSv1_2023)' \
	'  SPORT  = turbo condicionado (2024: PROD_KISSv1_2024; se activa si fricción OK)' \
	'  OFF    = aparcados (2025H1, E1_2022) hasta mejorar' ''
	@printf '%s\n' 'Fricción viva:' \
	'  deploy/live_fee_slip  => texto tipo "2/1", "2/2", "2/3"...' \
	'  SPORT se activa si live_fee_slip ∈ {2/1, 2/2}; si no, DRIVE.' ''
	@printf '%s\n' 'Ficheros:' \
	'  deploy/PLAN.txt, deploy/plan.vars, deploy/ACTIVE.mode, deploy/ACTIVE.tag, deploy/live_fee_slip'

.PHONY: deploy-plan
deploy-plan:
	@mkdir -p deploy
	@printf '%s\n' '# Deployment plan KISS' '' \
	'CORE (DRIVE):        PROD_KISSv1_2023' \
	'CONDITIONAL (SPORT): PROD_KISSv1_2024  - activo si live_fee_slip in {2/1, 2/2}' \
	'OFF:                 PROD_KISSv1_2025H1, PROD_E1_Y2_2022' > deploy/PLAN.txt
	@{ \
	  printf 'CORE_TAG=%s\n' 'PROD_KISSv1_2023'; \
	  printf 'CONDITIONAL_TAG=%s\n' 'PROD_KISSv1_2024'; \
	  printf "CONDITIONAL_FEE_SLIP_OK='%s'\n" '2/1 2/2'; \
	  printf "OFF_TAGS='%s'\n" 'PROD_KISSv1_2025H1 PROD_E1_Y2_2022'; \
	} > deploy/plan.vars
	@[ -f deploy/live_fee_slip ] || printf '%s\n' '2/2' > deploy/live_fee_slip
	@echo "[OK] deploy/PLAN.txt + deploy/plan.vars (y live_fee_slip default si faltaba)"

.PHONY: deploy-select
deploy-select:
	@bash -eu -c '\
	  d=deploy; mkdir -p "$$d"; \
	  core=PROD_KISSv1_2023; sport=PROD_KISSv1_2024; ok_set="2/1 2/2"; \
	  [ -f $$d/plan.vars ] && . $$d/plan.vars; \
	  [ -f $$d/live_fee_slip ] || echo "2/2" > $$d/live_fee_slip; \
	  live=$$(tr -d "\n" < $$d/live_fee_slip); \
	  mode=DRIVE; tag=$$core; \
	  for x in $$ok_set $${CONDITIONAL_FEE_SLIP_OK:-}; do \
	    if [ "$$live" = "$$x" ]; then mode=SPORT; tag=$$sport; fi; \
	  done; \
	  echo $$mode > $$d/ACTIVE.mode; echo $$tag > $$d/ACTIVE.tag; \
	  echo "[OK] mode=$$mode tag=$$tag live_fee_slip=$$live"; \
	'

.PHONY: deploy-status
deploy-status:
	@bash -eu -c '\
	  d=deploy; \
	  core=PROD_KISSv1_2023; CONDITIONAL_TAG=PROD_KISSv1_2024; CONDITIONAL_FEE_SLIP_OK="2/1 2/2"; \
	  [ -f $$d/plan.vars ] && . $$d/plan.vars; \
	  live=$$( [ -f $$d/live_fee_slip ] && tr -d "\n" < $$d/live_fee_slip || echo "n/a" ); \
	  mode=$$( [ -f $$d/ACTIVE.mode ] && tr -d "\n" < $$d/ACTIVE.mode || echo "n/a" ); \
	  tag=$$(  [ -f $$d/ACTIVE.tag  ] && tr -d "\n" < $$d/ACTIVE.tag  || echo "n/a" ); \
	  echo "Plan: CORE=$$core  SPORT=$$CONDITIONAL_TAG  OK_SET={$${CONDITIONAL_FEE_SLIP_OK}}"; \
	  echo "Live: fee_slip=$$live"; \
	  echo "Activo: MODE=$$mode  TAG=$$tag"; \
	'
