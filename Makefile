SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
MAKEFLAGS += --no-builtin-rules

FREEZES := reports/mini_accum/_freezes/E1_Y2_2022.freeze.txt \
           reports/mini_accum/_freezes/V1TOP_2023.freeze.txt \
           reports/mini_accum/_freezes/V1TOP_2024.freeze.txt \
           reports/mini_accum/_freezes/V1TOP_2025H1.freeze.txt

.PHONY: smoke seal retag check gate stress-costs

smoke:
	@echo "[YAML allowlist]"
	@if [ -f .yaml_validate_allowlist ]; then \
		{ while IFS= read -r f; do \
			[ -z "$$f" ] && continue; \
			python3 -c 'import sys,yaml,pathlib;p=pathlib.Path(sys.argv[1]);yaml.safe_load(p.open("r",encoding="utf-8"));print("[OK]",p)' "$$f"; \
		  done; } < .yaml_validate_allowlist; \
	else \
		echo "(skip) .yaml_validate_allowlist no existe"; \
	fi
	@$(MAKE) -s seal </dev/null
	@$(MAKE) -s retag </dev/null
	@$(MAKE) -s check </dev/null
	@echo "✅ Smoke 1→5 OK"

seal:
	@for f in $(FREEZES); do \
	  [ -f "$$f" ] || { echo "[MISS] $$f"; continue; }; \
	  yq -i \
	    '.version = (.version // 1) | .costs.fee_bps_per_side = 2.0 | .costs.slip_bps_per_side = 1.0 | \
	     .kpis.sats_mult = (.kpis.sats_mult // .sats_mult // "NA") | \
	     .kpis.mdd_vs_hodl = (.kpis.mdd_vs_hodl // .mdd_vs_hodl // "NA") | \
	     .kpis.flips = (.kpis.flips // .flips // 0) | \
	     .data_hashes.data_1d_sha256 = (.data_hashes.data_1d_sha256 // .data_1d_sha256 // "NA") | \
	     .data_hashes.data_4h_sha256 = (.data_hashes.data_4h_sha256 // .data_4h_sha256 // "NA") | \
	     del(.sats_mult, .mdd_vs_hodl, .flips, .data_1d_sha256, .data_4h_sha256)' "$$f"; \
	  printf '[SEALED] %s -> NetBTC=%s  MDD=%s  1D=%s 4H=%s  costs=%s/%s bps\n' "$$f" \
	    "$$(yq -r '.kpis.sats_mult' "$$f")" \
	    "$$(yq -r '.kpis.mdd_vs_hodl' "$$f")" \
	    "$$(yq -r '.data_hashes.data_1d_sha256' "$$f")" \
	    "$$(yq -r '.data_hashes.data_4h_sha256' "$$f")" \
	    "$$(yq -r '.costs.fee_bps_per_side' "$$f")" \
	    "$$(yq -r '.costs.slip_bps_per_side' "$$f")"; \
	done

retag:
	@scripts/mini_accum/dev/retag_safe.sh

check:
	@scripts/mini_accum/dev/check_freeze_tags.sh

gate:
	@mkdir -p reports/mini_accum
	@{ \
	  echo "== $$(date -u +'%Y-%m-%dT%H:%M:%SZ') thresholds: E1_S_MIN=$${E1_S_MIN:-2.9} E1_M_MAX=$${E1_M_MAX:-0.12} V1_S_MIN=$${V1_S_MIN:-1.0} V1_M_MAX=$${V1_M_MAX:-1.0} =="; \
	  scripts/mini_accum/dev/gate.sh; \
	} | tee -a reports/mini_accum/GATE_STATUS.txt
stress-costs:
	@mkdir -p reports/mini_accum
	@bash scripts/mini_accum/dev/stress_costs.sh 2>&1 | tee reports/mini_accum/STRESS_COSTS.txt
