contract-check:
	@source env/kiss_contract.env || true; \
	"$(HOME)/PycharmProjects/Bot_BTC/scripts/mini_accum/contract_check.zsh"

kiss-guard:
	@. .venv/bin/activate && \
	python scripts/mini_accum/kpi_kiss_guard.py \
	  --min-sats 1.00 --max-fpy 26 \
	  --manifest "$(HOME)/PycharmProjects/Bot_BTC/reports/mini_accum/kiss_v1/_snapshots/PROD_TOP/manifest.json)"
