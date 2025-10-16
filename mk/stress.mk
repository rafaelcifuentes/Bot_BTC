.PHONY: stress-help stress-head stress-head-save stress-prob stress-prob-save

stress-help:
	printf "%s\n" \
	"" \
	"# Headroom (ajusta umbrales si quieres)" \
	"E1_S_MIN=3.0 V1_M_MAX=0.9 make -s stress-head" \
	"" \
	"# Guardado" \
	"make -s stress-head-save && tail -n 5 reports/mini_accum/STRESS_HEAD.txt" \
	"" \
	"# (si te interesa) Agregados de probabilidad" \
	"make -s stress-prob" \
	"make -s stress-prob-save && tail -n 5 reports/mini_accum/STRESS_PROB.txt"

stress-head:
E1_S_MIN=${E1_S_MIN:-2.9} E1_M_MAX=${E1_M_MAX:-0.12} V1_S_MIN=${V1_S_MIN:-1.0} V1_M_MAX=${V1_M_MAX:-1.0} \
	  scripts/mini_accum/dev/stress_head.sh

stress-head-save:
	mkdir -p reports/mini_accum
	{ E1_S_MIN=${E1_S_MIN:-2.9} E1_M_MAX=${E1_M_MAX:-0.12} V1_S_MIN=${V1_S_MIN:-1.0} V1_M_MAX=${V1_M_MAX:-1.0} \
	  scripts/mini_accum/dev/stress_head.sh; } | tee -a reports/mini_accum/STRESS_HEAD.txt >/dev/null
	tail -n 5 reports/mini_accum/STRESS_HEAD.txt || true

stress-prob:
	$(MAKE) -s stress-head | \
	awk '/^PROD_/ {t=$$1; tot[t]++; if ($$NF=="OK") ok[t]++} \
	     /^PROD_/ && $$0 ~ /  2\/(1|2)  / {gate[t]++; if ($$NF=="OK") okg[t]++} \
	     END {for (t in tot) printf "%-16s all=%2d/%2d (%.1f%%)  gate=%2d/%2d (%.1f%%)\n", \
t, ok[t]+0, tot[t], (tot[t]?100.0*ok[t]/tot[t]:0), \
okg[t]+0, gate[t]+0, (gate[t]?100.0*okg[t]/gate[t]:0)}' | sort

stress-prob-save:
	mkdir -p reports/mini_accum
	$(MAKE) -s stress-prob | tee -a reports/mini_accum/STRESS_PROB.txt >/dev/null
	tail -n 5 reports/mini_accum/STRESS_PROB.txt || true
