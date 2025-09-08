#Asegura permisos:
chmod +x scripts/wk_runner.sh
# Correr todo (semana actual):

WTAG=w6 ./scripts/wk_runner.sh
# O solo ./scripts/wk_runner.sh

# Revisar compactado:
sed -n '1,120p' reports/w*/summary/decision.md
head -n 5 reports/w*/summary/compact_base_vs_stress.csv