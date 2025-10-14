set -euo pipefail
YEARS=("$@")
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
echo "[INFO] Repo: $REPO_ROOT"
echo "[INFO] Años a buscar: ${YEARS[*]:-<ninguno>}"
echo
echo "[INFO] .gitignore contiene 'tmp_wf'?"
grep -n "tmp_wf" .gitignore || echo "(no hay entrada de tmp_wf en .gitignore)"
echo
echo "========== (A) Barrido en filesystem =========="
for Y in "${YEARS[@]:-}"; do
  echo "[SCAN] *WF_${Y}*.csv"
  find . -type f -iname "*WF_${Y}*.csv" | sed "s#^#  - #g" || true
done
echo
echo "========== (B) Git history por ruta exacta =========="
for Y in "${YEARS[@]:-}"; do
  for P in \
    "tmp_wf/BTC-USD_4h_WF_${Y}.csv" \
    "data/tmp_wf/BTC-USD_4h_WF_${Y}.csv" \
    "tmp_wf/BTC-USD_1d_WF_${Y}.csv" \
    "data/tmp_wf/BTC-USD_1d_WF_${Y}.csv"
  do
    echo "[GIT] $P"
    git log --stat -- "$P" || echo "  (sin historial)"
    echo
  done
done
echo "========== (D) Commits que mencionan tmp_wf =========="
git log --all --name-status | grep -E "tmp_wf/" -n || echo "(ningún commit menciona tmp_wf/)"
echo
echo "Fin. Si no aparecen CSV, hay que re-generar."
