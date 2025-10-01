#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(git rev-parse --show-toplevel)"
HOOKS_DIR="${ROOT_DIR}/.git/hooks"
mkdir -p "${HOOKS_DIR}"
cat > "${HOOKS_DIR}/pre-commit" <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail
python3 scripts/git/precommit_check_docs.py
HOOK
chmod +x "${HOOKS_DIR}/pre-commit"
echo "[install] pre-commit instalado en ${HOOKS_DIR}/pre-commit"
