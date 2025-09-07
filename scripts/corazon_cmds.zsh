# corazon_cmds.zsh — versión mínima para probar "source"
emulate -L zsh
setopt no_aliases

runC_ping() { echo "[Corazon] functions loaded OK"; }
runC_status() {
  echo "[Corazon] status OK — EXCHANGE=${EXCHANGE:-binanceus}, OUT=${OUT:-reports}"
}
