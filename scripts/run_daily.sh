#!/bin/bash
# Daily Window run (used by launchd, see scripts/install_launchd.sh).
# Starts Ollama if it is not running, then runs the whole pipeline.
# Extra arguments go to window.run, e.g. `scripts/run_daily.sh --dry-run`.
set -u
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

echo "=== $(date '+%F %T %Z') ==="
if ! curl -s -m 3 localhost:11434/api/tags >/dev/null; then
  echo "ollama not running, starting it"
  nohup ollama serve >/dev/null 2>&1 &
  for _ in $(seq 1 30); do
    curl -s -m 2 localhost:11434/api/tags >/dev/null && break
    sleep 1
  done
fi

.venv/bin/python -m window.run "$@"
code=$?
echo "exit code $code"
exit $code
