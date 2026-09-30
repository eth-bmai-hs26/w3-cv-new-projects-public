#!/usr/bin/env bash
# Start the tile inspection platform (macOS / Linux):
#   ./run.sh                                        opens http://127.0.0.1:8000 in your browser
#   ./run.sh --model ~/Downloads/platform_model.pt  ... with the model you downloaded from Colab
#   ./run.sh --host 0.0.0.0                         ... reachable from a phone on the same Wi-Fi
# First run: creates .venv and installs the Python packages (a few minutes, needs internet).
# The web interface is shipped pre-built in frontend/dist, so Node.js is not needed.
set -euo pipefail
cd "$(dirname "$0")"

# 1. A Python 3.10+ interpreter: $PYTHON, else python3, else python.
find_python() {
  for cand in "${PYTHON:-}" python3 python; do
    [ -n "$cand" ] || continue
    if command -v "$cand" >/dev/null 2>&1 &&
       "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      echo "$cand"; return 0
    fi
  done
  return 1
}

# 2. The virtual environment. The marker file records which requirements.txt was
#    installed, so an interrupted or outdated install is repeated automatically.
MARKER=.venv/.installed
if [ ! -x .venv/bin/python ] || ! cmp -s requirements.txt "$MARKER" 2>/dev/null; then
  if ! PY="$(find_python)"; then
    echo "Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/"
    echo "(macOS: or 'brew install python'; Ubuntu/Debian: 'sudo apt install python3 python3-venv')."
    exit 1
  fi
  echo "Setting up .venv with $("$PY" --version) - one time, a few minutes..."
  [ -x .venv/bin/python ] || "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  # Linux without an NVIDIA GPU: the CPU build of PyTorch (~200 MB instead of ~2.5 GB).
  if [ "$(uname -s)" = "Linux" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
    .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
  .venv/bin/python -m pip install -r requirements.txt
  cp requirements.txt "$MARKER"
fi

# 3. The web interface (only rebuilt if frontend/dist was deleted).
if [ ! -f frontend/dist/index.html ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "Building the frontend..."
    (cd frontend && npm install --no-audit --no-fund && npm run build)
  else
    echo "frontend/dist is missing and Node.js is not installed: restore it with 'git checkout frontend/dist'."
    echo "The API still starts; open http://127.0.0.1:8000/docs"
  fi
fi

# 4. Start. Serving (no sub-command) also opens the browser.
if [ $# -eq 0 ] || [[ "$1" == -* ]]; then
  set -- "$@" --open
fi
exec .venv/bin/python -m backend "$@"
