#!/usr/bin/env bash
# METRICA installer for macOS and Linux: creates .venv in the repository and installs METRICA.
# Usage: ./scripts/install.sh [--dev] [--models]
set -euo pipefail
cd "$(dirname "$0")/.."

PY=""
for c in python3.12 python3.13 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    case "$v" in 3.11|3.12|3.13) PY="$c"; break ;; esac
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.11, 3.12 or 3.13 is required (python.org or your package manager)." >&2
  exit 1
fi
echo "Using $PY ($($PY --version))"

"$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel
EXTRA=""
for a in "$@"; do [ "$a" = "--dev" ] && EXTRA="[dev]"; done
python -m pip install -e ".${EXTRA}"

if [ ! -f frontend/dist/index.html ]; then
  if command -v npm >/dev/null 2>&1; then
    (cd frontend && npm ci && npm run build)
  else
    echo "WARNING: frontend/dist is missing and npm is not installed: the web interface will not be available." >&2
  fi
fi

for a in "$@"; do
  if [ "$a" = "--models" ]; then metrica models download; fi
done

metrica doctor || true
echo
echo "Installed. Next steps:"
echo "  source .venv/bin/activate"
echo "  metrica models download     # once, ~280 MB"
echo "  metrica serve"
