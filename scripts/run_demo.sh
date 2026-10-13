#!/usr/bin/env bash
# Synthetic end-to-end demo: renders procedural footage with exact ground truth, runs the full
# pipeline and writes evaluation.json. No real footage is involved.
# Usage: ./scripts/run_demo.sh [walk|static] [output folder]
set -euo pipefail
KIND="${1:-walk}"
OUT="${2:-$HOME/metrica-demo-$KIND}"
cd "$(dirname "$0")/.."
[ -d .venv ] && . .venv/bin/activate
metrica models download
metrica demo --kind "$KIND" --out "$OUT"
echo "Open the case in the app:  metrica serve --workspace \"$OUT\""
