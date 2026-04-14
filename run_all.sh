#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-click runner for SafeScale-MATD3 experiments
# Usage:  bash run_all.sh [--fast] [--nn]
#   --fast  : fewer seeds/episodes for a quick smoke-test
#   --nn    : use PyTorch+CUDA neural-network TD3 policies (requires torch)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

FAST_MODE=0
NN_MODE=0
for arg in "$@"; do
  [[ "$arg" == "--fast" ]] && FAST_MODE=1
  [[ "$arg" == "--nn"   ]] && NN_MODE=1
done

# ── Python / venv setup ───────────────────────────────────────────────────────
if [[ -f "requirements.txt" ]]; then
  if [[ ! -d ".venv" ]]; then
    echo "[setup] Creating virtual environment..."
    python3 -m venv .venv
  fi
  source .venv/bin/activate
  echo "[setup] Installing/verifying dependencies..."
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements.txt
  PYTHON_BIN="python"
else
  echo "[setup] No requirements.txt found – using system python3."
  PYTHON_BIN="python3"
fi

echo ""
echo "Python: $($PYTHON_BIN --version)"
echo "Working dir: ${ROOT_DIR}"
echo ""

# ── Build flag strings ────────────────────────────────────────────────────────
FLAGS=""
[[ $FAST_MODE -eq 1 ]] && FLAGS="$FLAGS --fast"
[[ $NN_MODE   -eq 1 ]] && FLAGS="$FLAGS --nn"

if [[ $FAST_MODE -eq 1 ]]; then
  echo "[mode] FAST – n_episodes=100, n_seeds=2, eval_episodes=20, episode_slots=60"
fi
if [[ $NN_MODE -eq 1 ]]; then
  echo "[mode] NN   – PyTorch+CUDA neural-network TD3 policies"
fi

# ── Run experiments ───────────────────────────────────────────────────────────
$PYTHON_BIN run_all.py $FLAGS

echo ""
echo "Done."
echo "Figures : ${ROOT_DIR}/outputs/figures"
echo "Summary : ${ROOT_DIR}/outputs/results/summary.json"
