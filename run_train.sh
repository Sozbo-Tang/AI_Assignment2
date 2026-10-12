#!/usr/bin/env bash
# One-click training for R-EDL on MNIST (official ICLR2024-REDL code, minimally adapted).
#
# Usage:   bash run_train.sh
# Effect:  creates a local venv, installs dependencies, downloads the datasets
#          (auto-fetched into code/data on first run), then trains the R-EDL
#          model on MNIST and evaluates OOD detection against KMNIST and FMNIST
#          (5 seeds x 60 epochs, as in the paper's config).
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

# 1/4 Virtual environment (first run only)
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating virtual environment in ./$VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "[1/4] Virtual environment already exists, skipping creation."
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 2/4 Dependencies (skipped when already importable)
if python -c "import torch, torchvision, sklearn, pandas" 2>/dev/null; then
    echo "[2/4] Dependencies already installed, skipping pip install."
else
    echo "[2/4] Installing dependencies (on Linux the torch wheel bundles CUDA 12.x)..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

python - <<'PYEOF'
import torch
info = f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
if torch.cuda.is_available():
    info += f" | GPU: {torch.cuda.get_device_name(0)}"
else:
    info += " | will run on CPU (much slower)"
print(f"[2/4] {info}")
PYEOF

# 3/4 Datasets: MNIST (ID) plus KMNIST and FashionMNIST (the OOD test sets)
# are fetched into code/data here if missing.
python - <<'PYEOF'
from torchvision.datasets import MNIST, KMNIST, FashionMNIST
for cls in (MNIST, KMNIST, FashionMNIST):
    cls(root="code/data", download=True)
print("[3/4] Datasets ready: MNIST (ID) + KMNIST, FashionMNIST (OOD).")
PYEOF

# 4/4 Train R-EDL on MNIST, then evaluate OOD detection
echo "[4/4] Training R-EDL on MNIST (5 seeds x 60 epochs; roughly 10-20 min per seed on an RTX 4060)..."
cd code
python main.py --configid "1_mnist/mnist-redl-train" --suffix redl

echo ""
echo "Done. Per-seed metrics (accuracy, OOD AUPR/AUROC per uncertainty measure):"
echo "  code/saved_models/mnist-redl-train/mnist-redl-train.csv"
echo "Model checkpoints (best/last per seed):"
echo "  code/saved_models/mnist-redl-train/<seed>_best , <seed>_last"
