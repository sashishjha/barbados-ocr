#!/bin/bash
#SBATCH --job-name=bbd_setup
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G

#SBATCH --time=02:00:00
#SBATCH --output=logs_checkpoint/setup_%j.out
#SBATCH --error=logs_checkpoint/setup_%j.err
#
# STEP 0. Build the ROCm environment and prove every candidate architecture
# runs a forward AND backward pass BEFORE any real training job is queued.
# You only get 2 concurrent jobs - never skip this.

mkdir -p logs_checkpoint
cd "${SLURM_SUBMIT_DIR}"

# ROCm path (system-wide, no module load needed)
export PATH="/opt/rocm/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

echo "=== Node: $(hostname) | GPUs: ==="
rocm-smi --showproductname 2>/dev/null | head -8 || true

VENV="${SLURM_SUBMIT_DIR}/.venv"
if [ ! -d "$VENV" ]; then
  echo "=== Creating venv ==="
  
  # Crucial: /tmp is 100% full on the cluster. Pip will crash extracting PyTorch.
  # Route all tmp allocations to the generous /mnt/data space.
  export TMPDIR="${SLURM_SUBMIT_DIR}/.pip_tmp_checkpoint"
  mkdir -p "$TMPDIR"
  
  python3 -m venv "$VENV"
else
  echo "=== Venv already exists, activating ==="
fi

source "$VENV/bin/activate"
pip install uv
uv pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision
uv pip install -r requirements.txt

# Now source common.sh (venv is active, ROCm path set)
source slurm/common.sh

echo "=== torch / ROCm sanity ==="
python3 -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count())"

echo "=== full smoke test (forward+backward per arm) ==="
python3 -m src.smoke_test --models trocr vlm ctc --images data/images --n 50

echo "=== eval harness unit tests ==="
python3 -m pytest tests/ -q || echo "pytest failed - review above"

report_disk
rm -rf "${TMPDIR:-}" 2>/dev/null || true
