#!/usr/bin/env bash
# Sourced by every sbatch script. Keeps module loads, env vars, and the storage
# cleanup policy in ONE place.
set -euo pipefail

# ---- Module system (must be sourced — not in PATH by default) ---------------
if [ -f /usr/share/modules/init/bash ]; then
  . /usr/share/modules/init/bash
fi

# ---- ROCm environment -------------------------------------------------------
module load rocm/6.2.4 2>/dev/null || true
# ROCm is installed system-wide at /opt/rocm — no 'module load' needed.
export PATH="/opt/rocm/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# Keep the HF cache inside the project so it counts against (and can be pruned
# from) our own 200 GB quota rather than a hidden home directory.
export HF_HOME="${SLURM_SUBMIT_DIR:-$PWD}/.hf_cache_checkpoint"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

# ROCm perf knobs that are safe defaults on MI300X.
export HSA_FORCE_FINE_GRAIN_PCIE=1
export MIOPEN_FIND_MODE=3          # fast kernel search, avoids long warmups
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# ---- venv ------------------------------------------------------------------
VENV="${SLURM_SUBMIT_DIR:-$PWD}/.venv"
if [ -d "$VENV" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "WARNING: no venv at $VENV - run slurm/00_setup_smoke_test.sbatch first"
fi

echo "=================================================================="
echo " job      : ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-none})"
echo " node     : $(hostname)"
echo " started  : $(date -Is)"
python3 -c "import torch; print(' torch:', torch.__version__, '| hip:', getattr(torch.version,'hip','n/a'), '| gpus:', torch.cuda.device_count())" 2>/dev/null || true
rocm-smi --showproductname 2>/dev/null | head -10 || true
echo "=================================================================="

# ---- storage discipline ----------------------------------------------------
report_disk () {
  echo "--- disk usage (200 GB quota) ---"
  du -sh "${SLURM_SUBMIT_DIR:-$PWD}"/{data,runs_checkpoint,.hf_cache_checkpoint} 2>/dev/null || true
  df -h "${SLURM_SUBMIT_DIR:-$PWD}" | tail -1
}

# Deletes everything EXCEPT dirs whose names contain "checkpoint" (cluster rule)
# and prunes all but best+last inside each checkpoint root.
cleanup_storage () {
  echo "--- cleanup ---"
  local root="${SLURM_SUBMIT_DIR:-$PWD}"
  find "$root/runs_checkpoint" -maxdepth 3 -type d -name "checkpoint_step*" 2>/dev/null \
    | sort | head -n -1 | while read -r d; do
        parent=$(dirname "$d")
        # keep the newest per parent; best_checkpoint is never matched here
        newest=$(find "$parent" -maxdepth 1 -type d -name "checkpoint_step*" | sort | tail -1)
        [ "$d" != "$newest" ] && rm -rf "$d" && echo "  pruned $d"
      done || true
  find "$root/runs_checkpoint" -name "*.tmp" -o -name "core.*" 2>/dev/null | xargs -r rm -f
  rm -rf "$root/runs_checkpoint/_smoke" 2>/dev/null || true
  report_disk
}

trap 'echo "--- job exiting at $(date -Is) ---"; cleanup_storage' EXIT
