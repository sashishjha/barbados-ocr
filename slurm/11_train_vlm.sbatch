#!/bin/bash
#SBATCH --job-name=bbd_vlm
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G

#SBATCH --time=24:00:00
#SBATCH --output=logs_checkpoint/vlm_%j.out
#SBATCH --error=logs_checkpoint/vlm_%j.err
#
# ARM B. bf16 LoRA - NO bitsandbytes, NO quantization config (ROCm).
#   FOLD=0 CONFIG=configs/qwen25vl_lora.yaml sbatch slurm/11_train_vlm.sbatch

mkdir -p logs_checkpoint preds_checkpoint
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

FOLD=${FOLD:-0}
CONFIG=${CONFIG:-configs/qwen25vl_lora.yaml}
# Find the GPU with the lowest VRAM usage and use it explicitly
# This works around Slurm's failure to isolate GPU memory from other users' zombie processes
BEST_GPU=$(rocm-smi --showmeminfo vram | grep "VRAM Total Used Memory" | sort -nk6 | head -n 1 | awk -F'[][]' '{print $2}')
echo "Selected Best GPU: $BEST_GPU"
export ROCR_VISIBLE_DEVICES=$BEST_GPU

python -m src.train_vlm --config "${CONFIG}" --fold "${FOLD}" --resume

echo "### OOF inference (n-best) for ensembling and pseudo-label filtering"
python -m src.infer \
    --config "${CONFIG}" \
    --checkpoint runs_checkpoint/qwen25vl_lora_checkpoints/best_checkpoint \
    --manifest data/derived_checkpoint/train_folds.csv \
    --out preds_checkpoint/vlm_fold${FOLD}_oof_nbest.csv \
    --top1_out preds_checkpoint/vlm_fold${FOLD}_oof.csv \
    --nbest 5

report_disk
