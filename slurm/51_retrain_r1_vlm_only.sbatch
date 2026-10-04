#!/bin/bash
#SBATCH --job-name=vlm_r1b
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=logs_checkpoint/vlm_r1b_%j.out
#SBATCH --error=logs_checkpoint/vlm_r1b_%j.err

mkdir -p logs_checkpoint
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

# Retrain Qwen-7B ONLY with the critical padding_side="left" fix.
# TrOCR-stage1 was dropped — it has no handwriting pre-training and hallucinated badly.
# Using GPU:0 only (1 GPU, more memory headroom for 7B model).
rm -rf runs_checkpoint/qwen25vl_lora_checkpoints_r1b

ROCR_VISIBLE_DEVICES=0 python -m src.train_vlm \
  --config configs/qwen25vl_lora.yaml \
  --fold 0 \
  --set output.checkpoint_dir=runs_checkpoint/qwen25vl_lora_checkpoints_r1b \
        output.dir=runs_checkpoint/qwen25vl_lora_r1b

echo "Qwen R1b done"
report_disk
