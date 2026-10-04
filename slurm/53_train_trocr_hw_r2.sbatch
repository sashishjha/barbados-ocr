#!/bin/bash
#SBATCH --job-name=trocr_hw_r2
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs_checkpoint/trocr_hw_r2_%j.out
#SBATCH --error=logs_checkpoint/trocr_hw_r2_%j.err

mkdir -p logs_checkpoint
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

# Train TrOCR-large-handwritten (microsoft/trocr-large-handwritten, MIT licensed)
# This is the handwriting-pretrained model (IAM weights), correct arm for this task.
# Replaces trocr-large-stage1 (which is synthetic-text-only and hallucinated badly).
# Output dir must contain "checkpoint" to survive rsync --delete.

ROCR_VISIBLE_DEVICES=0 python -m src.train_trocr \
  --config configs/trocr_large.yaml \
  --fold 0 \
  --set model.id=microsoft/trocr-large-handwritten \
        output.checkpoint_dir=runs_checkpoint/trocr_hw_checkpoints_r2 \
        output.dir=runs_checkpoint/trocr_hw_r2

echo "TrOCR-handwritten R2 training done"
report_disk
