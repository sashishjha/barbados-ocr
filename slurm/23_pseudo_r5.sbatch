#!/bin/bash
#SBATCH --job-name=pseudo_r5
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=logs_checkpoint/pseudo_r5_%j.out
#SBATCH --error=logs_checkpoint/pseudo_r5_%j.err

mkdir -p logs_checkpoint preds_checkpoint data/derived_checkpoint report
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

# Inference and Filtering (Steps a and b) already completed before the crash.
# Skipping directly to Step (c): Retrain.

echo "════ Step (c): Retrain Qwen-7B + TrOCR-hw in parallel (R5) ════"
echo "GPU 0: Qwen-7B R5 (padded crops, pseudo loss_weight=1.0)"
ROCR_VISIBLE_DEVICES=0 python -m src.train_vlm \
  --config configs/qwen25vl_lora.yaml \
  --fold 0 \
  --set train.pseudo_csv=data/derived_checkpoint/train_and_pseudo_r5.csv \
        output.checkpoint_dir=runs_checkpoint/qwen25vl_lora_checkpoints_r5 \
        output.dir=runs_checkpoint/qwen25vl_lora_r5 &
A=$!

echo "GPU 1: TrOCR-hw R5 (padded crops, pseudo loss_weight=1.0)"
ROCR_VISIBLE_DEVICES=1 python -m src.train_trocr \
  --config configs/trocr_large.yaml \
  --fold 0 \
  --set train.pseudo_csv=data/derived_checkpoint/train_and_pseudo_r5.csv \
        output.checkpoint_dir=runs_checkpoint/trocr_hw_checkpoints_r5 \
        output.dir=runs_checkpoint/trocr_hw_r5 &
B=$!

wait $A; RA=$?
wait $B; RB=$?
echo "exit codes: qwen=$RA  trocr=$RB"

report_disk
