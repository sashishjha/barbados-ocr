#!/bin/bash
#SBATCH --job-name=bbd_r1
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=logs_checkpoint/retrain_r1_%j.out
#SBATCH --error=logs_checkpoint/retrain_r1_%j.err

mkdir -p logs_checkpoint
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

# IMPORTANT: we start from scratch on the new config models (stage1 and 7B).
# Because we are changing base models, we CANNOT use --resume, we MUST train from scratch.
# We also wipe the old checkpoint dirs just to be safe.
rm -rf runs_checkpoint/trocr_large_checkpoints_r1
rm -rf runs_checkpoint/qwen25vl_lora_checkpoints_r1

# Train TrOCR
ROCR_VISIBLE_DEVICES=0 python -m src.train_trocr \
  --config configs/trocr_large.yaml \
  --fold 0 \
  --set output.checkpoint_dir=runs_checkpoint/trocr_large_checkpoints_r1 \
        output.dir=runs_checkpoint/trocr_large_r1 &
A=$!

# Train Qwen
ROCR_VISIBLE_DEVICES=1 python -m src.train_vlm \
  --config configs/qwen25vl_lora.yaml \
  --fold 0 \
  --set output.checkpoint_dir=runs_checkpoint/qwen25vl_lora_checkpoints_r1 \
        output.dir=runs_checkpoint/qwen25vl_lora_r1 &
B=$!

wait $A
echo "TrOCR R1 done"
wait $B
echo "Qwen R1 done"

report_disk
