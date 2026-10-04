#!/bin/bash
#SBATCH --job-name=vlm_r1b_res
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=logs_checkpoint/vlm_r1b_res_%j.out
#SBATCH --error=logs_checkpoint/vlm_r1b_res_%j.err

mkdir -p logs_checkpoint
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

# RESUME Qwen-7B training from step 750 (epoch 4).
# A ROCm GPU segfault killed the previous job midway.

ROCR_VISIBLE_DEVICES=0 python -m src.train_vlm \
  --config configs/qwen25vl_lora.yaml \
  --fold 0 \
  --resume \
  --set output.checkpoint_dir=runs_checkpoint/qwen25vl_lora_checkpoints_r1b \
        output.dir=runs_checkpoint/qwen25vl_lora_r1b

echo "Qwen R1b resumed done"
report_disk
