#!/bin/bash
#SBATCH --job-name=qwen_inf
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

#SBATCH --time=1:00:00
#SBATCH --output=logs_checkpoint/qwen_inf_%j.out
#SBATCH --error=logs_checkpoint/qwen_inf_%j.err

mkdir -p logs_checkpoint preds_checkpoint
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

echo "=== Running Qwen Inference on Test set ==="
python -m src.infer \
  --config configs/qwen25vl_lora.yaml \
  --checkpoint runs_checkpoint/qwen25vl_lora_checkpoints/best_checkpoint \
  --manifest data/Test.csv \
  --out preds_checkpoint/vlm_test_nbest.csv \
  --top1_out preds_checkpoint/vlm_test.csv \
  --nbest 5

echo "=== Generating Submission File ==="
# Since we only re-ran Qwen, we'll just format its top-1 output for submission.
python -c "
import pandas as pd
df = pd.read_csv('preds_checkpoint/vlm_test.csv')
df = df[['ID', 'Target']]
out = 'submissions/sub_vlm_fixed_pad.csv'
df.to_csv(out, index=False)
print('Wrote', out)
"
