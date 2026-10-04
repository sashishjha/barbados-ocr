#!/bin/bash
#SBATCH --job-name=infer_qwen
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=logs_checkpoint/infer_qwen_%j.out
#SBATCH --error=logs_checkpoint/infer_qwen_%j.err

mkdir -p logs_checkpoint
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

TAG="20260921_1900_qwen_r1b"

echo "### test inference: Qwen-7B (r1b) ONLY"
ROCR_VISIBLE_DEVICES=0 python -m src.infer \
  --config configs/qwen25vl_lora.yaml \
  --checkpoint runs_checkpoint/qwen25vl_lora_checkpoints_r1b/best_checkpoint \
  --manifest data/Test.csv \
  --out preds_checkpoint/vlm_r1b_test_nbest.csv \
  --top1_out preds_checkpoint/vlm_r1b_test.csv \
  --arm vlm --nbest 1

echo "### train-lexicon post-processing"
python -m src.postprocess \
  --pred_csv preds_checkpoint/vlm_r1b_test.csv \
  --train_csv data/derived_checkpoint/train_and_pseudo_r1.csv \
  --out_csv preds_checkpoint/vlm_r1b_test_pp.csv \
  --audit_csv report/postprocess_audit_${TAG}.csv

echo "### Create final submission CSV"
python -m src.make_submission \
  --test_csv data/Test.csv --pred_csv preds_checkpoint/vlm_r1b_test_pp.csv \
  --out_csv submissions/sub_qwen_${TAG}.csv \
  --fallbacks preds_checkpoint/vlm_r1b_test.csv \
  --train_csv data/derived_checkpoint/train_and_pseudo_r1.csv \
  --method "qwen7b_r1b+pp" --notes "Qwen 7B (r1b padding fixed) + postproc, NO TROCR"

report_disk
