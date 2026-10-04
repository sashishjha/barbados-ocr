#!/bin/bash
#SBATCH --job-name=bbd_ens_r4
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs_checkpoint/submit_r4_%j.out
#SBATCH --error=logs_checkpoint/submit_r4_%j.err

mkdir -p logs_checkpoint preds_checkpoint submissions
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

TAG="20260924_rover_r4"
TRAIN_CSV="data/derived_checkpoint/train_and_pseudo_r4.csv"

# Golden Model Preservation already handled manually before submission
# TrOCR peaked at 0.8781. Qwen peaked at 0.9097.

echo "### test inference: best Qwen (R4) + TrOCR-hw (R4) [Padded Crops]"
ROCR_VISIBLE_DEVICES=0 python -m src.infer --config configs/trocr_large.yaml \
  --checkpoint runs_checkpoint/BEST_trocr_hw_r4_val0878 \
  --manifest data/Test.csv --out preds_checkpoint/trocr_test_nbest_r4.csv \
  --top1_out preds_checkpoint/trocr_test_r4.csv --nbest 5 --arm trocr &
A=$!

ROCR_VISIBLE_DEVICES=1 python -m src.infer --config configs/qwen25vl_lora.yaml \
  --checkpoint runs_checkpoint/BEST_qwen7b_r4_val0910 \
  --manifest data/Test.csv --out preds_checkpoint/vlm_test_nbest_r4.csv \
  --top1_out preds_checkpoint/vlm_test_r4.csv --nbest 5 --arm vlm &
B=$!

wait $A; RA=$?
wait $B; RB=$?
echo "exit codes: trocr=$RA vlm=$RB"

echo "### ROVER ensemble (Qwen-R4 is very strong, weight 1.5)"
python -m src.ensemble \
  --candidates preds_checkpoint/trocr_test_nbest_r4.csv preds_checkpoint/vlm_test_nbest_r4.csv \
  --out preds_checkpoint/ensemble_test_r4.csv \
  --method rover --nbest 5 \
  --arm_weights "trocr=1.0,vlm=1.5" \
  --train_csv ${TRAIN_CSV} --lex_beta 0.15

echo "### post-processing"
python -m src.postprocess \
  --pred_csv preds_checkpoint/ensemble_test_r4.csv \
  --train_csv ${TRAIN_CSV} \
  --out_csv preds_checkpoint/ensemble_test_pp_r4.csv \
  --audit_csv report/postprocess_audit_${TAG}.csv

echo "### Final submission"
python -m src.make_submission \
  --test_csv data/Test.csv --pred_csv preds_checkpoint/ensemble_test_pp_r4.csv \
  --out_csv submissions/sub_${TAG}.csv \
  --fallbacks preds_checkpoint/vlm_test_r4.csv preds_checkpoint/trocr_test_r4.csv \
  --train_csv ${TRAIN_CSV} \
  --method "rover2_r4+pp" --notes "2-arm ROVER: Qwen-7B R4 (0.9097) + TrOCR-hw R4 (0.8781) on padded crops + postproc"

report_disk
