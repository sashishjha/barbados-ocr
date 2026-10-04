#!/bin/bash
#SBATCH --job-name=bbd_ens_r2
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs_checkpoint/submit_r2_%j.out
#SBATCH --error=logs_checkpoint/submit_r2_%j.err

mkdir -p logs_checkpoint preds_checkpoint submissions
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

TAG="20260923_rover_r2"
TRAIN_CSV="data/derived_checkpoint/train_and_pseudo_r2.csv"

echo "### test inference, one arm per GPU (nbest=5 for ROVER)"
ROCR_VISIBLE_DEVICES=0 python -m src.infer --config configs/trocr_large.yaml \
  --checkpoint runs_checkpoint/BEST_trocr_hw_r2_val0889 \
  --manifest data/Test.csv --out preds_checkpoint/trocr_test_nbest_r2.csv \
  --top1_out preds_checkpoint/trocr_test_r2.csv --nbest 5 --arm trocr &
A=$!

ROCR_VISIBLE_DEVICES=1 python -m src.infer --config configs/qwen25vl_lora.yaml \
  --checkpoint runs_checkpoint/BEST_qwen7b_r2_val0911 \
  --manifest data/Test.csv --out preds_checkpoint/vlm_test_nbest_r2.csv \
  --top1_out preds_checkpoint/vlm_test_r2.csv --nbest 5 --arm vlm &
B=$!

wait $A; RA=$?
wait $B; RB=$?
echo "exit codes: trocr=$RA vlm=$RB"
[ $RB -ne 0 ] && echo "WARNING: VLM arm failed; ensemble will be weaker"
[ $RA -ne 0 ] && echo "WARNING: TrOCR arm failed"

echo "### ROVER ensemble"
# Giving Qwen slightly more weight (1.3) since its val score is 0.911 vs TrOCR 0.889
python -m src.ensemble \
  --candidates preds_checkpoint/trocr_test_nbest_r2.csv preds_checkpoint/vlm_test_nbest_r2.csv \
  --out preds_checkpoint/ensemble_test_r2.csv \
  --method rover --nbest 5 \
  --arm_weights "trocr=1.0,vlm=1.3" \
  --train_csv ${TRAIN_CSV} --lex_beta 0.15

echo "### train-lexicon post-processing"
python -m src.postprocess \
  --pred_csv preds_checkpoint/ensemble_test_r2.csv \
  --train_csv ${TRAIN_CSV} \
  --out_csv preds_checkpoint/ensemble_test_pp_r2.csv \
  --audit_csv report/postprocess_audit_${TAG}.csv

echo "### Create final submission CSV"
python -m src.make_submission \
  --test_csv data/Test.csv --pred_csv preds_checkpoint/ensemble_test_pp_r2.csv \
  --out_csv submissions/sub_${TAG}.csv \
  --fallbacks preds_checkpoint/vlm_test_r2.csv preds_checkpoint/trocr_test_r2.csv \
  --train_csv ${TRAIN_CSV} \
  --method "rover2_r2+pp" --notes "2-arm ROVER (Qwen-7B + TrOCR-hw) on R2 pseudo labels + postproc"

report_disk
