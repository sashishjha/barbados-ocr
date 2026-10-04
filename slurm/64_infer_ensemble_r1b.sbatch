#!/bin/bash
#SBATCH --job-name=bbd_ens_r1b
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs_checkpoint/submit_r1b_%j.out
#SBATCH --error=logs_checkpoint/submit_r1b_%j.err

mkdir -p logs_checkpoint preds_checkpoint submissions
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

TAG="20260923_rover_r1b"
TRAIN_CSV="data/derived_checkpoint/train_and_pseudo_r2.csv"

echo "### test inference: best Qwen (R1b, LB 0.885) + TrOCR-hw"
ROCR_VISIBLE_DEVICES=0 python -m src.infer --config configs/trocr_large.yaml \
  --checkpoint runs_checkpoint/BEST_trocr_hw_r2_val0889 \
  --manifest data/Test.csv --out preds_checkpoint/trocr_test_nbest_r1b.csv \
  --top1_out preds_checkpoint/trocr_test_r1b.csv --nbest 5 --arm trocr &
A=$!

ROCR_VISIBLE_DEVICES=1 python -m src.infer --config configs/qwen25vl_lora.yaml \
  --checkpoint runs_checkpoint/BEST_qwen7b_r1b_LB0885 \
  --manifest data/Test.csv --out preds_checkpoint/vlm_test_nbest_r1b.csv \
  --top1_out preds_checkpoint/vlm_test_r1b.csv --nbest 5 --arm vlm &
B=$!

wait $A; RA=$?
wait $B; RB=$?
echo "exit codes: trocr=$RA vlm=$RB"

echo "### ROVER ensemble (Qwen-R1b is stronger on LB so weight 1.4)"
python -m src.ensemble \
  --candidates preds_checkpoint/trocr_test_nbest_r1b.csv preds_checkpoint/vlm_test_nbest_r1b.csv \
  --out preds_checkpoint/ensemble_test_r1b.csv \
  --method rover --nbest 5 \
  --arm_weights "trocr=1.0,vlm=1.4" \
  --train_csv ${TRAIN_CSV} --lex_beta 0.15

echo "### post-processing"
python -m src.postprocess \
  --pred_csv preds_checkpoint/ensemble_test_r1b.csv \
  --train_csv ${TRAIN_CSV} \
  --out_csv preds_checkpoint/ensemble_test_pp_r1b.csv \
  --audit_csv report/postprocess_audit_${TAG}.csv

echo "### Final submission"
python -m src.make_submission \
  --test_csv data/Test.csv --pred_csv preds_checkpoint/ensemble_test_pp_r1b.csv \
  --out_csv submissions/sub_${TAG}.csv \
  --fallbacks preds_checkpoint/vlm_test_r1b.csv preds_checkpoint/trocr_test_r1b.csv \
  --train_csv ${TRAIN_CSV} \
  --method "rover2_r1b+pp" --notes "2-arm ROVER: Qwen-7B R1b (LB0885) + TrOCR-hw + postproc"

report_disk
