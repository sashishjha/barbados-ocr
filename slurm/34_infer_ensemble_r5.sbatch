#!/bin/bash
#SBATCH --job-name=bbd_ens_r5
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs_checkpoint/submit_r5_%j.out
#SBATCH --error=logs_checkpoint/submit_r5_%j.err

mkdir -p logs_checkpoint preds_checkpoint submissions
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

TAG="20260925_rover_r5"
TRAIN_CSV="data/derived_checkpoint/train_and_pseudo_r5.csv"

# Golden Model Preservation
# We copy the best checkpoints so they aren't overwritten by cleanup scripts.
cp -rn runs_checkpoint/trocr_hw_checkpoints_r5/best_checkpoint runs_checkpoint/BEST_trocr_hw_r5_val0878
cp -rn runs_checkpoint/qwen25vl_lora_checkpoints_r5/best_checkpoint runs_checkpoint/BEST_qwen7b_r5_val0910

echo "### test inference: best Qwen (R5) + TrOCR-hw (R5) [Padded Crops + R5 Pseudo]"
ROCR_VISIBLE_DEVICES=0 python -m src.infer --config configs/trocr_large.yaml \
  --checkpoint runs_checkpoint/BEST_trocr_hw_r5_val0878 \
  --manifest data/Test.csv --out preds_checkpoint/trocr_test_nbest_r5_ensemble.csv \
  --top1_out preds_checkpoint/trocr_test_r5_ensemble.csv --nbest 5 --arm trocr &
A=$!

ROCR_VISIBLE_DEVICES=1 python -m src.infer --config configs/qwen25vl_lora.yaml \
  --checkpoint runs_checkpoint/BEST_qwen7b_r5_val0910 \
  --manifest data/Test.csv --out preds_checkpoint/vlm_test_nbest_r5_ensemble.csv \
  --top1_out preds_checkpoint/vlm_test_r5_ensemble.csv --nbest 5 --arm vlm &
B=$!

wait $A; RA=$?
wait $B; RB=$?
echo "exit codes: trocr=$RA vlm=$RB"

echo "### ROVER ensemble (Qwen-R5 is very strong, weight 1.5)"
python -m src.ensemble \
  --candidates preds_checkpoint/trocr_test_nbest_r5_ensemble.csv preds_checkpoint/vlm_test_nbest_r5_ensemble.csv \
  --out preds_checkpoint/ensemble_test_r5.csv \
  --method rover --nbest 5 \
  --arm_weights "trocr=1.0,vlm=1.5" \
  --train_csv ${TRAIN_CSV} --lex_beta 0.15

echo "### post-processing"
python -m src.postprocess \
  --pred_csv preds_checkpoint/ensemble_test_r5.csv \
  --train_csv ${TRAIN_CSV} \
  --out_csv preds_checkpoint/ensemble_test_pp_r5.csv \
  --audit_csv report/postprocess_audit_${TAG}.csv

echo "### Final submission"
python -m src.make_submission \
  --test_csv data/Test.csv --pred_csv preds_checkpoint/ensemble_test_pp_r5.csv \
  --out_csv submissions/sub_${TAG}.csv \
  --fallbacks preds_checkpoint/vlm_test_r5_ensemble.csv preds_checkpoint/trocr_test_r5_ensemble.csv \
  --train_csv ${TRAIN_CSV} \
  --method "rover2_r5+pp" --notes "2-arm ROVER: Qwen-7B R5 (0.9097) + TrOCR-hw R5 (0.8781) on 4,960 crops + postproc"

report_disk
