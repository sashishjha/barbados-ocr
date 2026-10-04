#!/bin/bash
#SBATCH --job-name=infer_r1
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=logs_checkpoint/infer_r1_%j.out
#SBATCH --error=logs_checkpoint/infer_r1_%j.err

mkdir -p logs_checkpoint preds_checkpoint submissions
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

TAG="r1_$(date +%Y%m%d_%H%M)"

echo "### test inference, one arm per GPU"
# Note: we are using the r1 checkpoints now
ROCR_VISIBLE_DEVICES=0 python -m src.infer --config configs/trocr_large.yaml \
  --checkpoint runs_checkpoint/trocr_large_checkpoints_r1/best_checkpoint \
  --manifest data/Test.csv --out preds_checkpoint/trocr_r1_test_nbest.csv \
  --top1_out preds_checkpoint/trocr_r1_test.csv --nbest 5 &
A=$!
ROCR_VISIBLE_DEVICES=1 python -m src.infer --config configs/qwen25vl_lora.yaml \
  --checkpoint runs_checkpoint/qwen25vl_lora_checkpoints_r1/best_checkpoint \
  --manifest data/Test.csv --out preds_checkpoint/vlm_r1_test_nbest.csv \
  --top1_out preds_checkpoint/vlm_r1_test.csv --nbest 5 &
B=$!

wait $A; RA=$?
wait $B; RB=$?

echo "exit codes: trocr=$RA vlm=$RB"
if [ $RA -ne 0 ] || [ $RB -ne 0 ]; then
    echo "ERROR: inference failed"
    exit 1
fi

echo "### ensemble (2-arm ROVER)"
python -m src.ensemble \
  --candidates preds_checkpoint/trocr_r1_test_nbest.csv preds_checkpoint/vlm_r1_test_nbest.csv \
  --out preds_checkpoint/ensemble_r1_test.csv \
  --method rover --nbest 3 \
  --arm_weights "trocr=1.0,vlm=1.2" \
  --train_csv data/derived_checkpoint/train_and_pseudo_r1.csv --lex_beta 0.15

echo "### train-lexicon post-processing"
python -m src.postprocess \
  --pred_csv preds_checkpoint/ensemble_r1_test.csv \
  --train_csv data/derived_checkpoint/train_and_pseudo_r1.csv \
  --out_csv preds_checkpoint/ensemble_r1_test_pp.csv \
  --audit_csv report/postprocess_audit_${TAG}.csv

echo "### Create final submission CSV"
python -m src.make_submission \
  --test_csv data/Test.csv --pred_csv preds_checkpoint/ensemble_r1_test_pp.csv \
  --out_csv submissions/sub_ensemble_${TAG}.csv \
  --fallbacks preds_checkpoint/vlm_r1_test.csv preds_checkpoint/trocr_r1_test.csv \
  --train_csv data/derived_checkpoint/train_and_pseudo_r1.csv \
  --method "rover2+pp" --notes "2-arm ROVER (Qwen-7B, TrOCR-stage1) + postproc"


report_disk
