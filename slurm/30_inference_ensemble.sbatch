#!/bin/bash
#SBATCH --job-name=bbd_submit
#SBATCH --partition=debug
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --mem=100G

#SBATCH --time=12:00:00
#SBATCH --output=logs_checkpoint/submit_%j.out
#SBATCH --error=logs_checkpoint/submit_%j.err
#
# Final pipeline: test inference for all arms (parallel across 3 GPUs) ->
# ensemble -> train-lexicon post-processing -> validated submission -> log row.

mkdir -p logs_checkpoint preds_checkpoint submissions
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

TAG=${TAG:-$(date +%Y%m%d_%H%M)}

echo "### test inference, one arm per GPU"
# Dynamically select the 3 freest GPUs to bypass Slurm zombie processes
BEST_GPUS=($(rocm-smi --showmeminfo vram | grep "VRAM Total Used Memory" | sort -nk8 | head -n 3 | awk -F'[][]' '{print $2}'))
GPU_A=${BEST_GPUS[0]}
GPU_B=${BEST_GPUS[1]}
GPU_C=${BEST_GPUS[2]}
echo "Selected Best GPUs: $GPU_A, $GPU_B, $GPU_C"

ROCR_VISIBLE_DEVICES=$GPU_A python -m src.infer --config configs/trocr_large.yaml \
  --checkpoint runs_checkpoint/trocr_large_checkpoints/best_checkpoint \
  --manifest data/Test.csv --out preds_checkpoint/trocr_test_nbest.csv \
  --top1_out preds_checkpoint/trocr_test.csv --nbest 5 &
A=$!
ROCR_VISIBLE_DEVICES=$GPU_B python -m src.infer --config configs/qwen25vl_lora.yaml \
  --checkpoint runs_checkpoint/qwen25vl_lora_checkpoints/best_checkpoint \
  --manifest data/Test.csv --out preds_checkpoint/vlm_test_nbest.csv \
  --top1_out preds_checkpoint/vlm_test.csv --nbest 5 --batch_size 1 &
B=$!
ROCR_VISIBLE_DEVICES=$GPU_C python -m src.infer --config configs/crnn_ctc.yaml \
  --checkpoint runs_checkpoint/crnn_ctc_checkpoints/best_checkpoint \
  --manifest data/Test.csv --out preds_checkpoint/ctc_test_nbest.csv \
  --top1_out preds_checkpoint/ctc_test.csv --nbest 5 &
C=$!
wait $A; RA=$?
wait $B; RB=$?
wait $C; RC=$?
echo "exit codes: trocr=$RA vlm=$RB ctc=$RC"
if [ $RB -ne 0 ]; then
  echo "WARNING: VLM arm failed; ensemble will be weaker"
fi

echo "### ensemble (weights should come from the Optuna decode study)"
python -m src.ensemble \
  --candidates preds_checkpoint/trocr_test_nbest.csv preds_checkpoint/vlm_test_nbest.csv preds_checkpoint/ctc_test_nbest.csv \
  --out preds_checkpoint/ensemble_test.csv \
  --method rover --nbest 3 \
  --arm_weights "${ARM_WEIGHTS:-trocr=1.0,vlm=1.3,ctc=0.7}" \
  --train_csv data/derived_checkpoint/train_clean.csv --lex_beta 0.15

echo "### train-lexicon post-processing (rules fitted on TRAIN only)"
python -m src.postprocess \
  --pred_csv preds_checkpoint/ensemble_test.csv \
  --train_csv data/derived_checkpoint/train_clean.csv \
  --out_csv preds_checkpoint/ensemble_test_pp.csv \
  --audit_csv report/postprocess_audit_${TAG}.csv

echo "### submission A: best ensemble"
python -m src.make_submission \
  --test_csv data/Test.csv --pred_csv preds_checkpoint/ensemble_test_pp.csv \
  --out_csv submissions/sub_ensemble_${TAG}.csv \
  --fallbacks preds_checkpoint/vlm_test.csv preds_checkpoint/trocr_test.csv preds_checkpoint/ctc_test.csv \
  --train_csv data/derived_checkpoint/train_clean.csv \
  --method "rover3+pp" --notes "3-arm ROVER + train-lexicon postproc"

echo "### submission B: best SINGLE model (private-split diversification)"
python -m src.postprocess --pred_csv preds_checkpoint/vlm_test.csv \
  --train_csv data/derived_checkpoint/train_clean.csv --out_csv preds_checkpoint/vlm_test_pp.csv
python -m src.make_submission \
  --test_csv data/Test.csv --pred_csv preds_checkpoint/vlm_test_pp.csv \
  --out_csv submissions/sub_single_${TAG}.csv \
  --fallbacks preds_checkpoint/trocr_test.csv preds_checkpoint/ctc_test.csv \
  --train_csv data/derived_checkpoint/train_clean.csv \
  --method "vlm_single+pp" --notes "single-model hedge vs public/private shift"

report_disk
