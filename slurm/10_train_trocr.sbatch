#!/bin/bash
#SBATCH --job-name=bbd_trocr
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G

#SBATCH --time=24:00:00
#SBATCH --output=logs_checkpoint/trocr_%j.out
#SBATCH --error=logs_checkpoint/trocr_%j.err
#
# ARM A. Resubmit this exact script after a 24h kill: --resume picks up the
# optimizer, scheduler, epoch/step and RNG state from the last checkpoint.
#   FOLD=0 sbatch slurm/10_train_trocr.sbatch

mkdir -p logs_checkpoint preds_checkpoint
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

FOLD=${FOLD:-0}
export ROCR_VISIBLE_DEVICES=0

python -m src.train_trocr \
    --config configs/trocr_large.yaml \
    --fold "${FOLD}" \
    --resume

echo "### OOF inference for the ensemble (n-best)"
python -m src.infer \
    --config configs/trocr_large.yaml \
    --checkpoint runs_checkpoint/trocr_large_checkpoints/best_checkpoint \
    --manifest data/derived_checkpoint/train_folds.csv \
    --out preds_checkpoint/trocr_fold${FOLD}_oof_nbest.csv \
    --top1_out preds_checkpoint/trocr_fold${FOLD}_oof.csv \
    --nbest 5

report_disk
