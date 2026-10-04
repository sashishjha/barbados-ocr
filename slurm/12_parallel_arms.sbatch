#!/bin/bash
#SBATCH --job-name=bbd_par3
#SBATCH --partition=debug
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=100G

#SBATCH --time=24:00:00
#SBATCH --output=logs_checkpoint/par3_%j.out
#SBATCH --error=logs_checkpoint/par3_%j.err
#
# Three INDEPENDENT experiments inside ONE allocation, round-robin pinned across
# the 3 GPUs. This is how we get 3 concurrent experiments out of a 2-job cap:
# the whole thing counts as a single job.
#
#   gpu0 = TrOCR arm
#   gpu1 = VLM arm
#   gpu2 = Optuna decode-tuning worker (cheap) or the CTC arm
#
# Every child gets --resume, so a 24h kill + resubmit continues all three.

mkdir -p logs_checkpoint preds_checkpoint
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

FOLD=${FOLD:-0}
THIRD=${THIRD:-ctc}     # ctc | optuna

declare -a CMDS
CMDS[0]="python -m src.train_trocr --config configs/trocr_large.yaml --fold ${FOLD} --resume"
CMDS[1]="python -m src.train_vlm   --config configs/qwen25vl_lora.yaml --fold ${FOLD} --resume"
if [ "$THIRD" = "optuna" ]; then
  CMDS[2]="python -m src.tune_optuna --config configs/trocr_large.yaml --mode decode \
           --candidates preds_checkpoint/trocr_fold${FOLD}_oof_nbest.csv preds_checkpoint/vlm_fold${FOLD}_oof_nbest.csv \
           --arms trocr vlm --oof_ref data/derived_checkpoint/oof_ref.csv \
           --n_trials 60 --timeout_min 300 --study_dir runs_checkpoint/optuna"
else
  CMDS[2]="python -m src.train_ctc --config configs/crnn_ctc.yaml --fold ${FOLD} --resume"
fi

declare -a PIDS
for IDX in 0 1 2; do
  GPU=$(( IDX % 3 ))
  echo ">>> launching arm ${IDX} on GPU ${GPU}: ${CMDS[$IDX]}"
  ROCR_VISIBLE_DEVICES=${GPU} ${CMDS[$IDX]} > "logs_checkpoint/par3_${SLURM_JOB_ID}_arm${IDX}.log" 2>&1 &
  PIDS[$IDX]=$!
done

# Wait and collect exit codes; do not let one failure hide the others.
FAIL=0
for IDX in 0 1 2; do
  if wait "${PIDS[$IDX]}"; then
    echo "arm ${IDX}: OK"
  else
    RC=$?
    echo "arm ${IDX}: FAILED (exit ${RC}) - see logs_checkpoint/par3_${SLURM_JOB_ID}_arm${IDX}.log"
    FAIL=1
  fi
done

report_disk
exit $FAIL
