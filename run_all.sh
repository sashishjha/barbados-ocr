#!/usr/bin/env bash
# One-command reproduction guide. Prints the ordered pipeline; pass --submit to
# actually queue the jobs (respecting the 2-job cap by chaining dependencies).
set -euo pipefail

SUBMIT=0
[ "${1:-}" = "--submit" ] && SUBMIT=1

cat <<'BANNER'
==============================================================================
 Barbados Historic Handwriting Challenge - full pipeline
 Cluster: RBCCPS SLURM, AMD MI300X, ROCm 6.2.4
 Limits : max 3 GPUs/job, max 2 jobs queued, 200 GB storage, 24 h debug queue
==============================================================================
BANNER

echo
echo "PRE-FLIGHT (login node, no GPU needed):"
echo "  python -m pytest tests/ -q"
echo
echo "STAGE 0  environment + smoke test        sbatch slurm/00_setup_smoke_test.sbatch"
echo "STAGE 1  data prep (CPU only)            sbatch slurm/01_prepare_data.sbatch"
echo "         -> read report/eda/length_buckets.csv BEFORE tuning anything"
echo "STAGE 2  three arms, 3 GPUs, ONE job     FOLD=0 sbatch slurm/12_parallel_arms.sbatch"
echo "         (or run arms individually: slurm/10_train_trocr.sbatch, 11_train_vlm.sbatch)"
echo "STAGE 3  decode tuning (cheap, do first) python -m src.tune_optuna --mode decode ..."
echo "STAGE 4  pseudo-label round              ROUND=1 sbatch slurm/20_pseudo_label_round.sbatch"
echo "STAGE 5  ensemble + submissions          TAG=v1 sbatch slurm/30_inference_ensemble.sbatch"
echo
echo "A 24h-killed job resumes by RESUBMITTING THE SAME SCRIPT (--resume is built in)."
echo

if [ "$SUBMIT" -eq 1 ]; then
  mkdir -p logs_checkpoint preds_checkpoint submissions
  J0=$(sbatch --parsable slurm/00_setup_smoke_test.sbatch)
  echo "queued setup      : $J0"
  J1=$(sbatch --parsable --dependency=afterok:$J0 slurm/01_prepare_data.sbatch)
  echo "queued data prep  : $J1"
  echo
  echo "Stopping here on purpose: inspect the smoke test and the EDA output before"
  echo "committing a 24 h training slot. Then submit stage 2 yourself."
else
  echo "Dry run. Re-run with --submit to queue stages 0-1."
fi
