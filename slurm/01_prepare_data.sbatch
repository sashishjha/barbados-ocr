#!/bin/bash
#SBATCH --job-name=bbd_prep
#SBATCH --partition=debug
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G

#SBATCH --time=04:00:00
#SBATCH --output=logs_checkpoint/prep_%j.out
#SBATCH --error=logs_checkpoint/prep_%j.err
#
# STEP 1. CPU-only data preparation. Produces everything under data/derived_checkpoint/
# and report/eda/. Fully deterministic from the seed.

mkdir -p logs_checkpoint
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

echo "### 1. corrupted-row verification + documented exclusions"
python -m src.data.exclusions \
    --train_csv data/Train.csv --image_dir data/images \
    --out_csv data/derived_checkpoint/train_clean.csv \
    --audit_csv report/exclusion_audit.csv

echo "### 2. multi-line census (known label-convention issue)"
python -m src.data.multiline \
    --train_csv data/derived_checkpoint/train_clean.csv --image_dir data/images \
    --out_csv data/derived_checkpoint/multiline_census.csv

echo "### 3. stratified folds + OOF reference"
python -m src.data.folds \
    --train_csv data/derived_checkpoint/train_clean.csv \
    --census_csv data/derived_checkpoint/multiline_census.csv \
    --out_csv data/derived_checkpoint/train_folds.csv \
    --oof_ref_csv data/derived_checkpoint/oof_ref.csv --n_splits 5 --seed 42

echo "### 4. EDA"
python -m src.eda --train_csv data/Train.csv --image_dir data/images \
    --test_csv data/Test.csv --out_dir report/eda

echo "### 5. StackMix crop bank (provided TRAIN images only)"
python -m src.data.stackmix \
    --train_csv data/derived_checkpoint/train_clean.csv --image_dir data/images \
    --bank_dir data/derived_checkpoint/stackmix_bank --seed 42

report_disk
