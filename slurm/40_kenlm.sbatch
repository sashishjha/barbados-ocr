#!/bin/bash
#SBATCH --job-name=kenlm
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs_checkpoint/kenlm_%j.out
#SBATCH --error=logs_checkpoint/kenlm_%j.err

mkdir -p logs_checkpoint data/derived_checkpoint/lm
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

# Ensure KenLM python package is installed
echo "Installing python kenlm wrapper..."
pip install https://github.com/kpu/kenlm/archive/master.zip

# Ensure lmplz and build_binary are available
# We will download and build them if they don't exist in ~/kenlm/build/bin
if [ ! -f ~/kenlm/build/bin/lmplz ]; then
    echo "Compiling KenLM C++ binaries in ~/kenlm..."
    cd ~
    wget -qO- https://kpu.github.io/kenlm/dependencies.sh | bash
    git clone https://github.com/kpu/kenlm.git
    cd kenlm
    mkdir -p build && cd build
    cmake ..
    make -j 16
    cd "${SLURM_SUBMIT_DIR}"
fi

export PATH="$HOME/kenlm/build/bin:$PATH"

echo "### Training KenLM LMs (Word 4-gram, Char 10-gram)..."
python -m src.kenlm_rescore \
    --action train \
    --train_csv data/derived_checkpoint/train_and_pseudo_r4.csv \
    --word_lm data/derived_checkpoint/lm/word_4gram.bin \
    --char_lm data/derived_checkpoint/lm/char_10gram.bin

echo "### Rescoring Test Set..."
python -m src.kenlm_rescore \
    --action rescore \
    --candidates preds_checkpoint/vlm_test_nbest_r4.csv preds_checkpoint/trocr_test_nbest_r4.csv \
    --out preds_checkpoint/rescored_test_r4.csv \
    --alpha_word 0.5 --alpha_char 0.1 --beta 1.5

echo "### Applying Post-processing to Rescored Candidates..."
python -m src.postprocess \
  --pred_csv preds_checkpoint/rescored_test_r4.csv \
  --train_csv data/derived_checkpoint/train_and_pseudo_r4.csv \
  --out_csv preds_checkpoint/rescored_test_pp_r4.csv \
  --audit_csv report/postprocess_audit_kenlm_r4.csv

echo "### Final submission generation"
python -m src.make_submission \
  --test_csv data/Test.csv --pred_csv preds_checkpoint/rescored_test_pp_r4.csv \
  --out_csv submissions/sub_20260924_kenlm_r4.csv \
  --fallbacks preds_checkpoint/vlm_test_r4.csv preds_checkpoint/trocr_test_r4.csv \
  --train_csv data/derived_checkpoint/train_and_pseudo_r4.csv \
  --method "kenlm_r4+pp" --notes "KenLM rescoring of Qwen-7B R4 + TrOCR-hw R4 (alpha_w=0.5, alpha_c=0.1, beta=1.5)"

echo "Done!"
