#!/bin/bash
#SBATCH --job-name=bbd_diag2
#SBATCH --partition=debug
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G

#SBATCH --time=2:00:00
#SBATCH --output=logs_checkpoint/diag2_%j.out
#SBATCH --error=logs_checkpoint/diag2_%j.err
#
# Day-1 Diagnostic Suite — Step A: val-set inference then diagnostics.
# We use fold=4 as our diagnostic val split (largest fold, matches oof_ref.csv stratum).
# Runs inference for all 3 arms on TRAIN fold-4 images, then computes all 5 diagnostics.

mkdir -p report/diagnostics logs_checkpoint preds_checkpoint
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

FOLD=${FOLD:-4}

# ---- build val manifest: train rows where fold == FOLD ----
echo "=== Building val manifest for fold ${FOLD} ==="
python - << 'PYEOF'
import os, pandas as pd
fold = int(os.environ.get("FOLD", 4))
df = pd.read_csv("data/derived_checkpoint/train_folds.csv")
val = df[df["fold"] == fold][["ID", "Target"]]
val.to_csv("preds_checkpoint/val_manifest.csv", index=False)
print(f"Val samples: {len(val)}")
PYEOF

# ---- inference on val images, one arm per GPU ----
echo "=== Val inference: 3 arms in parallel ==="
ROCR_VISIBLE_DEVICES=0 python -m src.infer \
  --config configs/trocr_large.yaml \
  --checkpoint runs_checkpoint/trocr_large_checkpoints/best_checkpoint \
  --manifest preds_checkpoint/val_manifest.csv \
  --out preds_checkpoint/val_trocr_nbest.csv \
  --top1_out preds_checkpoint/val_trocr.csv \
  --nbest 5 &
A=$!

ROCR_VISIBLE_DEVICES=1 python -m src.infer \
  --config configs/qwen25vl_lora.yaml \
  --checkpoint runs_checkpoint/qwen25vl_lora_checkpoints/best_checkpoint \
  --manifest preds_checkpoint/val_manifest.csv \
  --out preds_checkpoint/val_vlm_nbest.csv \
  --top1_out preds_checkpoint/val_vlm.csv \
  --nbest 5 &
B=$!

ROCR_VISIBLE_DEVICES=2 python -m src.infer \
  --config configs/crnn_ctc.yaml \
  --checkpoint runs_checkpoint/crnn_ctc_checkpoints/best_checkpoint \
  --manifest preds_checkpoint/val_manifest.csv \
  --out preds_checkpoint/val_ctc_nbest.csv \
  --top1_out preds_checkpoint/val_ctc.csv \
  --nbest 5 &
C=$!

wait $A; echo "trocr val done: $?"
wait $B; echo "vlm val done: $?"
wait $C; echo "ctc val done: $?"

# ---- run the full diagnostic suite ----
echo ""
echo "=== Running Day-1 Diagnostics ==="
python -m src.diagnostics \
  --gt_csv     preds_checkpoint/val_manifest.csv \
  --trocr_pred preds_checkpoint/val_trocr.csv \
  --vlm_pred   preds_checkpoint/val_vlm.csv \
  --ctc_pred   preds_checkpoint/val_ctc.csv \
  --image_dir  data/images \
  --test_csv   data/Test.csv \
  --lb_score   0.788 \
  --out_dir    report/diagnostics

echo ""
echo "=== Key outputs ==="
echo "--- Oracle CER ---"
cat report/diagnostics/oracle_cer.txt
echo ""
echo "--- LB Gap ---"
cat report/diagnostics/lb_gap.txt

report_disk
