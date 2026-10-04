#!/usr/bin/env bash
# Upload the six final ensemble members (LoRA adapters; full weights for TrOCR) to a Hugging Face model repo.
# Run from the project root on the machine that holds the checkpoints. Keep the token in this shell only:
#   read -rs HF_TOKEN; export HF_TOKEN
#   bash scripts/upload_members_to_hf.sh <hf-username>/barbados-ocr-adapters
# Layout in the repo = member names used in docs/REPRODUCIBILITY.md (q32, q3_vis, q3_vis_s7, q7_vis, q7_r8b, trocr_r5).
set -euo pipefail
REPO=${1:?usage: bash scripts/upload_members_to_hf.sh <hf-username>/<repo-name>}
: "${HF_TOKEN:?export HF_TOKEN first (read -rs HF_TOKEN; export HF_TOKEN)}"
HF=${HF_CLI:-.venv/bin/hf}
R=runs_checkpoint
$HF auth whoami
$HF repo create "$REPO" --repo-type model --private --exist-ok
up() { echo "== $2 <- $1"; $HF upload "$REPO" "$1" "$2" --repo-type model; }
up $R/BEST_qwen3vl32b_r10_f55_09002                          q32
up $R/qwen3vl8b_r17_vis128_all_checkpoints/best_checkpoint    q3_vis
up $R/qwen3vl8b_r18_vis128_all_s7_checkpoints/best_checkpoint q3_vis_s7
up $R/qwen7b_r18_vis128_all_checkpoints/best_checkpoint       q7_vis
up $R/BEST_qwen7b_r8b_f55_08829                               q7_r8b
up $R/BEST_trocr_hw_r5_val0878                                trocr_r5
echo "UPLOAD DONE: https://huggingface.co/$REPO"
