#!/bin/bash
#SBATCH --job-name=pseudo_r2
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=logs_checkpoint/pseudo_r2_%j.out
#SBATCH --error=logs_checkpoint/pseudo_r2_%j.err

mkdir -p logs_checkpoint preds_checkpoint data/derived_checkpoint report
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

ROUND=2

echo "=== ROUND 2 PSEUDO-LABELS ==="
echo "Primary arm: Qwen-7B R1b (LB 0.885)"
echo "Secondary arm: TrOCR-stage1 R1 (used ONLY for agreement filtering, not submission)"

echo "### (a) Re-infer TEST set with nbest=5 for beam stability filter"
# Qwen-7B R1b — high quality primary arm
ROCR_VISIBLE_DEVICES=0 python -m src.infer \
    --config configs/qwen25vl_lora.yaml \
    --checkpoint runs_checkpoint/BEST_qwen7b_r1b_LB0885 \
    --manifest data/Test.csv \
    --out preds_checkpoint/vlm_r1b_test_nbest_r2.csv \
    --top1_out preds_checkpoint/vlm_r1b_test_r2.csv \
    --arm vlm --nbest 5 &
P1=$!

# TrOCR-stage1 — used ONLY for cross-model agreement filter
ROCR_VISIBLE_DEVICES=1 python -m src.infer \
    --config configs/trocr_large.yaml \
    --checkpoint runs_checkpoint/trocr_large_checkpoints_r1/best_checkpoint \
    --manifest data/Test.csv \
    --out preds_checkpoint/trocr_r1_test_nbest_r2.csv \
    --top1_out preds_checkpoint/trocr_r1_test_r2.csv \
    --arm trocr --nbest 5 &
P2=$!

wait $P1 || { echo "Qwen inference failed"; exit 1; }
wait $P2 || { echo "TrOCR inference failed"; exit 1; }

echo "### (b) Filter pseudo-labels"
python - <<'PYEOF'
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.getcwd())
from Levenshtein import distance

print("=== Generating Round 2 Pseudo-Labels ===")

vlm_nbest = pd.read_csv("preds_checkpoint/vlm_r1b_test_nbest_r2.csv").fillna("")
trocr_nbest = pd.read_csv("preds_checkpoint/trocr_r1_test_nbest_r2.csv").fillna("")
train = pd.read_csv("data/derived_checkpoint/train_clean.csv")

# Extract top-1 and top-2 candidates per ID from each arm
def get_topk(df, k, arm_col="arm"):
    return df[df["rank"] < k].copy()

vlm_top1 = vlm_nbest[vlm_nbest["rank"] == 0].set_index("ID")[["Target", "score"]].rename(columns={"Target": "vlm", "score": "vlm_score"})
vlm_top2 = vlm_nbest[vlm_nbest["rank"] == 1].set_index("ID")[["Target"]].rename(columns={"Target": "vlm_2nd"})
trocr_top1 = trocr_nbest[trocr_nbest["rank"] == 0].set_index("ID")[["Target"]].rename(columns={"Target": "trocr"})

merged = vlm_top1.join(vlm_top2, how="left").join(trocr_top1, how="left").fillna("")

# Train set stats for sanity filters
train_lengths = train["Target"].str.len()
len_min = train_lengths.min() * 0.6
len_max = train_lengths.max() * 1.4
train_charset = set("".join(train["Target"].dropna().tolist()))

# Per-arm score percentile threshold (p40)
score_thresh = np.percentile(merged["vlm_score"].values, 40)

accepted, audit = [], []

def check_repetition(text):
    words = text.split()
    for i in range(len(words) - 3):
        if words[i] == words[i+1] == words[i+2] == words[i+3]:
            return True
    return False

for idx, row in merged.iterrows():
    tv = str(row["vlm"]).strip()
    tv2 = str(row.get("vlm_2nd", "")).strip()
    tt = str(row.get("trocr", "")).strip()
    score = row["vlm_score"]
    reason = None

    # Filter 1: hallucination (repetition loop)
    if check_repetition(tv):
        reason = "hallucination"
    # Filter 2: length sanity
    elif not (len_min <= len(tv) <= len_max):
        reason = "length_sanity"
    # Filter 3: charset sanity
    elif not set(tv).issubset(train_charset):
        reason = f"charset_oov:{set(tv)-train_charset}"
    # Filter 4: beam stability (top1 vs top2 edit distance)
    elif tv2 and distance(tv, tv2) / max(len(tv), 1) > 0.10:
        reason = "beam_unstable"
    # Filter 5: confidence percentile
    elif score < score_thresh:
        reason = "low_confidence"
    # Filter 6: cross-model agreement (VLM vs TrOCR, ≤15% edit distance)
    elif tt and distance(tv, tt) / max(len(tv), 1) > 0.15:
        reason = "cross_model_disagree"
    # Filter 7: non-empty
    elif not tv:
        reason = "empty"
    else:
        accepted.append({"ID": idx, "Target": tv, "source": "pseudo_r2", "fold": -1,
                         "loss_weight": 0.5, "vlm_score": score})

    audit.append({"ID": idx, "accepted": reason is None, "reason": reason or "ok",
                  "vlm": tv, "trocr": tt, "vlm_score": score})

print(f"Total test lines: {len(merged)}")
print(f"Accepted: {len(accepted)} ({len(accepted)/len(merged)*100:.1f}%)")
reasons = pd.DataFrame(audit)["reason"].value_counts()
print("Rejection reasons:"); print(reasons[reasons.index != "ok"])

# Write audit
pd.DataFrame(audit).to_csv("report/pseudo_round2_audit.csv", index=False)

# Combine with original train set (NOT the R1 pseudo-labels — fresh start)
pseudo_df = pd.DataFrame(accepted)
train["source"] = "train"
train["loss_weight"] = 1.0

# Cap at 30% of training set
max_pseudo = int(len(train) * 0.30)
if len(pseudo_df) > max_pseudo:
    pseudo_df = pseudo_df.sample(max_pseudo, random_state=42)
    print(f"Capped pseudo-labels to {max_pseudo}")

combined = pd.concat([train, pseudo_df], ignore_index=True)
out_path = "data/derived_checkpoint/train_and_pseudo_r2.csv"
combined.to_csv(out_path, index=False)
print(f"Wrote {out_path} with {len(combined)} rows ({len(pseudo_df)} pseudo-labels)")
PYEOF

echo "### (c) Retrain Qwen-7B with Round 2 pseudo-labels"
# Wipe old r2 checkpoint dir and retrain from scratch
rm -rf runs_checkpoint/qwen25vl_lora_checkpoints_r2

ROCR_VISIBLE_DEVICES=0 python -m src.train_vlm \
    --config configs/qwen25vl_lora.yaml \
    --fold 0 \
    --set output.checkpoint_dir=runs_checkpoint/qwen25vl_lora_checkpoints_r2 \
          output.dir=runs_checkpoint/qwen25vl_lora_r2 \
          data.pseudo_csv=data/derived_checkpoint/train_and_pseudo_r2.csv \
          data.pseudo_max_ratio=0.30 \
          data.pseudo_loss_weight=0.5

echo "Qwen R2 done"
report_disk
