#!/bin/bash
#SBATCH --job-name=pseudo_r3
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=logs_checkpoint/pseudo_r3_%j.out
#SBATCH --error=logs_checkpoint/pseudo_r3_%j.err

mkdir -p logs_checkpoint preds_checkpoint data/derived_checkpoint report
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

# ── Round 3 Pseudo-Labels ──────────────────────────────────────────────────
# Both arms are now genuinely strong:
#   Qwen-7B R1b   → LB 0.885
#   TrOCR-hw R2   → val 0.889
# We reuse the nbest=5 test predictions from the ensemble run (already exist).
# Filter uses both as cross-model agreement arms → much cleaner than Round 2.
# Then retrain BOTH models in parallel on 2 GPUs.

VLM_NBEST="preds_checkpoint/vlm_test_nbest_r1b.csv"
TROCR_NBEST="preds_checkpoint/trocr_test_nbest_r1b.csv"
TRAIN_CSV="data/Train.csv"
OUT_PSEUDO="data/derived_checkpoint/train_and_pseudo_r3.csv"
AUDIT_CSV="report/pseudo_round3_audit.csv"

echo "════ Step (a): filter pseudo-labels (Round 3) ════"
python - <<'PYEOF'
import os, sys
sys.path.insert(0, os.getcwd())
import pandas as pd, numpy as np, unicodedata, re

VLM_NBEST   = "preds_checkpoint/vlm_test_nbest_r1b.csv"
TROCR_NBEST = "preds_checkpoint/trocr_test_nbest_r1b.csv"
TRAIN_CSV   = "data/Train.csv"
OUT_CSV     = "data/derived_checkpoint/train_and_pseudo_r3.csv"
AUDIT_CSV   = "report/pseudo_round3_audit.csv"

train = pd.read_csv(TRAIN_CSV)
train_texts = train["Target"].dropna().astype(str).tolist()
min_len = min(len(t) for t in train_texts)
max_len = max(len(t) for t in train_texts)
train_chars = set("".join(train_texts))
print(f"[filter] train: n={len(train)}  len=[{min_len},{max_len}]  vocab={len(train_chars)}")

def edit_dist(a, b):
    a, b = str(a), str(b)
    m, n = len(a), len(b)
    dp = list(range(n+1))
    for i in range(1, m+1):
        prev, dp[0] = dp[0], i
        for j in range(1, n+1):
            temp = dp[j]
            dp[j] = prev if a[i-1]==b[j-1] else 1+min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]

def has_repetition(text, thresh=5):
    for n in range(3, 15):
        for i in range(len(text)-n*thresh+1):
            sub = text[i:i+n]
            if text[i:].startswith(sub*thresh):
                return True
    return False

def oov_ratio(text, vocab):
    chars = [c for c in text if not unicodedata.category(c).startswith('Z')]
    if not chars: return 1.0
    return sum(1 for c in chars if c not in vocab) / len(chars)

# Load VLM predictions (top-1 and top-2 for beam stability)
vlm = pd.read_csv(VLM_NBEST)
vlm_top1 = vlm[vlm["rank"]==1][["ID","Target"]].rename(columns={"Target":"vlm_top1"})
vlm_top2 = vlm[vlm["rank"]==2][["ID","Target"]].rename(columns={"Target":"vlm_top2"})

# Load TrOCR predictions (top-1 only for cross-model agreement)
trocr = pd.read_csv(TROCR_NBEST)
trocr_top1 = trocr[trocr["rank"]==1][["ID","Target"]].rename(columns={"Target":"trocr_top1"})

df = vlm_top1.merge(vlm_top2, on="ID", how="left") \
             .merge(trocr_top1, on="ID", how="left")

rows, audit = [], []
for _, row in df.iterrows():
    iid     = row["ID"]
    pred    = str(row["vlm_top1"])
    pred2   = str(row.get("vlm_top2", pred))
    trocr_p = str(row.get("trocr_top1", ""))
    reasons = []

    # 1. Repetition hallucination check
    if has_repetition(pred): reasons.append("repetition")

    # 2. Length sanity
    if len(pred) < 0.55*min_len or len(pred) > 1.45*max_len:
        reasons.append(f"length({len(pred)})")

    # 3. OOV charset check
    if oov_ratio(pred, train_chars) > 0.04: reasons.append("oov")

    # 4. Beam stability (top-1 vs top-2 must be close)
    if edit_dist(pred, pred2) > max(1, int(0.10 * len(pred))):
        reasons.append("beam_unstable")

    # 5. Cross-model agreement (both models now strong — tighter threshold 12%)
    if trocr_p:
        ed = edit_dist(pred, trocr_p)
        ref_len = max(len(pred), 1)
        if ed / ref_len > 0.12:
            reasons.append(f"disagree({ed/ref_len:.2f})")

    accepted = len(reasons) == 0
    audit.append({"ID": iid, "vlm_top1": pred, "trocr_top1": trocr_p,
                  "accepted": accepted, "reasons": "|".join(reasons)})
    if accepted:
        rows.append({"ID": iid, "Target": pred, "source": "pseudo_r3",
                     "loss_weight": 0.5})

pd.DataFrame(audit).to_csv(AUDIT_CSV, index=False)
print(f"[filter] accepted {len(rows)}/{len(df)} ({100*len(rows)/max(len(df),1):.1f}%)")

pseudo_df = pd.DataFrame(rows)
train_df  = train.copy()
train_df["source"]      = "real"
train_df["loss_weight"] = 1.0
# Cap pseudo at 30% of combined
cap = int(0.30 * (len(train_df) + len(pseudo_df)))
if len(pseudo_df) > cap:
    pseudo_df = pseudo_df.sample(cap, random_state=42)
    print(f"[filter] capped pseudo to {cap}")
combined = pd.concat([train_df, pseudo_df], ignore_index=True)
combined.to_csv(OUT_CSV, index=False)
print(f"[filter] combined CSV: {len(combined)} rows  -> {OUT_CSV}")
PYEOF

echo "════ Step (b): Retrain Qwen-7B + TrOCR-hw in parallel ════"
echo "GPU 0: Qwen-7B R1b → fine-tune on train+pseudo_r3"
ROCR_VISIBLE_DEVICES=0 python -m src.train_vlm \
  --config configs/qwen25vl_lora.yaml \
  --fold 0 \
  --set train.pseudo_csv=data/derived_checkpoint/train_and_pseudo_r3.csv \
        output.checkpoint_dir=runs_checkpoint/qwen25vl_lora_checkpoints_r3 \
        output.dir=runs_checkpoint/qwen25vl_lora_r3 &
A=$!

echo "GPU 1: TrOCR-hw → fine-tune on train+pseudo_r3"
ROCR_VISIBLE_DEVICES=1 python -m src.train_trocr \
  --config configs/trocr_large.yaml \
  --fold 0 \
  --set train.pseudo_csv=data/derived_checkpoint/train_and_pseudo_r3.csv \
        output.checkpoint_dir=runs_checkpoint/trocr_hw_checkpoints_r3 \
        output.dir=runs_checkpoint/trocr_hw_r3 &
B=$!

wait $A; RA=$?
wait $B; RB=$?
echo "exit codes: qwen=$RA  trocr=$RB"
echo "Qwen R3 done: $(grep 'new best checkpoint: metric=' logs_checkpoint/pseudo_r3_${SLURM_JOB_ID}.out | grep qwen | tail -1 || true)"

report_disk
