#!/bin/bash
#SBATCH --job-name=pseudo_r4
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=logs_checkpoint/pseudo_r4_%j.out
#SBATCH --error=logs_checkpoint/pseudo_r4_%j.err

mkdir -p logs_checkpoint preds_checkpoint data/derived_checkpoint report
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

# ── Round 4: Padded-Crop Pseudo-Labels ────────────────────────────────────
# KEY CHANGE vs R3: vertical_pad_frac=0.20 is now baked into preprocess.py
# (applied automatically via PreprocessConfig default + vlm_thumbnail).
# Both teacher models are the new R3 gold checkpoints:
#   Qwen-7B R3   → val 0.9100 (new PB)
#   TrOCR-hw R3  → val 0.8896

echo "════ Step (a): Generate N-best on Test with R3 padded-crop models ════"

echo "GPU 0: Qwen-7B R3 inference (padded crops, nbest=5)"
ROCR_VISIBLE_DEVICES=0 python -m src.infer --config configs/qwen25vl_lora.yaml \
  --checkpoint runs_checkpoint/BEST_qwen7b_r3_val0910 \
  --manifest data/Test.csv --out preds_checkpoint/vlm_test_nbest_r4.csv \
  --top1_out preds_checkpoint/vlm_test_r4.csv --nbest 5 --arm vlm &
A=$!

echo "GPU 1: TrOCR-hw R3 inference (padded crops, nbest=5)"
ROCR_VISIBLE_DEVICES=1 python -m src.infer --config configs/trocr_large.yaml \
  --checkpoint runs_checkpoint/BEST_trocr_hw_r3_val0889 \
  --manifest data/Test.csv --out preds_checkpoint/trocr_test_nbest_r4.csv \
  --top1_out preds_checkpoint/trocr_test_r4.csv --nbest 5 --arm trocr &
B=$!

wait $A; RA=$?
wait $B; RB=$?
echo "exit codes: qwen=$RA  trocr=$RB"

echo "════ Step (b): Filter pseudo-labels (Round 4) ════"
python - <<'PYEOF'
import os, sys
sys.path.insert(0, os.getcwd())
import pandas as pd, numpy as np, unicodedata

VLM_NBEST   = "preds_checkpoint/vlm_test_nbest_r4.csv"
TROCR_NBEST = "preds_checkpoint/trocr_test_nbest_r4.csv"
TRAIN_CSV   = "data/Train.csv"
OUT_CSV     = "data/derived_checkpoint/train_and_pseudo_r4.csv"
AUDIT_CSV   = "report/pseudo_round4_audit.csv"

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

vlm   = pd.read_csv(VLM_NBEST)
trocr = pd.read_csv(TROCR_NBEST)
vlm_top1  = vlm[vlm["rank"]==0][["ID","Target"]].rename(columns={"Target":"vlm_top1"})
vlm_top2  = vlm[vlm["rank"]==1][["ID","Target"]].rename(columns={"Target":"vlm_top2"})
trocr_top1 = trocr[trocr["rank"]==0][["ID","Target"]].rename(columns={"Target":"trocr_top1"})

df = vlm_top1.merge(vlm_top2, on="ID", how="left") \
             .merge(trocr_top1, on="ID", how="left")

rows, audit = [], []
for _, row in df.iterrows():
    iid     = row["ID"]
    pred    = str(row["vlm_top1"])
    pred2   = str(row.get("vlm_top2", pred))
    trocr_p = str(row.get("trocr_top1", ""))
    reasons = []

    if has_repetition(pred): reasons.append("repetition")
    if len(pred) < 0.55*min_len or len(pred) > 1.45*max_len:
        reasons.append(f"length({len(pred)})")
    if oov_ratio(pred, train_chars) > 0.04: reasons.append("oov")
    if edit_dist(pred, pred2) > max(1, int(0.10 * len(pred))):
        reasons.append("beam_unstable")
    if trocr_p:
        ed = edit_dist(pred, trocr_p)
        ref_len = max(len(pred), 1)
        # R4: tighter threshold (10%) since both models are now stronger
        if ed / ref_len > 0.10:
            reasons.append(f"disagree({ed/ref_len:.2f})")

    accepted = len(reasons) == 0
    audit.append({"ID": iid, "vlm_top1": pred, "trocr_top1": trocr_p,
                  "accepted": accepted, "reasons": "|".join(reasons)})
    if accepted:
        rows.append({"ID": iid, "Target": pred, "source": "pseudo_r4",
                     "loss_weight": 0.8})  # R4: higher trust (0.8 per roadmap)

pd.DataFrame(audit).to_csv(AUDIT_CSV, index=False)
print(f"[filter] accepted {len(rows)}/{len(df)} ({100*len(rows)/max(len(df),1):.1f}%)")

pseudo_df = pd.DataFrame(rows)
train_df  = train.copy()
train_df["source"]      = "real"
train_df["loss_weight"] = 1.0
cap = int(0.35 * (len(train_df) + len(pseudo_df)))
if len(pseudo_df) > cap:
    pseudo_df = pseudo_df.sample(cap, random_state=42)
    print(f"[filter] capped pseudo to {cap}")
combined = pd.concat([train_df, pseudo_df], ignore_index=True)
combined.to_csv(OUT_CSV, index=False)
print(f"[filter] combined CSV: {len(combined)} rows  -> {OUT_CSV}")
PYEOF

echo "════ Step (c): Retrain Qwen-7B + TrOCR-hw in parallel (padded crops) ════"
echo "GPU 0: Qwen-7B R4 (padded crops, higher pseudo loss_weight=0.8)"
ROCR_VISIBLE_DEVICES=0 python -m src.train_vlm \
  --config configs/qwen25vl_lora.yaml \
  --fold 0 \
  --set train.pseudo_csv=data/derived_checkpoint/train_and_pseudo_r4.csv \
        output.checkpoint_dir=runs_checkpoint/qwen25vl_lora_checkpoints_r4 \
        output.dir=runs_checkpoint/qwen25vl_lora_r4 &
A=$!

echo "GPU 1: TrOCR-hw R4 (padded crops, higher pseudo loss_weight=0.8)"
ROCR_VISIBLE_DEVICES=1 python -m src.train_trocr \
  --config configs/trocr_large.yaml \
  --fold 0 \
  --set train.pseudo_csv=data/derived_checkpoint/train_and_pseudo_r4.csv \
        output.checkpoint_dir=runs_checkpoint/trocr_hw_checkpoints_r4 \
        output.dir=runs_checkpoint/trocr_hw_r4 &
B=$!

wait $A; RA=$?
wait $B; RB=$?
echo "exit codes: qwen=$RA  trocr=$RB"

report_disk
