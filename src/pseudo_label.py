import pandas as pd
import numpy as np
import re
import os

def check_repetition(text):
    # Matches sequences like "and and and" or character stutters
    words = text.split()
    for i in range(len(words) - 3):
        if words[i] == words[i+1] == words[i+2] == words[i+3]:
            return True
    return False

print("=== Generating Round 1 Pseudo-Labels ===")

vlm = pd.read_csv("preds_checkpoint/vlm_test.csv").fillna("")
trocr = pd.read_csv("preds_checkpoint/trocr_test.csv").fillna("")
ctc = pd.read_csv("preds_checkpoint/ctc_test.csv").fillna("")

# Ensure alignment
vlm = vlm.set_index("ID")
trocr = trocr.set_index("ID")
ctc = ctc.set_index("ID")

merged = vlm.join(trocr, lsuffix="_vlm", rsuffix="_trocr").join(ctc)
merged = merged.rename(columns={"Target": "Target_ctc"})

accepted = []
rejected_hallucination = 0
rejected_disagreement = 0

for idx, row in merged.iterrows():
    tv = row["Target_vlm"].strip()
    tt = row["Target_trocr"].strip()
    tc = row["Target_ctc"].strip()
    
    # 1. Hallucination Filters
    if check_repetition(tv):
        rejected_hallucination += 1
        continue
        
    # 2. Length disagreement filter (VLM vs CTC)
    # If VLM is vastly longer than CTC, it probably hallucinated off the end of the crop.
    if len(tv) > len(tc) + 15 and len(tc) > 0:
        rejected_hallucination += 1
        continue

    # 3. Agreement logic (Simple voting for Round 1 since we lack KenLM currently)
    # We take VLM as the primary since it scored 0.878, but we want it verified.
    # If VLM == TrOCR or VLM == CTC, we keep it. 
    # Or if levenshtein distance is very small (proxy for high agreement).
    from Levenshtein import distance
    d_v_t = distance(tv, tt)
    d_v_c = distance(tv, tc)
    
    max_len = max(len(tv), 1)
    if d_v_t / max_len < 0.15 or d_v_c / max_len < 0.15:
        # High agreement -> Accept
        accepted.append({"ID": idx, "Target": tv, "source": "pseudo_r1", "fold": -1})
    else:
        rejected_disagreement += 1

print(f"Total Test lines: {len(merged)}")
print(f"Accepted: {len(accepted)} ({len(accepted)/len(merged)*100:.1f}%)")
print(f"Rejected (Hallucination): {rejected_hallucination}")
print(f"Rejected (Disagreement): {rejected_disagreement}")

# Combine with train set
train = pd.read_csv("data/derived_checkpoint/train_folds.csv")
train["source"] = "train"
pseudo_df = pd.DataFrame(accepted)

combined = pd.concat([train, pseudo_df], ignore_index=True)
out_path = "data/derived_checkpoint/train_and_pseudo_r1.csv"
combined.to_csv(out_path, index=False)
print(f"Wrote {out_path} with {len(combined)} rows.")

def assert_no_manual_overrides(directory: str):
    import os
    if os.path.exists(os.path.join(directory, "manual_labels.csv")):
        raise RuntimeError("Manual labels are not permitted.")
