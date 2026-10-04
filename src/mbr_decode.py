import os
import sys
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import Levenshtein

def compute_cer(ref, hyp):
    if len(ref) == 0:
        return float(len(hyp) > 0)
    return Levenshtein.distance(ref, hyp) / len(ref)

def compute_wer(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()
    if len(ref_words) == 0:
        return float(len(hyp_words) > 0)
    return Levenshtein.distance(ref_words, hyp_words) / len(ref_words)

def blended_distance(ref, hyp):
    return 0.7 * compute_cer(ref, hyp) + 0.3 * compute_wer(ref, hyp)

def calibrate_scores(df):
    """
    Z-score normalize the log probabilities per model arm so they are directly comparable.
    Then applies a softmax with temperature T.
    """
    df = df.copy()
    arms = df['arm'].unique()
    for arm in arms:
        mask = df['arm'] == arm
        scores = df.loc[mask, 'score']
        mu = scores.mean()
        sigma = scores.std()
        if pd.isna(sigma) or sigma == 0:
            sigma = 1.0
        # Z-score
        df.loc[mask, 'calibrated_score'] = (scores - mu) / sigma
    
    return df

def process_group(args):
    iid, group_df = args
    # Group contains hypotheses from both arms.
    # Softmax the calibrated scores to get probabilities
    T = 1.0  # Temperature
    scores = group_df['calibrated_score'].values / T
    probs = np.exp(scores - np.max(scores))
    probs = probs / np.sum(probs)
    
    hypotheses = group_df['Target'].astype(str).values
    
    best_hyp = None
    min_risk = float('inf')
    
    for i, hyp_i in enumerate(hypotheses):
        risk_i = 0.0
        for j, hyp_j in enumerate(hypotheses):
            if i != j:
                risk_i += probs[j] * blended_distance(hyp_j, hyp_i) # True is hyp_j, Prediction is hyp_i
        
        if risk_i < min_risk:
            min_risk = risk_i
            best_hyp = hyp_i
            
    return {"ID": iid, "Target": best_hyp}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()
    
    print(f"[mbr] Loading {len(args.candidates)} candidate files...")
    dfs = []
    for f in args.candidates:
        df = pd.read_csv(f)
        if "arm" not in df.columns:
            if "model" in df.columns:
                df = df.rename(columns={"model": "arm"})
            else:
                df["arm"] = os.path.basename(f)
        dfs.append(df)
        
    merged = pd.concat(dfs, ignore_index=True)
    
    # Drop NaNs
    merged = merged.dropna(subset=['Target'])
    
    print("[mbr] Calibrating scores across models...")
    merged = calibrate_scores(merged)
    
    print("[mbr] Grouping by ID...")
    groups = list(merged.groupby("ID"))
    
    print(f"[mbr] Running MBR decoding on {len(groups)} images using {args.n_jobs} workers...")
    
    results = []
    with Pool(args.n_jobs) as p:
        for res in tqdm(p.imap_unordered(process_group, groups), total=len(groups)):
            results.append(res)
            
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)
    print(f"[mbr] Wrote {len(out_df)} rows to {args.out}")

if __name__ == "__main__":
    main()
