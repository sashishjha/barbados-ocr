import sys
import pandas as pd
from tqdm import tqdm
from confusion_network import confusion_network_combine

def load_beams(df, model_name):
    # groups by ID and creates a list of (text, score)
    # returns dict {ID: [ (text, score), ... ] }
    res = {}
    for iid, group in df.groupby("ID"):
        beams = []
        for _, row in group.iterrows():
            text = str(row['Target']) if pd.notna(row['Target']) else ""
            score = float(row['score'])
            beams.append((text, score))
        res[iid] = beams
    return res

def main():
    vlm_csv = "preds_checkpoint/vlm_test_nbest_r5_ensemble.csv"
    trocr_csv = "preds_checkpoint/trocr_test_nbest_r5_ensemble.csv"
    out_csv = "submissions/sub_20260925_confusion_network_r5.csv"
    
    print("Loading test predictions...")
    vlm = pd.read_csv(vlm_csv)
    trocr = pd.read_csv(trocr_csv)
    
    vlm_beams = load_beams(vlm, "qwen")
    trocr_beams = load_beams(trocr, "trocr")
    
    all_ids = set(vlm_beams.keys()).union(set(trocr_beams.keys()))
    
    results = []
    print("Running Confusion Network combination...")
    for iid in tqdm(all_ids):
        model_beams = {}
        if iid in vlm_beams:
            model_beams['qwen'] = vlm_beams[iid]
        if iid in trocr_beams:
            model_beams['trocr'] = trocr_beams[iid]
            
        # Let's use max_per_model=5 so we use all available beams since our alignment can handle it
        # Cap at 2 candidates per model so one model cannot dominate a cluster.
        # We can also add model_trust. Let's just trust them equally for now.
        best_text = confusion_network_combine(model_beams, max_per_model=2, mode="mbr")
        
        results.append({"ID": iid, "Target": best_text})
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {out_csv}")

if __name__ == "__main__":
    main()
