"""
local_cv_eval.py

Addresses the meta-point from the review: right now every ensembling /
pseudo-labeling idea costs a leaderboard submission to evaluate, which is
slow, limited, and risks overfitting to the public split (this is also
exactly the ambiguity around how the KenLM alpha weights were tuned).

Two entry points:

1. run_kfold_cv(...) -- full k-fold harness. You plug in your own
   train_fold_fn / predict_fold_fn (this script has no way to actually train
   your Qwen/TrOCR models); it handles fold splitting and scoring.

2. score_combiner_from_nbest(...) -- the more immediately useful one given
   where you are right now: if you already have n-best predictions from
   your trained models on a HELD-OUT slice of Train.csv (i.e. predictions
   from a checkpoint that slice wasn't trained on), you can directly compare
   plain-ROVER vs whole-sequence-MBR vs the confusion-network hybrid vs any
   other combine_fn on data with known ground truth, with zero retraining.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from text_metrics import competition_score


def run_kfold_cv(
    train_df: pd.DataFrame,
    k: int,
    train_fold_fn: Callable[[pd.DataFrame], object],
    predict_fold_fn: Callable[[object, pd.DataFrame], pd.DataFrame],
    seed: int = 13,
) -> Dict:
    """train_fold_fn(train_subset_df) -> model_bundle
       predict_fold_fn(model_bundle, val_subset_df) -> DataFrame[ID, Target]
    Returns per-fold scores, overall mean/std, and the concatenated
    per-example predictions/scores for error analysis."""
    idx = np.arange(len(train_df))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)

    fold_scores, all_rows = [], []
    for fold_i in range(k):
        val_idx = folds[fold_i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_i])
        tr_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)

        model = train_fold_fn(tr_df)
        preds = predict_fold_fn(model, val_df)

        merged = val_df[["ID", "Target"]].rename(columns={"Target": "ref"}).merge(
            preds.rename(columns={"Target": "hyp"})[["ID", "hyp"]], on="ID", how="left"
        )
        merged["hyp"] = merged["hyp"].fillna("")
        merged["score"] = [
            competition_score(str(r), str(h)) for r, h in zip(merged["ref"], merged["hyp"])
        ]
        fold_score = merged["score"].mean()
        fold_scores.append(fold_score)
        merged["fold"] = fold_i
        all_rows.append(merged)
        print(f"fold {fold_i + 1}/{k}: mean score (lower=better) = {fold_score:.4f}  n={len(merged)}")

    overall = pd.concat(all_rows, ignore_index=True)
    result = {
        "fold_scores": fold_scores,
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
        "predictions": overall,
    }
    print(f"\noverall CV score: {result['mean']:.4f} +/- {result['std']:.4f}")
    return result


def _beams_for(nbest_df: pd.DataFrame, id_, max_beams: int) -> List[Tuple[str, float]]:
    sub = nbest_df[nbest_df["ID"] == id_].sort_values("rank").head(max_beams)
    if "score" in sub.columns:
        return list(zip(sub["Target"].astype(str), sub["score"].astype(float)))
    # no confidence column available -> monotonically decreasing weight by rank
    return [(t, 1.0 / (r + 1)) for r, t in enumerate(sub["Target"].astype(str))]


def score_combiner_from_nbest(
    vlm_nbest_df: pd.DataFrame,
    trocr_nbest_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    combine_fn: Callable[[Dict[str, List[Tuple[str, float]]]], str],
    max_beams: int = 5,
) -> pd.DataFrame:
    """vlm_nbest_df / trocr_nbest_df: columns ID, rank, Target, optional score
       -- predictions on a HELD-OUT slice of Train.csv (not the fold those
       checkpoints trained on).
       ref_df: columns ID, Target -- ground truth for that same slice.
       combine_fn: e.g. `lambda b: confusion_network_combine(b, max_per_model=2)`
                   or  `lambda b: rover_combine([b['qwen'][0][0], b['trocr'][0][0]])`
    Prints and returns per-example + mean competition score so you can
    directly compare combiners without spending an LB submission."""
    rows = []
    for _, row in ref_df.iterrows():
        id_ = row["ID"]
        ref = str(row["Target"])
        model_beams = {
            "qwen": _beams_for(vlm_nbest_df, id_, max_beams),
            "trocr": _beams_for(trocr_nbest_df, id_, max_beams),
        }
        hyp = combine_fn(model_beams)
        rows.append({"ID": id_, "ref": ref, "hyp": hyp, "score": competition_score(ref, hyp)})
    out = pd.DataFrame(rows)
    print(f"mean competition score (lower=better): {out['score'].mean():.4f}  n={len(out)}")
    return out
