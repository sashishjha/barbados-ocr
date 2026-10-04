"""
Length-weighted Character Error Rate, reimplemented to match the organizers'
`compute_weighted_cer(gt_csv, pred_csv, weight_factor=...)` call signature.

RECONSTRUCTION NOTE (read before trusting any number this produces)
-------------------------------------------------------------------
`Starters.zip` ships `eval_metrics.py`, which does:

    from cer import compute_weighted_cer
    cer_result = compute_weighted_cer(gt_csv, pred_csv, weight_factor=weight_factor)
    ...
    final_score = cer_weight * cer_result["score"] + wer_weight * wer_result["score"]
    # with cer_weight=0.7, wer_weight=0.3 hardcoded as defaults

but the module implementing `compute_weighted_cer` lives in the organizers'
private `evaluations/` folder and was NOT included. Two things are therefore
reconstructed here, and both are configurable:

1. WEIGHTING SCHEME. The rules say longer reference transcriptions carry more
   weight. We implement per-sample weight `w_i = len(ref_i) ** weight_factor`,
   with the default `weight_factor=0.5` from the starter configs, and aggregate
   as a weighted mean of per-sample error rates. Alternative schemes
   ("micro": global sum(edits)/sum(len), and "uniform") are selectable via
   `scheme=` so the harness can be re-pointed if organizers publish the code.

2. SCORE POLARITY. `eval_metrics.py` calls the returned value `score` and the
   competition leaderboard's top sits around 0.91. An error rate of 0.91 would
   be near-worthless, so `score` is almost certainly an accuracy-like quantity,
   i.e. `1 - error_rate`, HIGHER IS BETTER. We return both:
       result["error"] -> weighted CER (lower better)
       result["score"] -> 1 - error, clipped to [0, 1] (higher better)
   `final_score` in our harness uses `score`, matching the starter code.

Missing / empty predictions are scored as fully wrong (error = 1.0 for that
sample) rather than skipped, matching the stated competition rule that every
test ID must receive a non-empty prediction.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

try:  # allow both `python src/eval/cer.py` and package import
    from .edit_distance import levenshtein
    from .normalize import get_normalizer
except ImportError:  # pragma: no cover
    from edit_distance import levenshtein
    from normalize import get_normalizer


ID_CANDIDATES = ("ID", "Id", "id", "new_id", "new id", "trapp_id")
TEXT_CANDIDATES = ("Target", "target", "text", "label", "transcription")


def _pick_column(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


def load_pairs(gt_csv_path: str, pred_csv_path: str):
    """Return (ids, refs, hyps, total_reference) inner-joined on ID."""
    gt = pd.read_csv(gt_csv_path, dtype=str, keep_default_na=False)
    pr = pd.read_csv(pred_csv_path, dtype=str, keep_default_na=False)

    gt_id, gt_txt = _pick_column(gt, ID_CANDIDATES), _pick_column(gt, TEXT_CANDIDATES)
    pr_id, pr_txt = _pick_column(pr, ID_CANDIDATES), _pick_column(pr, TEXT_CANDIDATES)

    gt = gt[[gt_id, gt_txt]].rename(columns={gt_id: "ID", gt_txt: "ref"})
    pr = pr[[pr_id, pr_txt]].rename(columns={pr_id: "ID", pr_txt: "hyp"})
    pr = pr.drop_duplicates(subset="ID", keep="last")

    merged = gt.merge(pr, on="ID", how="left")
    merged["hyp"] = merged["hyp"].fillna("")
    return merged, len(gt)


def compute_weighted_cer(
    gt_csv_path: str,
    pred_csv_path: str,
    weight_factor: float = 0.5,
    normalization: str = "raw",
    scheme: str = "length_pow",
    per_sample_out: Optional[str] = None,
) -> Dict:
    """Weighted CER. See module docstring for reconstruction caveats."""
    merged, total_reference = load_pairs(gt_csv_path, pred_csv_path)
    norm = get_normalizer(normalization)

    rows = []
    matched = 0
    for _, r in merged.iterrows():
        ref = norm(r["ref"])
        hyp = norm(r["hyp"])
        if len(ref) == 0:
            continue  # a reference with no characters carries no CER signal
        matched += 1
        dist = levenshtein(list(ref), list(hyp))
        err = min(1.0, dist / len(ref))
        rows.append(
            {
                "ID": r["ID"],
                "ref_len": len(ref),
                "hyp_len": len(hyp),
                "edits": dist,
                "cer": err,
                "weight": float(len(ref)) ** float(weight_factor),
            }
        )

    if matched == 0:
        return {"score": 0.0, "error": 1.0, "matched": 0, "total_reference": total_reference}

    df = pd.DataFrame(rows)
    if per_sample_out:
        df.to_csv(per_sample_out, index=False)

    if scheme == "length_pow":
        error = float((df["cer"] * df["weight"]).sum() / df["weight"].sum())
    elif scheme == "micro":
        error = float(df["edits"].sum() / df["ref_len"].sum())
    elif scheme == "uniform":
        error = float(df["cer"].mean())
    else:
        raise ValueError(f"Unknown scheme '{scheme}'")

    error = max(0.0, min(1.0, error))
    return {
        "score": 1.0 - error,
        "error": error,
        "matched": matched,
        "total_reference": total_reference,
        "per_sample": df,
    }
