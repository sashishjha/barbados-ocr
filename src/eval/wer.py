"""
Length-weighted Word Error Rate. Mirrors cer.py exactly; see that module's
docstring for the full reconstruction caveats (the organizers' `evaluations/`
package was not shipped in Starters.zip).

Tokenization: whitespace split after normalization. Per-sample weight uses the
CHARACTER length of the reference (`len(ref)`), not the word count, because the
competition text says "longer reference transcriptions carry more weight" and
the same `weight_factor` value is passed to both the WER and CER functions by
`eval_metrics.py`. Switch to word-count weighting with `weight_on="words"` if
organizers clarify otherwise.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

try:
    from .edit_distance import levenshtein
    from .normalize import get_normalizer
    from .cer import load_pairs
except ImportError:  # pragma: no cover
    from edit_distance import levenshtein
    from normalize import get_normalizer
    from cer import load_pairs


def compute_weighted_wer(
    gt_csv_path: str,
    pred_csv_path: str,
    weight_factor: float = 0.5,
    normalization: str = "raw",
    scheme: str = "length_pow",
    weight_on: str = "chars",
    per_sample_out: Optional[str] = None,
) -> Dict:
    merged, total_reference = load_pairs(gt_csv_path, pred_csv_path)
    norm = get_normalizer(normalization)

    rows = []
    matched = 0
    for _, r in merged.iterrows():
        ref_s = norm(r["ref"])
        hyp_s = norm(r["hyp"])
        ref = ref_s.split()
        hyp = hyp_s.split()
        if len(ref) == 0:
            continue
        matched += 1
        dist = levenshtein(ref, hyp)
        err = min(1.0, dist / len(ref))
        base = len(ref_s) if weight_on == "chars" else len(ref)
        rows.append(
            {
                "ID": r["ID"],
                "ref_words": len(ref),
                "hyp_words": len(hyp),
                "edits": dist,
                "wer": err,
                "weight": float(base) ** float(weight_factor),
            }
        )

    if matched == 0:
        return {"score": 0.0, "error": 1.0, "matched": 0, "total_reference": total_reference}

    df = pd.DataFrame(rows)
    if per_sample_out:
        df.to_csv(per_sample_out, index=False)

    if scheme == "length_pow":
        error = float((df["wer"] * df["weight"]).sum() / df["weight"].sum())
    elif scheme == "micro":
        error = float(df["edits"].sum() / df["ref_words"].sum())
    elif scheme == "uniform":
        error = float(df["wer"].mean())
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
