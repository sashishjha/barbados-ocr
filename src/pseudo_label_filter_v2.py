"""
pseudo_label_filter_v2.py

Fixes relative to the original R4/R5 filtering script, in order of how much
they likely mattered for the R5 plateau:

1. STRUCTURAL FIX (the big one): the original filter was a hard accept/reject
   gate -- any disagreement, beam instability, length outlier, or OOV char
   dropped the example entirely, and every SURVIVING example used the VLM's
   raw top-1 as its target. That selects almost exactly for "cases the model
   already gets right and agrees with TrOCR on" -- which is why adding this
   data in R5 added ~no new signal (on top of the loss-weight mixup you
   already found). This version:
     - keeps every example, but assigns it a continuous `pseudo_weight` in
       (0, 1] instead of a boolean,
     - and, whenever both models produced a prediction, uses the
       ROVER-recombined text as the TARGET instead of the VLM's raw guess.
   That means the hard, disagreement-heavy tail (the informative part your
   old filter discarded) now contributes real signal -- the ensemble's best
   guess, at a reduced weight -- instead of nothing.
   Repetition loops and extreme OOV are still hard-excluded (weight=0):
   those are genuinely corrupted output, not just "hard examples".

2. BUG FIX: `row.get("vlm_top2", pred)` / `row.get("trocr_top1", "")` do NOT
   fall back the way you'd expect. pandas Series.get() only uses the default
   when the KEY is missing, not when the stored VALUE is NaN -- and after a
   left-merge, missing beams show up as NaN under an existing column, not as
   a missing key. So a missing top-2 beam or missing TrOCR prediction became
   the literal string "nan", which then produced a large edit distance
   against nearly anything and got the example wrongly flagged as
   beam_unstable / disagree. Fixed with explicit pd.isna() checks below.

3. Length filter now uses configurable train-set PERCENTILES (default
   1st/99th) instead of raw min()/max(), so one outlier training line can no
   longer set the bound for the entire test set.

4. has_repetition's default now catches 3x+ consecutive repeats (was 5x+),
   which also catches shorter degenerate-decoding loops.

5. The `audit` list in the original script was created but never populated
   or written -- every row's reasons are now recorded and saved to AUDIT_CSV.

Usage: adjust the CONFIG block below to your paths, then run as a script, or
import `score_example` / `build_reference_stats` directly for testing.
"""
from __future__ import annotations

import math
import unicodedata
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rover import rover_combine

# ------------------------------- CONFIG -------------------------------
VLM_NBEST = "preds_checkpoint/vlm_test_nbest_r4.csv"
TROCR_NBEST = "preds_checkpoint/trocr_test_nbest_r4.csv"
TRAIN_CSV = "data/Train.csv"
OUT_CSV = "data/derived_checkpoint/train_and_pseudo_r5.csv"
AUDIT_CSV = "report/pseudo_round5_audit.csv"

LENGTH_PERCENTILES = (1, 99)   # was raw min()/max()
OOV_THRESH = 0.04              # same as original
OOV_HARD_MULTIPLE = 3.0        # OOV ratio beyond this multiple -> hard reject
BEAM_UNSTABLE_FRAC = 0.10      # same as original
DISAGREE_FRAC = 0.10           # same as original
DECAY_K = 3.0                  # controls how fast soft weight decays with severity
REPETITION_MIN_REPEATS = 3     # was implicitly 5
# ------------------------------------------------------------------------


def edit_dist(a: str, b: str) -> int:
    a, b = str(a), str(b)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def has_repetition(text: str, min_repeats: int = REPETITION_MIN_REPEATS,
                    min_unit: int = 3, max_unit: int = 14) -> bool:
    for n in range(min_unit, max_unit + 1):
        span = n * min_repeats
        if len(text) < span:
            continue
        for i in range(len(text) - span + 1):
            sub = text[i:i + n]
            if text[i:].startswith(sub * min_repeats):
                return True
    return False


def oov_ratio(text: str, vocab: set) -> float:
    chars = [c for c in text if not unicodedata.category(c).startswith("Z")]
    if not chars:
        return 1.0
    return sum(1 for c in chars if c not in vocab) / len(chars)


def build_reference_stats(train_texts: List[str], length_percentiles=LENGTH_PERCENTILES) -> Dict:
    lengths = [len(t) for t in train_texts]
    lo, hi = np.percentile(lengths, length_percentiles)
    return {
        "min_len": lo,
        "max_len": hi,
        "vocab": set("".join(train_texts)),
    }


def _safe_str_or_none(value) -> Optional[str]:
    """Correct replacement for `row.get(col, default)`: returns None if the
    value is missing/NaN, otherwise the string value. Callers decide the
    actual fallback behaviour explicitly (see score_example) instead of
    silently getting the literal string "nan"."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return str(value)


def score_example(
    vlm_top1,
    vlm_top2,
    trocr_top1,
    ref_stats: Dict,
    oov_thresh: float = OOV_THRESH,
    oov_hard_multiple: float = OOV_HARD_MULTIPLE,
    beam_unstable_frac: float = BEAM_UNSTABLE_FRAC,
    disagree_frac: float = DISAGREE_FRAC,
    decay_k: float = DECAY_K,
) -> Dict:
    """Score one candidate pseudo-label. Returns:
        target       -- text to use as the pseudo-label
        weight       -- continuous weight in [0, 1]; 0 means hard-excluded
        hard_reject  -- True only for repetition / extreme OOV
        reasons      -- audit trail (always populated)
    """
    pred = str(vlm_top1)
    pred2 = _safe_str_or_none(vlm_top2)
    trocr_p = _safe_str_or_none(trocr_top1)

    reasons: List[str] = []
    weight = 1.0
    hard_reject = False

    if has_repetition(pred):
        reasons.append("repetition")
        hard_reject = True

    if len(pred) < ref_stats["min_len"] or len(pred) > ref_stats["max_len"]:
        reasons.append(f"length({len(pred)})")
        weight *= 0.5  # soft penalty now -- percentile bounds are already conservative

    ov = oov_ratio(pred, ref_stats["vocab"])
    if ov > oov_thresh:
        reasons.append(f"oov({ov:.3f})")
        if ov > oov_hard_multiple * oov_thresh:
            hard_reject = True
        else:
            weight *= math.exp(-decay_k * ov)

    if pred2 is not None:
        ed = edit_dist(pred, pred2)
        frac = ed / max(len(pred), 1)
        if frac > beam_unstable_frac:
            reasons.append(f"beam_unstable({frac:.2f})")
            weight *= math.exp(-decay_k * frac)
    else:
        reasons.append("no_top2_beam")  # informational only, not penalized:
        # a single confident beam with nothing to compare against is NOT
        # evidence of instability (the original bug treated it as such).

    target = pred
    if trocr_p is not None:
        ed = edit_dist(pred, trocr_p)
        frac = ed / max(len(pred), 1)
        if frac > disagree_frac:
            reasons.append(f"disagree({frac:.2f})")
            weight *= math.exp(-decay_k * frac)
            # Key change: recombine instead of dropping. The target for a
            # disagreement case is the ensemble's best guess (ROVER), kept
            # at reduced weight -- not the VLM's raw, possibly-wrong guess,
            # and not nothing.
            target = rover_combine([pred, trocr_p], confidences=[1.0, 1.0])
    else:
        reasons.append("no_trocr_pred")
        weight *= 0.7  # one signal source missing; mild penalty, not a hard drop

    if hard_reject:
        weight = 0.0

    return {
        "target": target,
        "weight": round(weight, 4),
        "hard_reject": hard_reject,
        "reasons": reasons,
    }


def run(vlm_nbest_csv=VLM_NBEST, trocr_nbest_csv=TROCR_NBEST, train_csv=TRAIN_CSV,
        out_csv=OUT_CSV, audit_csv=AUDIT_CSV, min_weight_to_keep: float = 0.05):
    train = pd.read_csv(train_csv)
    train_texts = train["Target"].dropna().astype(str).tolist()
    ref_stats = build_reference_stats(train_texts)

    vlm = pd.read_csv(vlm_nbest_csv)
    trocr = pd.read_csv(trocr_nbest_csv)
    vlm_top1 = vlm[vlm["rank"] == 0][["ID", "Target"]].rename(columns={"Target": "vlm_top1"})
    vlm_top2 = vlm[vlm["rank"] == 1][["ID", "Target"]].rename(columns={"Target": "vlm_top2"})
    trocr_top1 = trocr[trocr["rank"] == 0][["ID", "Target"]].rename(columns={"Target": "trocr_top1"})

    df = vlm_top1.merge(vlm_top2, on="ID", how="left").merge(trocr_top1, on="ID", how="left")

    rows, audit = [], []
    for _, row in df.iterrows():
        result = score_example(row["vlm_top1"], row.get("vlm_top2"), row.get("trocr_top1"), ref_stats)
        audit.append({"ID": row["ID"], **result, "reasons": ";".join(result["reasons"])})
        if result["weight"] >= min_weight_to_keep:
            rows.append({
                "ID": row["ID"],
                "Target": result["target"],
                "source": "pseudo_r5",
                "loss_weight": result["weight"],  # <-- per-example weight now, not a flat 0.8
            })

    pseudo_df = pd.DataFrame(rows)
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(audit_csv, index=False)
    pseudo_df.to_csv(out_csv, index=False)
    print(f"kept {len(pseudo_df)}/{len(df)} pseudo-labels "
          f"({(audit_df['weight'] == 0).sum()} hard-rejected, "
          f"{((audit_df['weight'] > 0) & (audit_df['weight'] < 1)).sum()} soft-down-weighted)")
    return pseudo_df, audit_df


if __name__ == "__main__":
    run()
