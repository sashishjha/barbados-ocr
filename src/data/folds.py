"""
Validation design: stratified K-fold on (transcription-length bucket x
multi-line flag).

Why these strata:
  * The metric is length-weighted, so the long-line population must be
    represented identically in every fold or OOF comparisons are noise.
  * The multi-line label-convention inconsistency (known issue #3) is a
    per-image property; if it clusters into one fold, model selection breaks.

Also builds an OOF reference CSV (`oof_ref.csv`) in exactly the ID,Target shape
the eval harness expects, so `eval_metrics.py` runs unchanged on OOF preds.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

# Length buckets chosen from the verified corpus stats given in the reference
# bundle (min 37, mean 62.3, median 62, max 120 chars). Roughly quartile-ish,
# with the top bucket isolating the long lines that dominate the metric.
LENGTH_BINS: List[int] = [0, 50, 62, 75, 10_000]
LENGTH_LABELS = ["short", "mid_low", "mid_high", "long"]


def add_strata(df: pd.DataFrame, census_csv: str | None = None) -> pd.DataFrame:
    df = df.copy()
    df["Target"] = df["Target"].fillna("")
    df["label_len"] = df["Target"].str.len()
    df["len_bucket"] = pd.cut(
        df["label_len"], bins=LENGTH_BINS, labels=LENGTH_LABELS, right=False
    ).astype(str)

    if census_csv and os.path.exists(census_csv):
        c = pd.read_csv(census_csv, dtype={"ID": str})
        keep = [c_ for c_ in ("ID", "is_multiline", "n_lines", "label_convention")
                if c_ in c.columns]
        df = df.merge(c[keep], on="ID", how="left")
    if "is_multiline" not in df.columns:
        df["is_multiline"] = False
    df["is_multiline"] = df["is_multiline"].fillna(False).astype(bool)

    df["stratum"] = df["len_bucket"] + "|" + df["is_multiline"].map({True: "ml", False: "sl"})
    return df


def make_folds(
    train_csv: str,
    out_csv: str,
    oof_ref_csv: str,
    census_csv: str | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold

    df = pd.read_csv(train_csv, dtype={"ID": str})
    df = add_strata(df, census_csv)

    # Collapse strata with too few members so StratifiedKFold does not error.
    counts = df["stratum"].value_counts()
    rare = counts[counts < n_splits].index
    df.loc[df["stratum"].isin(rare), "stratum"] = "rare"

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1
    for k, (_, va) in enumerate(skf.split(df, df["stratum"])):
        df.loc[df.index[va], "fold"] = k
    assert (df["fold"] >= 0).all(), "unassigned rows in fold split"

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    df[["ID", "Target"]].to_csv(oof_ref_csv, index=False)

    print(f"[FOLDS] {len(df)} rows, {n_splits} folds, seed={seed} -> {out_csv}")
    print(f"[FOLDS] OOF reference (ID,Target) -> {oof_ref_csv}")
    piv = pd.crosstab(df["fold"], df["stratum"])
    print("[FOLDS] stratum distribution per fold:")
    print(piv.to_string())
    print("\n[FOLDS] label-length summary per fold (long lines drive the metric):")
    print(df.groupby("fold")["label_len"].describe()[["count", "mean", "50%", "max"]].to_string())
    return df


def long_line_weights(df: pd.DataFrame, weight_factor: float = 0.5,
                      power: float = 1.0) -> np.ndarray:
    """Sampling weights that over-sample long lines, matching the metric's
    length weighting. `power` scales the aggressiveness (curriculum: ramp from
    0.0 in early epochs to `power` in later epochs)."""
    w = np.power(df["label_len"].astype(float).clip(lower=1.0), weight_factor * power)
    return w / w.sum()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/derived_checkpoint/train_clean.csv")
    ap.add_argument("--census_csv", default="data/derived_checkpoint/multiline_census.csv")
    ap.add_argument("--out_csv", default="data/derived_checkpoint/train_folds.csv")
    ap.add_argument("--oof_ref_csv", default="data/derived_checkpoint/oof_ref.csv")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    make_folds(a.train_csv, a.out_csv, a.oof_ref_csv, a.census_csv, a.n_splits, a.seed)
