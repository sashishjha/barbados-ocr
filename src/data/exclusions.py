"""
Documented exclusion of corrupted training rows.

Rules context: the organizers ruled (discussion board) that excluding corrupted
training rows IS allowed provided it is documented. The original `Train.csv` is
never modified; this module produces a *derived* clean CSV plus an audit trail.

21 IDs were publicly flagged in the discussion thread. We do NOT trust that list
blindly: `verify_exclusions()` re-derives each flag programmatically
(blank/near-blank image, degenerate label, length outlier) and writes a report
saying, per ID, whether our automated checks agree. Any ID the checks cannot
confirm is still excluded (organizers flagged it) but is marked
`confirmed=False` in the audit CSV so the decision is visible to reviewers.

No human ever inspects an image and edits a label here. Everything is code.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

# Publicly flagged in the competition discussion board (empty images /
# label mismatches). Verbatim, unmodified.
FLAGGED_IDS: List[str] = [
    "79tMUVyfIdy3GzkG", "rh8o7bdCGOIBFPwH", "F8DYDDp2AvW9Dytw", "mfQxfOmeRmwBh0g8",
    "EcxuqKeZl7OQexfB", "PO7QQLWIFOT65BTz", "yNyf3Tp0zc7DFj5F", "JU7lRwk3jKkus24Z",
    "R6iYPb7MHFiHtXH6", "VmrEALeZiP1Y6nF9", "t0UrASljcgzvBAnO", "KH5g92Q3DA5Bo6xi",
    "N78M3v6GKmUzF7sk", "0CCrVKAom8EK53jj", "3ZxOeKcOr5wUYyk0", "PJbM7Q1SrblrSWt6",
    "WwlTCykxjP3c4kfo", "baTY3OlGskirWgFc", "u3b4JNo5bqpfE7Js", "MfT9S5oghk9ywNSC",
    "8H2ITJSWZhAD6eh0",
]


# --------------------------------------------------------------------------
# Programmatic checks
# --------------------------------------------------------------------------
def image_stats(path: str) -> Dict:
    """Cheap blank/near-blank detection: std of grayscale pixels and ink ratio."""
    from PIL import Image

    if not os.path.exists(path):
        return {"exists": False, "std": np.nan, "ink_ratio": np.nan,
                "width": np.nan, "height": np.nan}
    with Image.open(path) as im:
        g = np.asarray(im.convert("L"), dtype=np.float32)
    thr = max(0.0, g.mean() - 2.0 * g.std())
    return {
        "exists": True,
        "std": float(g.std()),
        "ink_ratio": float((g < thr).mean()) if g.size else 0.0,
        "width": int(g.shape[1]),
        "height": int(g.shape[0]),
    }


def verify_exclusions(
    train_csv: str,
    image_dir: str,
    blank_std_thresh: float = 6.0,
    blank_ink_thresh: float = 0.002,
    length_z_thresh: float = 4.0,
) -> pd.DataFrame:
    """Re-derive corruption evidence for every training row.

    Returns a per-row audit frame with boolean reason columns:
      missing_image, blank_image, empty_label, short_label, length_outlier,
      flagged_by_organizers, auto_exclude, confirmed
    """
    df = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
    df["Target"] = df["Target"].fillna("")
    df["label_len"] = df["Target"].str.len()

    lens = df["label_len"].astype(float)
    mu, sd = lens.mean(), max(lens.std(), 1e-6)
    df["length_z"] = (lens - mu) / sd

    stats = [image_stats(os.path.join(image_dir, f"{i}.jpg")) for i in df["ID"]]
    sdf = pd.DataFrame(stats)
    df = pd.concat([df.reset_index(drop=True), sdf], axis=1)

    df["missing_image"] = ~df["exists"].fillna(False)
    df["blank_image"] = (
        df["exists"].fillna(False)
        & ((df["std"] < blank_std_thresh) | (df["ink_ratio"] < blank_ink_thresh))
    )
    df["empty_label"] = df["Target"].str.strip() == ""
    df["short_label"] = df["label_len"] < 5
    df["length_outlier"] = df["length_z"].abs() > length_z_thresh
    df["flagged_by_organizers"] = df["ID"].isin(FLAGGED_IDS)

    df["auto_exclude"] = (
        df["missing_image"] | df["blank_image"] | df["empty_label"] | df["short_label"]
    )
    # An organizer-flagged ID is confirmed if at least one automated check fires.
    df["confirmed"] = df["flagged_by_organizers"] & (
        df["auto_exclude"] | df["length_outlier"]
    )
    return df


def build_clean_train(
    train_csv: str,
    image_dir: str,
    out_csv: str,
    audit_csv: str,
    use_flagged: bool = True,
    use_auto: bool = True,
) -> pd.DataFrame:
    """Write the derived clean training CSV plus the audit trail.

    Train.csv itself is opened read-only and never rewritten.
    """
    audit = verify_exclusions(train_csv, image_dir)

    drop = pd.Series(False, index=audit.index)
    if use_flagged:
        drop |= audit["flagged_by_organizers"]
    if use_auto:
        drop |= audit["auto_exclude"]
    audit["excluded"] = drop

    clean = audit.loc[~drop, ["ID", "Target"]].copy()

    for p in (out_csv, audit_csv):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    clean.to_csv(out_csv, index=False)
    audit.to_csv(audit_csv, index=False)

    n_flagged = int(audit["flagged_by_organizers"].sum())
    n_conf = int(audit["confirmed"].sum())
    print(f"[EXCLUSIONS] rows in Train.csv          : {len(audit)}")
    print(f"[EXCLUSIONS] organizer-flagged present  : {n_flagged}/{len(FLAGGED_IDS)}")
    print(f"[EXCLUSIONS]   ...auto-confirmed        : {n_conf}")
    print(f"[EXCLUSIONS] additional auto-detected   : "
          f"{int((audit['auto_exclude'] & ~audit['flagged_by_organizers']).sum())}")
    print(f"[EXCLUSIONS] total excluded             : {int(drop.sum())}")
    print(f"[EXCLUSIONS] clean rows written         : {len(clean)} -> {out_csv}")
    print(f"[EXCLUSIONS] audit trail                : {audit_csv}")
    return clean


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/Train.csv")
    ap.add_argument("--image_dir", default="data/images")
    ap.add_argument("--out_csv", default="data/derived_checkpoint/train_clean.csv")
    ap.add_argument("--audit_csv", default="report/exclusion_audit.csv")
    ap.add_argument("--no_auto", action="store_true",
                    help="Only drop the 21 organizer-flagged IDs")
    a = ap.parse_args()
    build_clean_train(a.train_csv, a.image_dir, a.out_csv, a.audit_csv,
                      use_auto=not a.no_auto)
