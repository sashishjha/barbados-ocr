"""
Multi-line image detection and the label-convention census (known issue #3).

The organizers acknowledged an UNRESOLVED inconsistency: most multi-line images
are labelled with only the centred/main line's text, but a subset carries the
full multi-line transcription. No ruling has been issued.

What this module does (all automated, no eyeballing):
  1. `count_text_lines()` - horizontal projection profile on a binarized,
     deskewed crop; count contiguous ink bands separated by valleys.
  2. `census()` - for every training row, record (n_detected_lines, label_len,
     label_len / n_lines). Rows where the label length per detected line is
     close to the single-line population mean are consistent with the
     CENTRE-ONLY convention; rows whose label length scales with the number of
     detected lines are consistent with the FULL convention.
  3. The output census CSV drives two things downstream:
     - a `multiline` stratification key for K-fold (so folds are comparable);
     - an optional `label_convention` feature for label-noise-tolerant training
       (see configs: train.loss.label_noise).

We deliberately do NOT delete or rewrite any of these labels. The majority
convention is measured, and training is made robust to the minority
(loss clamping / sample down-weighting), which is defensible under either
eventual organizer ruling.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold -> boolean ink mask (True = ink)."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    total = gray.size
    sum_all = float(np.dot(np.arange(256), hist))
    sumB = wB = 0.0
    best, thr = 0.0, 0
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        between = wB * wF * ((sumB / wB) - ((sum_all - sumB) / wF)) ** 2
        if between >= best:
            best, thr = between, i
    return gray <= thr


def deskew_angle(ink: np.ndarray, max_deg: float = 6.0, step: float = 0.5) -> float:
    """Estimate skew by maximizing the variance of the horizontal projection."""
    from scipy.ndimage import rotate  # scipy is in requirements

    best_a, best_v = 0.0, -1.0
    for a in np.arange(-max_deg, max_deg + 1e-9, step):
        r = rotate(ink.astype(np.float32), a, reshape=False, order=0)
        v = float(r.sum(axis=1).var())
        if v > best_v:
            best_v, best_a = v, float(a)
    return best_a


def count_text_lines(
    path: str,
    min_band_frac: float = 0.04,
    valley_frac: float = 0.15,
    do_deskew: bool = True,
) -> Dict:
    """Count ink bands via horizontal projection profile.

    Returns n_lines, band heights, and the centre-band ink share (useful for the
    centre-only convention hypothesis).
    """
    from PIL import Image

    if not os.path.exists(path):
        return {"n_lines": 0, "bands": [], "centre_share": np.nan,
                "height": 0, "width": 0, "skew": 0.0}

    with Image.open(path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.uint8)

    ink = _binarize(gray)
    skew = 0.0
    if do_deskew:
        try:
            skew = deskew_angle(ink)
            if abs(skew) > 1e-6:
                from scipy.ndimage import rotate

                ink = rotate(ink.astype(np.float32), skew, reshape=False, order=0) > 0.5
        except Exception:
            skew = 0.0

    proj = ink.sum(axis=1).astype(np.float32)
    if proj.max() <= 0:
        return {"n_lines": 0, "bands": [], "centre_share": np.nan,
                "height": int(gray.shape[0]), "width": int(gray.shape[1]), "skew": skew}

    # Smooth then threshold at a fraction of the peak to find bands.
    k = max(1, int(round(gray.shape[0] * 0.01)))
    kernel = np.ones(k, dtype=np.float32) / k
    smooth = np.convolve(proj, kernel, mode="same")
    thr = valley_frac * smooth.max()
    above = smooth > thr

    bands: List[tuple] = []
    start = None
    for i, v in enumerate(above):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, len(above)))

    min_h = max(2, int(min_band_frac * gray.shape[0]))
    bands = [(a, b) for a, b in bands if (b - a) >= min_h]

    centre_share = np.nan
    if bands:
        mid = gray.shape[0] / 2.0
        ink_per_band = [float(proj[a:b].sum()) for a, b in bands]
        c_idx = int(np.argmin([abs(((a + b) / 2.0) - mid) for a, b in bands]))
        tot = sum(ink_per_band)
        centre_share = ink_per_band[c_idx] / tot if tot > 0 else np.nan

    return {
        "n_lines": len(bands),
        "bands": bands,
        "centre_share": centre_share,
        "height": int(gray.shape[0]),
        "width": int(gray.shape[1]),
        "skew": skew,
    }


def census(train_csv: str, image_dir: str, out_csv: str) -> pd.DataFrame:
    """Build the multi-line / label-convention census over the training set."""
    df = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
    df["label_len"] = df["Target"].fillna("").str.len()

    recs = []
    for i, rid in enumerate(df["ID"]):
        r = count_text_lines(os.path.join(image_dir, f"{rid}.jpg"))
        recs.append({k: r[k] for k in ("n_lines", "centre_share", "height", "width", "skew")})
        if (i + 1) % 250 == 0:
            print(f"[MULTILINE] {i+1}/{len(df)}")
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)

    df["is_multiline"] = df["n_lines"] >= 2
    df["aspect"] = df["width"] / df["height"].replace(0, np.nan)

    # Reference: mean label length of confidently single-line images.
    single = df.loc[df["n_lines"] == 1, "label_len"]
    ref_len = float(single.mean()) if len(single) else float(df["label_len"].mean())
    df["expected_full_len"] = ref_len * df["n_lines"].clip(lower=1)
    df["len_ratio_full"] = df["label_len"] / df["expected_full_len"]
    df["len_ratio_centre"] = df["label_len"] / max(ref_len, 1e-6)

    # Convention inference for multi-line rows only. A label close to ONE line's
    # worth of text -> centre-only; close to n_lines' worth -> full.
    def infer(row):
        if not row["is_multiline"]:
            return "single_line"
        d_centre = abs(row["len_ratio_centre"] - 1.0)
        d_full = abs(row["len_ratio_full"] - 1.0)
        if min(d_centre, d_full) > 0.6:
            return "ambiguous"
        return "centre_only" if d_centre <= d_full else "full_multiline"

    df["label_convention"] = df.apply(infer, axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("\n[MULTILINE CENSUS]")
    print(f"  single-line reference label length : {ref_len:.1f} chars")
    print(f"  multi-line images detected         : {int(df['is_multiline'].sum())} "
          f"/ {len(df)} ({100*df['is_multiline'].mean():.1f}%)")
    print("  label convention among multi-line rows:")
    vc = df.loc[df["is_multiline"], "label_convention"].value_counts()
    for k, v in vc.items():
        print(f"    {k:<16} {v:>5}  ({100*v/max(1,int(df['is_multiline'].sum())):.1f}%)")
    print(f"  -> census written to {out_csv}")
    print("  NOTE: the MAJORITY convention above is what training should be biased")
    print("        toward; the minority is handled by label-noise-tolerant loss.")
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/derived_checkpoint/train_clean.csv")
    ap.add_argument("--image_dir", default="data/images")
    ap.add_argument("--out_csv", default="data/derived_checkpoint/multiline_census.csv")
    a = ap.parse_args()
    census(a.train_csv, a.image_dir, a.out_csv)
