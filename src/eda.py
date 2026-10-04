"""
Scripted EDA. Everything is written to report/eda/ as CSV + PNG so the notebook
is a thin viewer and the analysis itself is reproducible from one command.

Covers the roadmap's EDA checklist:
  1. image size / aspect distributions
  2. transcription length distribution, with the long-line population isolated
     and each length bucket's SHARE OF THE METRIC WEIGHT computed
  3. character-set inventory (archaic chars, ligatures, punctuation, digits)
  4. multi-line census (delegated to src/data/multiline.py)
  5. duplicate / near-duplicate detection (exact text dupes, and image
     near-duplicates via a difference hash)
  6. corrupted-row verification (delegated to src/data/exclusions.py)

Usage:
    python -m src.eda --train_csv data/Train.csv --image_dir data/images \
        --test_csv data/Test.csv --out_dir report/eda
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd


def _save(df: pd.DataFrame, out_dir: str, name: str):
    p = os.path.join(out_dir, name)
    df.to_csv(p, index=False)
    print(f"  -> {p}")


def _plot(fn, out_dir: str, name: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = fn(plt)
        p = os.path.join(out_dir, name)
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {p}")
    except Exception as e:
        print(f"  [skip plot {name}: {e}]")


def dhash(path: str, hash_size: int = 8) -> int | None:
    """Difference hash for near-duplicate image detection."""
    from PIL import Image

    if not os.path.exists(path):
        return None
    with Image.open(path) as im:
        g = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def run(train_csv: str, image_dir: str, test_csv: str | None, out_dir: str,
        sample_images: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
    df["Target"] = df["Target"].fillna("")
    df["label_len"] = df["Target"].str.len()
    df["n_words"] = df["Target"].str.split().apply(len)

    print("\n=== 1. Corpus overview ===")
    print(f"train rows: {len(df)}")
    print(df["label_len"].describe().to_string())
    print(f"words per line: mean {df['n_words'].mean():.1f}, max {df['n_words'].max()}")

    # ---------------- 2. length distribution + metric weight share
    print("\n=== 2. Length distribution and metric weight share ===")
    bins = [0, 45, 55, 62, 70, 80, 10_000]
    labels = ["<45", "45-54", "55-61", "62-69", "70-79", "80+"]
    df["bucket"] = pd.cut(df["label_len"], bins=bins, labels=labels, right=False)
    # Metric weight w_i = len^0.5 (the reconstructed weight_factor=0.5 scheme).
    df["metric_weight"] = df["label_len"].astype(float).clip(lower=1) ** 0.5
    bstats = df.groupby("bucket", observed=True).agg(
        n=("ID", "count"), mean_len=("label_len", "mean"),
        weight_sum=("metric_weight", "sum")).reset_index()
    bstats["row_share"] = bstats["n"] / bstats["n"].sum()
    bstats["weight_share"] = bstats["weight_sum"] / bstats["weight_sum"].sum()
    print(bstats.to_string(index=False))
    print("NOTE: weight_share is what actually drives the leaderboard. Optimise "
          "the buckets with the largest weight_share, not the largest row_share.")
    _save(bstats, out_dir, "length_buckets.csv")
    _save(df[["ID", "label_len", "n_words", "bucket"]], out_dir, "label_lengths.csv")

    def _len_plot(plt):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["label_len"], bins=40)
        ax.axvline(df["label_len"].median(), ls="--", label=f"median {df['label_len'].median():.0f}")
        ax.set_xlabel("transcription length (chars)"); ax.set_ylabel("count")
        ax.set_title("Transcription length distribution"); ax.legend()
        return fig

    _plot(_len_plot, out_dir, "length_hist.png")

    # ---------------- 3. character inventory
    print("\n=== 3. Character-set inventory ===")
    cnt = Counter("".join(df["Target"]))
    chars = pd.DataFrame(sorted(cnt.items(), key=lambda kv: -kv[1]),
                         columns=["char", "count"])
    chars["codepoint"] = chars["char"].apply(lambda c: f"U+{ord(c):04X}")
    chars["category"] = chars["char"].apply(
        lambda c: "digit" if c.isdigit() else "alpha" if c.isalpha()
        else "space" if c.isspace() else "punct/other")
    print(f"distinct characters: {len(chars)}")
    print(chars.head(25).to_string(index=False))
    rare = chars[chars["count"] <= 5]
    print(f"\ncharacters appearing <=5 times: {len(rare)} "
          "(archaic marks, ligatures, stray glyphs) - these are the CER long tail")
    print(rare.to_string(index=False))
    _save(chars, out_dir, "charset.csv")

    # Markers of archaic orthography worth knowing about explicitly.
    markers = {
        "caret_insertion(^)": df["Target"].str.contains(r"\^").sum(),
        "tilde/macron(~)": df["Target"].str.contains("~").sum(),
        "long_s(ſ)": df["Target"].str.contains("ſ").sum(),
        "superscript_abbrev(e.g. w^ch)": df["Target"].str.contains(r"\w\^\w").sum(),
        "xpian/xtian": df["Target"].str.lower().str.contains("xpian|xtian").sum(),
        "strikethrough(~~)": df["Target"].str.contains("~~").sum(),
        "digits": df["Target"].str.contains(r"\d").sum(),
        "currency/pound": df["Target"].str.contains("£").sum(),
    }
    print("\narchaic / structural markers (rows containing):")
    for k, v in markers.items():
        print(f"  {k:<32} {int(v):>5}")
    _save(pd.DataFrame(markers.items(), columns=["marker", "rows"]),
          out_dir, "archaic_markers.csv")

    # ---------------- 5. duplicates
    print("\n=== 5. Duplicate / near-duplicate detection ===")
    dup_text = df[df.duplicated("Target", keep=False)].sort_values("Target")
    print(f"rows sharing an identical transcription: {len(dup_text)} "
          f"({dup_text['Target'].nunique()} distinct strings)")
    print("These are genuine re-scans of the same clause in this corpus; keeping "
          "them is fine, but they MUST NOT straddle folds or CV leaks.")
    _save(dup_text[["ID", "Target"]], out_dir, "duplicate_texts.csv")

    # ---------------- 1. image geometry + near-duplicate images
    ids = df["ID"].tolist()
    if sample_images:
        ids = ids[:sample_images]
    print(f"\n=== 1b. Image geometry (n={len(ids)}) ===")
    from PIL import Image

    recs = []
    hashes = {}
    for i, rid in enumerate(ids):
        p = os.path.join(image_dir, f"{rid}.jpg")
        if not os.path.exists(p):
            recs.append({"ID": rid, "exists": False, "width": np.nan,
                         "height": np.nan, "aspect": np.nan})
            continue
        with Image.open(p) as im:
            w, h = im.size
        recs.append({"ID": rid, "exists": True, "width": w, "height": h,
                     "aspect": w / max(h, 1)})
        hv = dhash(p)
        if hv is not None:
            hashes.setdefault(hv, []).append(rid)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(ids)}")

    geo = pd.DataFrame(recs)
    if geo["exists"].any():
        print(geo[["width", "height", "aspect"]].describe().to_string())
        print(f"missing image files: {int((~geo['exists']).sum())}")
        _save(geo, out_dir, "image_geometry.csv")

        def _asp(plt):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(geo["aspect"].dropna(), bins=50)
            ax.set_xlabel("width / height"); ax.set_ylabel("count")
            ax.set_title("Image aspect-ratio distribution")
            return fig

        _plot(_asp, out_dir, "aspect_hist.png")

        near = {k: v for k, v in hashes.items() if len(v) > 1}
        print(f"exact dHash collisions (near-duplicate images): {len(near)} groups")
        if near:
            rows = [{"hash": k, "ids": ",".join(v), "n": len(v)} for k, v in near.items()]
            _save(pd.DataFrame(rows).sort_values("n", ascending=False),
                  out_dir, "near_duplicate_images.csv")
    else:
        print("No images found - image geometry, near-duplicate detection and the")
        print("multi-line census require ./data/images/ to be populated.")

    # ---------------- test manifest
    if test_csv and os.path.exists(test_csv):
        t = pd.read_csv(test_csv, dtype=str, keep_default_na=False)
        print(f"\n=== Test manifest: {len(t)} IDs ===")
        overlap = set(t.iloc[:, 0]) & set(df["ID"])
        print(f"ID overlap with train: {len(overlap)} (expected 0)")

    print("\nEDA complete. Next: run src.data.multiline (census) and "
          "src.data.exclusions (corrupted-row verification).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/Train.csv")
    ap.add_argument("--image_dir", default="data/images")
    ap.add_argument("--test_csv", default="data/Test.csv")
    ap.add_argument("--out_dir", default="report/eda")
    ap.add_argument("--sample_images", type=int, default=None)
    a = ap.parse_args()
    run(a.train_csv, a.image_dir, a.test_csv, a.out_dir, a.sample_images)


if __name__ == "__main__":
    main()
