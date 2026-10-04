"""
StackMix augmentation - officially permitted by the organizers (24 Aug ruling).

What is allowed, and what this implements
------------------------------------------
Segment word-level crops from the PROVIDED TRAINING images only, then
programmatically recombine them side-by-side into new synthetic line images
whose label is the concatenation of the source crops' labels. Fully automated
and reproducible; no test images, no external image or text data, no manual
crop selection or correction.

Compliance guardrails baked in:
  * `build_crop_bank()` refuses any CSV that lacks a Target column, so a test
    manifest can never be used as a source.
  * Crop extraction is purely algorithmic (vertical projection valleys) and the
    word<->crop alignment is accepted only when the detected gap count matches
    the whitespace-token count of the label. Rows that do not align are DROPPED,
    never hand-fixed.
  * Every generated sample records its source (image_id, word_index) list in the
    manifest so a reviewer can regenerate it bit-for-bit from the seed.

Why it helps here: the corpus is ~4k lines of a closed legal vocabulary with
heavy repetition ("To all Christian people to whome these presents shall come").
Recombining real word crops produces novel long lines in the same hand and the
same vocabulary - and long lines are exactly where the length-weighted metric
lives.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .multiline import _binarize
from .preprocess import estimate_background


@dataclass
class StackMixConfig:
    enabled: bool = True
    p: float = 0.35                    # probability a training sample is StackMix
    min_words: int = 6                 # generated line length, in words
    max_words: int = 16
    target_len_range: Tuple[int, int] = (60, 120)   # chars; bias toward long lines
    gap_px_range: Tuple[int, int] = (10, 34)
    height_jitter: float = 0.06
    baseline_jitter_px: int = 3
    same_source_bias: float = 0.35     # prefer crops from the same page/hand
    min_crop_width: int = 8
    bank_dir: Optional[str] = None
    max_bank_images: Optional[int] = None
    seed: int = 42


# --------------------------------------------------------------------------
# Crop bank construction
# --------------------------------------------------------------------------
def segment_words(path: str, min_gap_px: int = 6, min_width: int = 8) -> List[Tuple[int, int]]:
    """Vertical projection profile -> word column spans (x0, x1)."""
    if not os.path.exists(path):
        return []
    with Image.open(path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.uint8)
    ink = _binarize(gray)
    col = ink.sum(axis=0)
    if col.max() == 0:
        return []

    on = col > 0
    spans, start = [], None
    gap = 0
    for i, v in enumerate(on):
        if v:
            if start is None:
                start = i
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap >= min_gap_px:
                    spans.append((start, i - gap + 1))
                    start = None
                    gap = 0
    if start is not None:
        spans.append((start, len(on)))
    return [(a, b) for a, b in spans if (b - a) >= min_width]


def build_crop_bank(
    train_csv: str,
    image_dir: str,
    out_dir: str,
    cfg: StackMixConfig,
    manifest_name: str = "stackmix_crop_bank.csv",
) -> pd.DataFrame:
    """Extract aligned word crops from the training set into a reusable bank."""
    df = pd.read_csv(train_csv, dtype={"ID": str})
    if "Target" not in df.columns:
        raise ValueError(
            "StackMix source CSV must contain a Target column. Refusing to run: "
            "this guard exists so test images can never enter the crop bank."
        )
    if cfg.max_bank_images:
        df = df.head(cfg.max_bank_images)

    os.makedirs(out_dir, exist_ok=True)
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    records: List[Dict] = []
    n_aligned = n_skipped = 0

    for _, row in df.iterrows():
        rid, label = str(row["ID"]), str(row["Target"])
        words = label.split()
        path = os.path.join(image_dir, f"{rid}.jpg")
        spans = segment_words(path, min_width=cfg.min_crop_width)

        # Accept ONLY when segmentation count matches token count. This is the
        # automated alignment rule; mismatches are dropped, never corrected.
        if not words or len(spans) != len(words):
            n_skipped += 1
            continue

        with Image.open(path) as im:
            img = im.convert("L")
            for k, ((x0, x1), word) in enumerate(zip(spans, words)):
                crop = img.crop((x0, 0, x1, img.height))
                fn = f"{rid}__{k:02d}.png"
                crop.save(os.path.join(crops_dir, fn))
                records.append({
                    "crop_file": fn,
                    "source_id": rid,
                    "word_index": k,
                    "word": word,
                    "width": crop.width,
                    "height": crop.height,
                })
        n_aligned += 1

    bank = pd.DataFrame(records)
    manifest = os.path.join(out_dir, manifest_name)
    bank.to_csv(manifest, index=False)

    print(f"[STACKMIX] source lines aligned : {n_aligned}")
    print(f"[STACKMIX] source lines skipped : {n_skipped} (segmentation/token mismatch)")
    print(f"[STACKMIX] word crops banked    : {len(bank)}")
    print(f"[STACKMIX] unique vocabulary    : {bank['word'].nunique() if len(bank) else 0}")
    print(f"[STACKMIX] manifest             : {manifest}")
    return bank


# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------
class StackMixGenerator:
    """Samples word crops from the bank and composites a new line image."""

    def __init__(self, bank: pd.DataFrame, crops_dir: str, cfg: StackMixConfig):
        if len(bank) == 0:
            raise ValueError("Empty StackMix crop bank")
        self.bank = bank.reset_index(drop=True)
        self.crops_dir = crops_dir
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.by_source = {k: v.index.to_numpy()
                          for k, v in self.bank.groupby("source_id")}
        self.median_h = int(self.bank["height"].median())

    def _pick_indices(self) -> List[int]:
        cfg = self.cfg
        n_words = int(self.rng.integers(cfg.min_words, cfg.max_words + 1))
        target_len = int(self.rng.integers(*cfg.target_len_range))

        idxs: List[int] = []
        cur_len = 0
        anchor_src = None
        for _ in range(n_words):
            if anchor_src is not None and self.rng.random() < cfg.same_source_bias:
                pool = self.by_source[anchor_src]
                i = int(pool[self.rng.integers(len(pool))])
            else:
                i = int(self.rng.integers(len(self.bank)))
                anchor_src = self.bank.at[i, "source_id"]
            idxs.append(i)
            cur_len += len(str(self.bank.at[i, "word"])) + 1
            if cur_len >= target_len:
                break
        return idxs

    def generate(self) -> Tuple[Image.Image, str, List[Dict]]:
        cfg = self.cfg
        idxs = self._pick_indices()

        tiles, words, provenance = [], [], []
        for i in idxs:
            r = self.bank.iloc[i]
            p = os.path.join(self.crops_dir, r["crop_file"])
            if not os.path.exists(p):
                continue
            with Image.open(p) as im:
                tile = im.convert("L")
                jitter = 1.0 + float(self.rng.uniform(-cfg.height_jitter, cfg.height_jitter))
                nh = max(8, int(self.median_h * jitter))
                nw = max(4, int(tile.width * nh / max(tile.height, 1)))
                tiles.append(np.asarray(tile.resize((nw, nh), Image.BICUBIC)))
            words.append(str(r["word"]))
            provenance.append({"source_id": r["source_id"], "word_index": int(r["word_index"])})

        if not tiles:
            raise RuntimeError("StackMix produced no tiles")

        pad = cfg.baseline_jitter_px
        canvas_h = max(t.shape[0] for t in tiles) + 2 * pad
        gaps = [int(self.rng.integers(*cfg.gap_px_range)) for _ in range(len(tiles) - 1)]
        canvas_w = sum(t.shape[1] for t in tiles) + sum(gaps) + 2 * pad

        bg = int(np.median([np.median(t[:2, :]) for t in tiles]))
        canvas = np.full((canvas_h, canvas_w), bg, dtype=np.uint8)

        x = pad
        for j, t in enumerate(tiles):
            dy = int(self.rng.integers(-pad, pad + 1))
            y = max(0, min(canvas_h - t.shape[0], pad + dy))
            region = canvas[y:y + t.shape[0], x:x + t.shape[1]]
            # darken-blend so tile paper tone does not overwrite the canvas
            canvas[y:y + t.shape[0], x:x + t.shape[1]] = np.minimum(region, t)
            x += t.shape[1]
            if j < len(gaps):
                x += gaps[j]

        return Image.fromarray(canvas).convert("RGB"), " ".join(words), provenance

    def maybe_generate(self):
        """Returns (img, label, provenance) with probability cfg.p, else None."""
        if not self.cfg.enabled or self.rng.random() >= self.cfg.p:
            return None
        try:
            return self.generate()
        except Exception:
            return None


def load_generator(bank_dir: str, cfg: StackMixConfig) -> Optional[StackMixGenerator]:
    manifest = os.path.join(bank_dir, "stackmix_crop_bank.csv")
    if not os.path.exists(manifest):
        return None
    bank = pd.read_csv(manifest)
    return StackMixGenerator(bank, os.path.join(bank_dir, "crops"), cfg)


def materialize(
    bank_dir: str, out_dir: str, n_samples: int, cfg: StackMixConfig
) -> pd.DataFrame:
    """Optionally pre-render a fixed StackMix set to disk (reproducible from the
    seed). Useful when dataloader CPU is the bottleneck."""
    gen = load_generator(bank_dir, cfg)
    if gen is None:
        raise FileNotFoundError(f"No crop bank in {bank_dir}; run build_crop_bank first")

    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for i in range(n_samples):
        img, label, prov = gen.generate()
        sid = f"stackmix_{cfg.seed}_{i:06d}"
        img.save(os.path.join(img_dir, f"{sid}.jpg"), quality=95)
        rows.append({"ID": sid, "Target": label, "provenance": json.dumps(prov)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "stackmix_train.csv"), index=False)
    print(f"[STACKMIX] materialized {len(df)} synthetic lines -> {out_dir}")
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/derived_checkpoint/train_clean.csv")
    ap.add_argument("--image_dir", default="data/images")
    ap.add_argument("--bank_dir", default="data/derived_checkpoint/stackmix_bank")
    ap.add_argument("--materialize", type=int, default=0,
                    help="If >0, also pre-render this many synthetic lines")
    ap.add_argument("--out_dir", default="data/derived_checkpoint/stackmix_rendered")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = StackMixConfig(seed=a.seed)
    build_crop_bank(a.train_csv, a.image_dir, a.bank_dir, cfg)
    if a.materialize > 0:
        materialize(a.bank_dir, a.out_dir, a.materialize, cfg)
