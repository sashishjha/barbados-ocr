"""
Torch datasets shared by the TrOCR and CTC arms, plus the HF-style record
builder used by the VLM arm.

Features:
  * fold-aware train/val splitting from train_folds.csv
  * pseudo-label mixing at a capped ratio with a per-sample loss weight
  * StackMix injection (provided-images-only synthetic lines)
  * long-line curriculum via a WeightedRandomSampler
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from .augment import AugmentConfig, Augmenter
from .preprocess import PreprocessConfig, preprocess_image
from .stackmix import StackMixConfig, load_generator


@dataclass
class SplitSpec:
    folds_csv: str
    fold: int
    image_dir: str
    pseudo_csv: Optional[str] = None
    pseudo_max_ratio: float = 0.30
    pseudo_loss_weight: float = 0.5
    pseudo_image_dir: Optional[str] = None
    use_all_folds: bool = False   # final models: train on every labelled line (val is then not held out)


def load_split(spec: SplitSpec, seed: int = 42):
    """Return (train_df, val_df). Pseudo-labels go into TRAIN ONLY, never val."""
    df = pd.read_csv(spec.folds_csv, dtype={"ID": str})
    df["Target"] = df["Target"].fillna("")
    df["is_pseudo"] = False
    df["loss_weight"] = 1.0
    df["image_dir"] = spec.image_dir

    val = df[df["fold"] == spec.fold].reset_index(drop=True)
    train = (df if spec.use_all_folds else df[df["fold"] != spec.fold]).reset_index(drop=True)

    if spec.pseudo_csv and os.path.exists(spec.pseudo_csv):
        p = pd.read_csv(spec.pseudo_csv, dtype={"ID": str})
        p = p[p["Target"].fillna("").str.strip() != ""].copy()
        # Hard safety: a pseudo-label must never collide with a real train ID.
        p = p[~p["ID"].isin(set(df["ID"]))]
        cap = int(len(train) * spec.pseudo_max_ratio)
        if len(p) > cap:
            p = p.sample(n=cap, random_state=seed)
        p["is_pseudo"] = True
        p["loss_weight"] = spec.pseudo_loss_weight
        p["fold"] = -1
        p["label_len"] = p["Target"].str.len()
        p["image_dir"] = spec.pseudo_image_dir or spec.image_dir
        keep = ["ID", "Target", "fold", "label_len", "is_pseudo", "loss_weight", "image_dir"]
        for c in keep:
            if c not in p.columns:
                p[c] = np.nan
        train = pd.concat([train[keep], p[keep]], ignore_index=True)
        print(f"[DATA] mixed in {len(p)} pseudo-labels "
              f"(cap {spec.pseudo_max_ratio:.0%}, loss_weight {spec.pseudo_loss_weight})")

    print(f"[DATA] fold={spec.fold}  train={len(train)}  val={len(val)}")
    return train.reset_index(drop=True), val.reset_index(drop=True)


class LineOCRDataset(Dataset):
    """Yields dict(image=PIL.Image, text=str, id=str, loss_weight=float)."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        pre_cfg: PreprocessConfig,
        aug_cfg: Optional[AugmentConfig] = None,
        stackmix_cfg: Optional[StackMixConfig] = None,
        stackmix_bank_dir: Optional[str] = None,
        train: bool = True,
        raw_pil: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.pre_cfg = pre_cfg
        self.train = train
        self.raw_pil = raw_pil
        self.augmenter = Augmenter(aug_cfg) if (train and aug_cfg and aug_cfg.enabled) else None

        self.stackmix = None
        if train and stackmix_cfg and stackmix_cfg.enabled and stackmix_bank_dir:
            self.stackmix = load_generator(stackmix_bank_dir, stackmix_cfg)
            if self.stackmix is None:
                print(f"[DATA] StackMix requested but no crop bank at {stackmix_bank_dir}; "
                      "continuing without it")

    def __len__(self) -> int:
        return len(self.df)

    def set_augment_strength(self, s: float) -> None:
        """Curriculum hook: trainer ramps augmentation strength across epochs."""
        if self.augmenter:
            self.augmenter.cfg.strength = float(s)

    def _resolve_path(self, row) -> str:
        base = row.get("image_dir") or self.image_dir
        if isinstance(base, float) and np.isnan(base):
            base = self.image_dir
        return os.path.join(str(base), f"{row['ID']}.jpg")

    def __getitem__(self, i: int) -> Dict:
        row = self.df.iloc[i]

        if self.stackmix is not None:
            got = self.stackmix.maybe_generate()
            if got is not None:
                img, text, _ = got
                if not self.raw_pil:
                    img = preprocess_image(img, self.pre_cfg)
                if self.augmenter:
                    img = self.augmenter(img)
                return {"image": img, "text": text, "id": f"stackmix::{row['ID']}",
                        "loss_weight": 1.0}

        path = self._resolve_path(row)
        if not os.path.exists(path):
            # Never crash a long training run on one bad path.
            img = Image.new("RGB", (self.pre_cfg.max_width, self.pre_cfg.target_height),
                            (255, 255, 255))
        elif self.raw_pil:
            img = Image.open(path).convert("RGB")
        else:
            img = preprocess_image(path, self.pre_cfg)

        if self.augmenter:
            img = self.augmenter(img)

        return {
            "image": img,
            "text": str(row["Target"]),
            "id": str(row["ID"]),
            "loss_weight": float(row.get("loss_weight", 1.0)),
        }


def long_line_sampler(df: pd.DataFrame, weight_factor: float = 0.5,
                      power: float = 1.0, num_samples: Optional[int] = None,
                      seed: int = 42) -> WeightedRandomSampler:
    """Over-sample long lines. `power=0` -> uniform; ramp it up in later epochs
    so the model spends its final capacity where the weighted metric lives."""
    lens = df["Target"].fillna("").str.len().astype(float).clip(lower=1.0)
    w = np.power(lens.to_numpy(), weight_factor * power)
    w = w * df.get("loss_weight", pd.Series(1.0, index=df.index)).to_numpy()
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=num_samples or len(df),
        replacement=True,
        generator=g,
    )


def vlm_records(df: pd.DataFrame, image_dir: str) -> List[Dict]:
    """Flat record list for the VLM trainer's collator."""
    out = []
    for _, r in df.iterrows():
        base = r.get("image_dir") or image_dir
        if isinstance(base, float) and np.isnan(base):
            base = image_dir
        out.append({
            "image": os.path.join(str(base), f"{r['ID']}.jpg"),
            "text": " ".join(str(r["Target"]).split()),
            "id": str(r["ID"]),
            "loss_weight": float(r.get("loss_weight", 1.0)),
        })
    return out
