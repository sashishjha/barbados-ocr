"""
On-the-fly augmentation. PROVIDED IMAGES ONLY - no external image or text
corpus is ever touched, which keeps this inside the "train only on the provided
dataset" rule (geometric/photometric distortion of supplied images is standard
practice and explicitly allowed).

Implemented, all seeded and config-driven:
  * elastic warp (grid displacement)
  * small affine: rotation, shear, scale, translation
  * brightness / contrast / gamma jitter
  * ink fade (local erosion of stroke darkness)
  * blot / bleed-through simulation (random dark and light blobs)
  * background tint (paper yellowing)
  * gaussian blur + gaussian/speckle noise
  * random horizontal stretch (writing-width variability)

Deliberately NOT included: horizontal/vertical flips, 90-degree rotations,
cutout over large regions - all destroy readable text and would inject label
noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class AugmentConfig:
    enabled: bool = True
    p_elastic: float = 0.30
    elastic_alpha: float = 8.0
    elastic_sigma: float = 6.0

    p_affine: float = 0.50
    max_rotate_deg: float = 2.0
    max_shear: float = 0.06
    scale_range: tuple = (0.94, 1.06)
    max_translate_frac: float = 0.02

    p_photometric: float = 0.60
    brightness_range: tuple = (0.80, 1.20)
    contrast_range: tuple = (0.80, 1.25)
    gamma_range: tuple = (0.85, 1.20)

    p_ink_fade: float = 0.30
    ink_fade_strength: tuple = (0.15, 0.55)

    p_blot: float = 0.20
    blot_count: tuple = (1, 4)
    blot_radius_frac: tuple = (0.01, 0.05)

    p_tint: float = 0.25
    tint_strength: tuple = (0.02, 0.12)

    p_blur: float = 0.20
    blur_radius: tuple = (0.3, 1.1)
    p_noise: float = 0.25
    noise_sigma: tuple = (2.0, 9.0)

    p_stretch: float = 0.25
    stretch_range: tuple = (0.90, 1.12)

    # vertical window shift: content moves up/down by up to this fraction of the height (label unchanged);
    # teaches robustness to where the crop window sits relative to the centre line. Off by default.
    p_vshift: float = 0.0
    vshift_frac: float = 0.08

    # Curriculum: multiply every probability by this factor (ramped by trainer)
    strength: float = 1.0
    seed: Optional[int] = None
    _rng: object = field(default=None, repr=False, compare=False)


class Augmenter:
    def __init__(self, cfg: AugmentConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------------ utils
    def _p(self, base: float) -> bool:
        return self.rng.random() < min(1.0, base * self.cfg.strength)

    def _u(self, lo, hi):
        return float(self.rng.uniform(lo, hi))

    # ------------------------------------------------------------------ ops
    def elastic(self, arr: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter, map_coordinates

        h, w = arr.shape[:2]
        a, s = self.cfg.elastic_alpha, self.cfg.elastic_sigma
        dx = gaussian_filter(self.rng.random((h, w)) * 2 - 1, s, mode="constant") * a
        dy = gaussian_filter(self.rng.random((h, w)) * 2 - 1, s, mode="constant") * a * 0.5
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        coords = [np.reshape(y + dy, -1), np.reshape(x + dx, -1)]
        if arr.ndim == 2:
            return map_coordinates(arr, coords, order=1, mode="nearest").reshape(h, w)
        chans = [
            map_coordinates(arr[..., c], coords, order=1, mode="nearest").reshape(h, w)
            for c in range(arr.shape[2])
        ]
        return np.stack(chans, axis=-1)

    def affine(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        ang = self._u(-self.cfg.max_rotate_deg, self.cfg.max_rotate_deg)
        shear = self._u(-self.cfg.max_shear, self.cfg.max_shear)
        scale = self._u(*self.cfg.scale_range)
        tx = self._u(-1, 1) * self.cfg.max_translate_frac * w
        ty = self._u(-1, 1) * self.cfg.max_translate_frac * h

        rad = np.deg2rad(ang)
        cos, sin = np.cos(rad) / scale, np.sin(rad) / scale
        cx, cy = w / 2.0, h / 2.0
        a, b = cos, sin + shear
        d, e = -sin, cos
        c = cx - a * cx - b * cy + tx
        f = cy - d * cx - e * cy + ty

        bg = int(np.median(np.asarray(img.convert("L"))[:3, :]))
        fill = (bg, bg, bg) if img.mode == "RGB" else bg
        return img.transform((w, h), Image.AFFINE, (a, b, c, d, e, f),
                             resample=Image.BILINEAR, fillcolor=fill)

    def photometric(self, img: Image.Image) -> Image.Image:
        img = ImageEnhance.Brightness(img).enhance(self._u(*self.cfg.brightness_range))
        img = ImageEnhance.Contrast(img).enhance(self._u(*self.cfg.contrast_range))
        g = self._u(*self.cfg.gamma_range)
        lut = [int(np.clip(((i / 255.0) ** g) * 255.0, 0, 255)) for i in range(256)]
        if img.mode == "RGB":
            lut = lut * 3
        return img.point(lut)

    def ink_fade(self, arr: np.ndarray) -> np.ndarray:
        """Pull dark pixels toward the background: simulates faded iron-gall ink."""
        from scipy.ndimage import gaussian_filter

        strength = self._u(*self.cfg.ink_fade_strength)
        g = arr if arr.ndim == 2 else arr.mean(axis=2)
        bg = float(np.median(g[:3, :]))
        mask = gaussian_filter(self.rng.random(g.shape[:2]).astype(np.float32), 12.0)
        mask = (mask - mask.min()) / max(1e-6, mask.max() - mask.min())
        fade = (mask * strength)[..., None] if arr.ndim == 3 else mask * strength
        out = arr.astype(np.float32) * (1 - fade) + bg * fade
        return np.clip(out, 0, 255).astype(np.uint8)

    def blots(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        out = arr.astype(np.float32).copy()
        n = int(self.rng.integers(self.cfg.blot_count[0], self.cfg.blot_count[1] + 1))
        yy, xx = np.mgrid[0:h, 0:w]
        for _ in range(n):
            cy, cx = self.rng.integers(0, h), self.rng.integers(0, w)
            r = max(1.0, self._u(*self.cfg.blot_radius_frac) * max(h, w))
            blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))
            dark = self.rng.random() < 0.6
            amp = self._u(20, 70) * (-1 if dark else 1)
            delta = blob * amp
            out += delta[..., None] if out.ndim == 3 else delta
        return np.clip(out, 0, 255).astype(np.uint8)

    def tint(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            return img
        s = self._u(*self.cfg.tint_strength)
        arr = np.asarray(img).astype(np.float32)
        arr[..., 2] *= (1 - s)          # drop blue -> warm parchment
        arr[..., 1] *= (1 - s * 0.4)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def noise(self, arr: np.ndarray) -> np.ndarray:
        sigma = self._u(*self.cfg.noise_sigma)
        return np.clip(
            arr.astype(np.float32) + self.rng.normal(0, sigma, arr.shape), 0, 255
        ).astype(np.uint8)

    def stretch(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        f = self._u(*self.cfg.stretch_range)
        nw = max(8, int(round(w * f)))
        img = img.resize((nw, h), Image.BICUBIC)
        if nw < w:
            bg = int(np.median(np.asarray(img.convert("L"))[:3, :]))
            canvas = Image.new(img.mode, (w, h),
                               (bg, bg, bg) if img.mode == "RGB" else bg)
            canvas.paste(img, (0, 0))
            return canvas
        return img.crop((0, 0, w, h))

    def vshift(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        dy = int(round(self._u(-self.cfg.vshift_frac, self.cfg.vshift_frac) * h))
        if dy == 0:
            return img
        bg = int(np.median(np.asarray(img.convert("L"))[:3, :]))
        canvas = Image.new(img.mode, (w, h), (bg, bg, bg) if img.mode == "RGB" else bg)
        canvas.paste(img, (0, dy))
        return canvas

    # ------------------------------------------------------------------ main
    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.cfg.enabled:
            return img

        if self._p(self.cfg.p_vshift):
            img = self.vshift(img)
        if self._p(self.cfg.p_stretch):
            img = self.stretch(img)
        if self._p(self.cfg.p_affine):
            img = self.affine(img)

        arr = np.asarray(img)
        if self._p(self.cfg.p_elastic):
            arr = self.elastic(arr)
        if self._p(self.cfg.p_ink_fade):
            arr = self.ink_fade(arr)
        if self._p(self.cfg.p_blot):
            arr = self.blots(arr)
        if self._p(self.cfg.p_noise):
            arr = self.noise(arr)
        img = Image.fromarray(arr.astype(np.uint8))

        if self._p(self.cfg.p_photometric):
            img = self.photometric(img)
        if self._p(self.cfg.p_tint):
            img = self.tint(img)
        if self._p(self.cfg.p_blur):
            img = img.filter(ImageFilter.GaussianBlur(self._u(*self.cfg.blur_radius)))
        return img
