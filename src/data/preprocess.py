"""
Deterministic image preprocessing shared by every arm.

Pipeline (each step toggleable from config):
  grayscale -> CLAHE contrast normalization -> projection-profile deskew
  -> aspect-preserving resize -> right-pad to the model's canvas.

Three padding regimes are provided so the ablation asked for in the roadmap is
a config flag rather than a code change:
  * "right"   : pad only on the right (text left-aligned)      [default]
  * "center"  : split padding left/right
  * "replicate": pad by replicating the border column (no hard edge)

Padding value defaults to the estimated page background rather than pure white,
which avoids teaching the model a spurious "bright block = end of line" cue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image

from .multiline import _binarize, deskew_angle


@dataclass
class PreprocessConfig:
    grayscale: bool = True
    clean_background: bool = True
    clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    deskew: bool = True
    deskew_max_deg: float = 6.0
    target_height: int = 96
    max_width: int = 1024
    pad_mode: str = "right"      # right | center | replicate
    pad_value: int = -1          # -1 => estimate background from image border
    to_rgb: bool = True          # TrOCR / VLM processors expect 3 channels
    vertical_pad_frac: float = 0.20  # expand crop ±20% vertically to preserve ascenders/descenders


def clean_image_background(gray: np.ndarray) -> np.ndarray:
    """Removes non-uniform background illumination and stretches contrast.
    Uses cv2 if available, otherwise falls back to a simpler numpy method."""
    try:
        import cv2
        # Estimate background by dilating the dark text out
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        
        # Blur the background to smooth it out
        bg = cv2.GaussianBlur(bg, (21, 21), 0)
        
        # Divide by background to normalize (requires float32)
        norm = 255.0 * (gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0))
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        
        # Stretch contrast
        lo, hi = np.percentile(norm, 1), np.percentile(norm, 99)
        if hi > lo:
            norm = np.clip((norm - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
            
        return norm
    except Exception:
        # Numpy fallback: median filter approximation using block reduction
        from scipy.ndimage import median_filter
        try:
            bg = median_filter(gray, size=15)
            norm = 255.0 * (gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0))
            norm = np.clip(norm, 0, 255).astype(np.uint8)
            lo, hi = np.percentile(norm, 1), np.percentile(norm, 99)
            if hi > lo:
                norm = np.clip((norm - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
            return norm
        except Exception:
            return gray

def apply_clahe(gray: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE. Uses OpenCV when available, else a numpy tile-histogram fallback
    so the pipeline still runs on a login node without cv2."""
    try:
        import cv2

        c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        return c.apply(gray.astype(np.uint8))
    except Exception:
        h, w = gray.shape
        out = np.empty_like(gray, dtype=np.uint8)
        ys = np.linspace(0, h, grid + 1).astype(int)
        xs = np.linspace(0, w, grid + 1).astype(int)
        for i in range(grid):
            for j in range(grid):
                tile = gray[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
                if tile.size == 0:
                    continue
                lo, hi = np.percentile(tile, 2), np.percentile(tile, 98)
                if hi <= lo:
                    out[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = tile
                else:
                    out[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = np.clip(
                        (tile - lo) * 255.0 / (hi - lo), 0, 255
                    ).astype(np.uint8)
        return out


def estimate_background(gray: np.ndarray) -> int:
    """Median of the 3-pixel border ring = paper tone."""
    if gray.size == 0:
        return 255
    ring = np.concatenate([
        gray[:3, :].ravel(), gray[-3:, :].ravel(),
        gray[:, :3].ravel(), gray[:, -3:].ravel(),
    ])
    return int(np.median(ring))


def pad_vertical(gray: np.ndarray, frac: float = 0.20) -> np.ndarray:
    """Expand the crop ±frac vertically so ascenders/descenders survive tight boxes.

    The organizers' bounding boxes frequently clip the top/bottom of glyphs.
    Adding 20% padding on each side (filled with the estimated paper background)
    recovers those pixels before any other processing touches the image.
    """
    if frac <= 0:
        return gray
    h = gray.shape[0]
    pad_h = max(1, int(round(h * frac)))
    bg = estimate_background(gray)
    top = np.full((pad_h, gray.shape[1]), bg, dtype=np.uint8)
    bot = np.full((pad_h, gray.shape[1]), bg, dtype=np.uint8)
    return np.concatenate([top, gray, bot], axis=0)


def deskew(gray: np.ndarray, max_deg: float = 6.0) -> np.ndarray:
    try:
        from scipy.ndimage import rotate

        ink = _binarize(gray)
        ang = deskew_angle(ink, max_deg=max_deg)
        if abs(ang) < 0.25:
            return gray
        bg = estimate_background(gray)
        out = rotate(gray.astype(np.float32), ang, reshape=False,
                     order=1, mode="constant", cval=float(bg))
        return np.clip(out, 0, 255).astype(np.uint8)
    except Exception:
        return gray


def resize_and_pad(
    gray: np.ndarray, target_height: int, max_width: int,
    pad_mode: str = "right", pad_value: int = -1,
) -> Tuple[np.ndarray, float]:
    """Aspect-preserving resize to target_height, then pad/crop to max_width."""
    h, w = gray.shape
    if h == 0 or w == 0:
        return np.full((target_height, max_width), 255, np.uint8), 1.0

    scale = target_height / float(h)
    new_w = max(1, min(max_width, int(round(w * scale))))
    im = Image.fromarray(gray).resize((new_w, target_height), Image.BICUBIC)
    arr = np.asarray(im, dtype=np.uint8)

    if new_w == max_width:
        return arr, scale

    pad_total = max_width - new_w
    if pad_mode == "replicate":
        left_pad, right_pad = 0, pad_total
        edge = arr[:, -1:]
        pad_block = np.repeat(edge, right_pad, axis=1)
        return np.concatenate([arr, pad_block], axis=1), scale

    val = estimate_background(gray) if pad_value < 0 else int(pad_value)
    if pad_mode == "center":
        left_pad = pad_total // 2
        right_pad = pad_total - left_pad
    else:  # "right"
        left_pad, right_pad = 0, pad_total

    blocks = []
    if left_pad:
        blocks.append(np.full((target_height, left_pad), val, np.uint8))
    blocks.append(arr)
    if right_pad:
        blocks.append(np.full((target_height, right_pad), val, np.uint8))
    return np.concatenate(blocks, axis=1), scale


def preprocess_image(path_or_img, cfg: PreprocessConfig) -> Image.Image:
    """Full deterministic pipeline. Returns a PIL image (RGB if cfg.to_rgb)."""
    if isinstance(path_or_img, (str, bytes)):
        with Image.open(path_or_img) as im:
            gray = np.asarray(im.convert("L"), dtype=np.uint8)
    elif isinstance(path_or_img, Image.Image):
        gray = np.asarray(path_or_img.convert("L"), dtype=np.uint8)
    else:
        gray = np.asarray(path_or_img, dtype=np.uint8)
        if gray.ndim == 3:
            gray = np.asarray(Image.fromarray(gray).convert("L"), dtype=np.uint8)

    # Expand bounding box vertically FIRST to recover clipped ascenders/descenders.
    if cfg.vertical_pad_frac > 0:
        gray = pad_vertical(gray, cfg.vertical_pad_frac)

    if getattr(cfg, "clean_background", False):
        gray = clean_image_background(gray)

    if cfg.clahe:
        gray = apply_clahe(gray, cfg.clahe_clip, cfg.clahe_grid)
    if cfg.deskew:
        gray = deskew(gray, cfg.deskew_max_deg)

    arr, _ = resize_and_pad(gray, cfg.target_height, cfg.max_width,
                            cfg.pad_mode, cfg.pad_value)
    im = Image.fromarray(arr)
    return im.convert("RGB") if cfg.to_rgb else im


def vlm_thumbnail(path: str, vertical_pad_frac: float = 0.20,
                  target_height: int = 0, max_width: int = 4096, isolate_band: bool = False) -> Image.Image:
    """Aspect-ratio-bucketed thumbnail, ported verbatim in spirit from the
    organizers' VLM starter `load_image()`. Kept separate from the pipeline
    above because the VLM processor prefers native aspect ratios over a
    fixed-canvas pad.

    vertical_pad_frac: expand the raw crop this fraction up and down before
    thumbnailing so ascenders/descenders are not clipped.
    """
    img = Image.open(path).convert("RGB")
    if isolate_band:
        img = isolate_ink_band(img)

    # Expand vertically in PIL space before any size transformation.
    if vertical_pad_frac > 0:
        gray = np.asarray(img.convert("L"), dtype=np.uint8)
        gray_padded = pad_vertical(gray, vertical_pad_frac)
        # Reconstruct RGB by padding each channel with the background tone.
        bg = estimate_background(gray)
        h_orig, w = img.size[1], img.size[0]
        pad_h = gray_padded.shape[0] - h_orig
        pad_top = max(1, int(round(h_orig * vertical_pad_frac)))
        pad_bot = pad_h - pad_top
        rgb = np.asarray(img, dtype=np.uint8)
        top_block = np.full((pad_top, w, 3), bg, dtype=np.uint8)
        bot_block = np.full((pad_bot, w, 3), bg, dtype=np.uint8)
        rgb_padded = np.concatenate([top_block, rgb, bot_block], axis=0)
        img = Image.fromarray(rgb_padded, mode="RGB")

    w, h = img.size
    if target_height:
        # Fixed-height resize (up or down), aspect kept: every line gets the same
        # glyph scale. Width capped so extreme lines stay within max_pixels.
        new_w = min(max_width, max(1, round(w * target_height / max(h, 1))))
        return img.resize((new_w, target_height), Image.BICUBIC)
    aspect = w / max(h, 1)
    if aspect > 12:
        img.thumbnail((1400, 256))
    elif aspect > 6:
        img.thumbnail((1400, 384))
    elif aspect > 3:
        img.thumbnail((1200, 512))
    else:
        img.thumbnail((1024, 512))
    return img



_STYLE_CACHE = {}


def style_tag(path: str) -> str:
    """Scan fingerprint for the prompt: native pixel size + paper colour. The fixed-height resize used for the VLM
    erases the native scale, which identifies the source volume (and with it transcription conventions)."""
    t = _STYLE_CACHE.get(path)
    if t is None:
        im = Image.open(path)
        w, h = im.size
        im.draft("RGB", (max(1, w // 4), max(1, h // 4)))
        a = np.asarray(im.convert("RGB").resize((64, 16)), dtype=float)
        bg = np.percentile(a.reshape(-1, 3), 80, axis=0)
        t = f"Scan: {w}x{h} px, paper rgb({int(bg[0])},{int(bg[1])},{int(bg[2])})."
        _STYLE_CACHE[path] = t
    return t


def isolate_ink_band(img, min_ink_cover: float = 0.40, rel_thresh: float = 0.20, pad_frac: float = 0.15):
    """Crop a line image vertically to the contiguous row band that holds the line's ink (target line), dropping blank margins
    and the partial neighbouring lines. Band = rows around the strongest smoothed ink peak where the profile stays above
    rel_thresh*peak, widened until it covers >= min_ink_cover of all ink, plus pad_frac of the band height on both sides."""
    from scipy.ndimage import gaussian_filter1d
    g = np.asarray(img.convert("L"), dtype=np.float32)
    h = g.shape[0]
    if h < 24:
        return img
    bg = np.percentile(g, 85); lo = np.percentile(g, 3)
    ink = (g < bg - 0.35 * max(1.0, bg - lo)).astype(np.float32)
    prof = gaussian_filter1d(ink.sum(1), max(2.0, 0.02 * h))
    if prof.max() <= 0:
        return img
    pk = int(prof.argmax()); thr = rel_thresh * prof[pk]
    y0 = pk
    while y0 > 0 and prof[y0 - 1] > thr: y0 -= 1
    y1 = pk
    while y1 < h - 1 and prof[y1 + 1] > thr: y1 += 1
    tot = prof.sum()
    while prof[y0:y1 + 1].sum() < min_ink_cover * tot and (y0 > 0 or y1 < h - 1):
        if y0 > 0 and (y1 >= h - 1 or prof[y0 - 1] >= prof[y1 + 1]): y0 -= 1
        elif y1 < h - 1: y1 += 1
    bh = y1 - y0 + 1; pad = int(round(pad_frac * bh))
    y0 = max(0, y0 - pad); y1 = min(h - 1, y1 + pad)
    return img.crop((0, y0, img.size[0], y1 + 1)) if (y1 - y0) >= 12 else img
