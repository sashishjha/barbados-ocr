"""
confusion_network.py -- word-level ROVER + MBR hybrid ("confusion network")
for combining N-best lists from MULTIPLE, differently-calibrated models
(e.g. Qwen2.5-VL beams + TrOCR beams).

Addresses three specific failure modes diagnosed in the whole-sequence MBR
approach:

  1. Whole-sequence MBR can only SELECT one of the pooled hypotheses; it can
     never recombine "half right from model A, half right from model B".
     -> Fix: align hypotheses word-by-word (via rover.Profile) BEFORE doing
        any risk minimization, so the final output can mix segments.

  2. Beams from one model are near-duplicates of each other (highly
     correlated), so pooling e.g. 5+5 raw beams lets whichever model's
     cluster is larger dominate risk estimates regardless of correctness.
     -> Fix: cap how many beams per model enter the vote (max_per_model),
        and calibrate confidence WITHIN each model before combining.

  3. Raw beam scores from a VLM and an encoder-decoder aren't on the same
     scale, so pooled z-scoring/softmax silently favors whichever model's
     raw scores happen to have more spread.
     -> Fix: z-score + softmax EACH model's own beam scores separately
        (so each model's beams form a distribution that sums to 1), then
        apply an explicit, tunable per-model trust weight on top -- tune
        that trust weight against local CV (see local_cv_eval.py), not
        blindly assumed to be 1:1.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Sequence, Tuple

from rover import Profile, majority_vote


def _char_edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def word_dist(a: Optional[str], b: Optional[str]) -> float:
    """Distance between two column entries (a real word or GAP/None).
    Two gaps: 0. A gap vs a real word: 1 (maximally different).
    Two real words: normalized character edit distance in [0, 1]."""
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 1.0
    if a == b:
        return 0.0
    return _char_edit_distance(a, b) / max(len(a), len(b), 1)


def mbr_vote(col_tokens: Sequence[Optional[str]], hyp_conf: Sequence[float]) -> Optional[str]:
    """MBR-style generalization of majority_vote: pick the candidate token
    (or GAP) actually present in this column that minimizes the
    confidence-weighted expected word_dist to every other entry in the
    column. Unlike plain majority voting, a near-miss spelling variant
    ('twelth' vs 'twelfth') partially supports the correct word instead of
    splitting the vote as a totally separate candidate."""
    # Deterministic candidate order (first-occurrence), NOT set(col_tokens):
    # Python's set iteration order for strings depends on the process's hash
    # seed (randomized by default), which silently made tie-breaks
    # non-reproducible across runs. dict.fromkeys preserves first-seen order.
    candidates = list(dict.fromkeys(col_tokens))
    total_w = sum(hyp_conf) or 1.0
    best_cand, best_key = None, None
    for cand in candidates:
        risk = sum(w * word_dist(cand, other) for other, w in zip(col_tokens, hyp_conf)) / total_w
        # Round before comparing: mathematically-tied candidates can differ by
        # ~1e-16 purely from floating-point summation order, which would
        # otherwise silently override the intended first-occurrence tie-break
        # below with arbitrary noise.
        key = (round(risk, 9), cand is None)  # prefer real word on tie
        if best_key is None or key < best_key:
            best_key, best_cand = key, cand
    return best_cand


def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _zscore(xs: List[float]) -> List[float]:
    if len(xs) <= 1:
        return [0.0 for _ in xs]
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    sd = math.sqrt(var) or 1.0
    return [(x - mean) / sd for x in xs]


def calibrated_beams(
    model_beams: Dict[str, List[Tuple[str, float]]],
    max_per_model: int = 2,
    model_trust: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float]]:
    """Turn {'qwen': [(text, raw_score), ...], 'trocr': [...]} into a flat
    list of (text, confidence) pairs, calibrated so no single model
    dominates purely by beam count or raw-score scale.

    Steps per model: keep top `max_per_model` beams (by raw score), z-score
    THOSE raw scores within the model, softmax within the model (so each
    model's kept beams sum to 1 unit of "confidence mass"), then multiply
    by an explicit model_trust weight (default 1.0 -- set this from local
    CV results, e.g. if TrOCR is empirically stronger on your held-out
    folds, give it trust > 1.0 relative to the VLM).
    """
    model_trust = model_trust or {}
    out: List[Tuple[str, float]] = []
    for model_name, beams in model_beams.items():
        if not beams:
            continue
        beams_sorted = sorted(beams, key=lambda tb: tb[1], reverse=True)[:max_per_model]
        texts = [t for t, _ in beams_sorted]
        raw_scores = [s for _, s in beams_sorted]
        z = _zscore(raw_scores)
        weights = _softmax(z)
        trust = model_trust.get(model_name, 1.0)
        for text, w in zip(texts, weights):
            out.append((text, w * trust))
    return out


def confusion_network_combine(
    model_beams: Dict[str, List[Tuple[str, float]]],
    max_per_model: int = 2,
    model_trust: Optional[Dict[str, float]] = None,
    mode: str = "mbr",
) -> str:
    """End-to-end: calibrate -> align -> collapse.
    mode='mbr' uses the edit-distance-aware risk-minimization vote;
    mode='rover' falls back to plain majority voting (for comparison)."""
    beams = calibrated_beams(model_beams, max_per_model=max_per_model, model_trust=model_trust)
    if not beams:
        return ""
    prof = Profile()
    for text, conf in beams:
        if text:
            prof.add(text.split(), conf)
    vote_fn = mbr_vote if mode == "mbr" else majority_vote
    return prof.collapse(vote_fn)
