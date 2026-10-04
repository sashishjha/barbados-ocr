"""text_metrics.py -- CER / WER / competition metric, shared by local_cv_eval.py."""
from __future__ import annotations
from typing import Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Generic edit distance over any sequence of comparable items
    (characters or words)."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def cer(ref: str, hyp: str) -> float:
    ref_chars = list(ref)
    if not ref_chars:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref_chars, list(hyp)) / len(ref_chars)


def wer(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    if not ref_words:
        return 0.0 if hyp.split() else 0.0
    return levenshtein(ref_words, hyp.split()) / len(ref_words)


def competition_score(ref: str, hyp: str, cer_weight: float = 0.7, wer_weight: float = 0.3) -> float:
    """Lower is better -- matches the 0.7*CER + 0.3*WER metric referenced in the doc."""
    return cer_weight * cer(ref, hyp) + wer_weight * wer(ref, hyp)
