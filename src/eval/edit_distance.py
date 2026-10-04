"""Pure-python Levenshtein distance over arbitrary sequences (chars or words).

No external dependency so the eval harness can run on a login node without a
GPU environment. jiwer/evaluate are optional cross-checks only (see tests).
"""

from __future__ import annotations

from typing import Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Standard O(len(a)*len(b)) time, O(min) space edit distance."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(a) + 1))
    for j, cb in enumerate(b, start=1):
        cur = [j] + [0] * len(a)
        for i, ca in enumerate(a, start=1):
            cur[i] = min(
                prev[i] + 1,        # deletion
                cur[i - 1] + 1,     # insertion
                prev[i - 1] + (ca != cb),  # substitution
            )
        prev = cur
    return prev[len(a)]
