"""
Text normalization used by the evaluation harness.

IMPORTANT / TRANSPARENCY NOTE
-----------------------------
The organizers' `eval_metrics.py` (reproduced byte-identically across all three
starter kits in the reference bundle) imports `compute_weighted_wer` and
`compute_weighted_cer` from an `evaluations/` package that was **NOT shipped**
inside `Starters.zip`. We therefore do not know the exact normalization the
official scorer applies.

We implement three selectable normalization regimes and default to `raw`
(the most conservative assumption: the scorer compares strings as-is apart
from CSV parsing). Every training/selection run logs the metric under all
three regimes so that model selection is not silently overfitted to a guess.

Set `eval.normalization` in the config to one of: raw | strip_ws | aggressive
"""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def norm_raw(text: str) -> str:
    """Only guarantee a string and collapse nothing. NaN/None -> ''."""
    if text is None:
        return ""
    s = str(text)
    if s.lower() == "nan":
        return ""
    return s


def norm_strip_ws(text: str) -> str:
    """Unicode NFKC + whitespace collapse + strip. Casing/punctuation kept."""
    s = norm_raw(text)
    s = unicodedata.normalize("NFKC", s)
    return _WS_RE.sub(" ", s).strip()


def norm_aggressive(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace. Diagnostic ceiling."""
    s = norm_strip_ws(text).lower()
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


NORMALIZERS = {
    "raw": norm_raw,
    "strip_ws": norm_strip_ws,
    "aggressive": norm_aggressive,
}


def get_normalizer(name: str = "raw"):
    if name not in NORMALIZERS:
        raise ValueError(f"Unknown normalization '{name}'. Choose from {list(NORMALIZERS)}")
    return NORMALIZERS[name]
