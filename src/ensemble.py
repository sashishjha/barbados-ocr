"""
Ensembling across arms.

Two strategies, both selectable and both tunable on OOF data:

1. ROVER-style character voting (`--method rover`)
   Align all candidate strings to a pivot (the candidate with the lowest mean
   edit distance to the others), then vote position-by-position on the aligned
   character streams. Because the metric is 0.7*CER + 0.3*WER, character-level
   fusion is worth more here than sentence-level selection.

2. Minimum-expected-weighted-CER selection (`--method mbr`)
   Minimum Bayes Risk: pick the candidate minimising its expected CER against
   all other candidates, weighted by each candidate's model score. This never
   invents a string that no model produced, so it is the safer of the two.

Weights per arm can be set from OOF performance (`--arm_weights trocr=1.0,vlm=1.3`).
Tune the choice of method, the arm weights, and the lexicon-rescoring betas on
OOF - never on the public leaderboard, which the competition page describes
inconsistently (30/70 vs 20/80) and should be treated as noisy.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.eval.edit_distance import levenshtein
from src.postprocess import TrainLexicon, lexicon_rescore
from src.utils.common import get_logger

log = get_logger("ensemble")


def _rel_cer(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return levenshtein(list(a), list(b)) / max(1, len(b) if b else len(a))


# --------------------------------------------------------------------------
def align_to_pivot(pivot: str, other: str) -> List[Tuple[int, str]]:
    """Needleman-Wunsch style backtrace giving, for each pivot position, the
    character `other` contributes ('' for a deletion). Insertions in `other`
    are appended to the preceding pivot slot."""
    n, m = len(pivot), len(other)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1,
                           dp[i - 1, j - 1] + (pivot[i - 1] != other[j - 1]))
    slots = [""] * (n + 1)   # slot n is the tail
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + (pivot[i - 1] != other[j - 1]):
            slots[i - 1] = other[j - 1] + slots[i - 1]
            i, j = i - 1, j - 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            slots[i] = other[j - 1] + slots[i]   # insertion -> attach to slot i
            j -= 1
        else:
            slots[i - 1] = ""                     # deletion
            i -= 1
    return list(enumerate(slots))


def rover(candidates: List[str], weights: List[float]) -> str:
    """Weighted per-position character vote against the best pivot."""
    cands = [c for c in candidates if isinstance(c, str)]
    if not cands:
        return ""
    if len(cands) == 1:
        return cands[0]

    # Pivot = candidate with the lowest weighted mean distance to the others.
    costs = []
    for i, c in enumerate(cands):
        d = sum(w * _rel_cer(c, o) for j, (o, w) in enumerate(zip(cands, weights)) if j != i)
        costs.append(d)
    p_idx = int(np.argmin(costs))
    pivot = cands[p_idx]

    votes: Dict[int, Counter] = defaultdict(Counter)
    for c, w in zip(cands, weights):
        for pos, ch in align_to_pivot(pivot, c):
            votes[pos][ch] += w

    out = []
    for pos in range(len(pivot) + 1):
        v = votes.get(pos)
        if not v:
            continue
        best = max(v.items(), key=lambda kv: (kv[1], kv[0] != ""))
        out.append(best[0])
    return "".join(out)


def mbr(candidates: List[str], weights: List[float]) -> str:
    """Minimum Bayes Risk under CER: pick the candidate with the lowest
    weighted expected CER against the rest."""
    cands = [c for c in candidates if isinstance(c, str)]
    if not cands:
        return ""
    risks = [
        sum(w * _rel_cer(c, o) for j, (o, w) in enumerate(zip(cands, weights)) if j != i)
        for i, c in enumerate(cands)
    ]
    return cands[int(np.argmin(risks))]


# --------------------------------------------------------------------------
def parse_arm_weights(s: str | None) -> Dict[str, float]:
    if not s:
        return {}
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def combine(
    candidate_csvs: List[str],
    method: str = "rover",
    arm_weights: Dict[str, float] | None = None,
    use_nbest: int = 1,
    lexicon: TrainLexicon | None = None,
    lex_alpha: float = 1.0,
    lex_beta: float = 0.0,
) -> pd.DataFrame:
    arm_weights = arm_weights or {}
    frames = []
    for p in candidate_csvs:
        df = pd.read_csv(p, dtype={"ID": str}, keep_default_na=False)
        if "rank" not in df.columns:
            df["rank"] = 0
        if "score" not in df.columns:
            df["score"] = 0.0
        if "arm" not in df.columns:
            df["arm"] = os.path.splitext(os.path.basename(p))[0]
        frames.append(df[["ID", "rank", "Target", "score", "arm"]])
    cand = pd.concat(frames, ignore_index=True)
    cand = cand[cand["rank"] < use_nbest]

    rows = []
    for rid, g in cand.groupby("ID"):
        texts = g["Target"].astype(str).tolist()
        ws = [arm_weights.get(a, 1.0) * (1.0 / (1 + r))
              for a, r in zip(g["arm"], g["rank"])]

        if method == "rover":
            text = rover(texts, ws)
        elif method == "mbr":
            text = mbr(texts, ws)
        elif method == "lexicon":
            if lexicon is None:
                raise ValueError("method=lexicon requires --train_csv for the lexicon")
            text = lexicon_rescore(list(zip(texts, g["score"].astype(float).tolist())),
                                   lexicon, lex_alpha, lex_beta)
        else:
            raise ValueError(f"unknown method {method}")

        if lexicon is not None and method != "lexicon" and lex_beta > 0:
            # Guard: if fusion produced a string with much worse train-lexicon
            # coverage than the best single candidate, fall back to that candidate.
            best_single = max(texts, key=lexicon.coverage)
            if lexicon.coverage(text) + 1e-9 < lexicon.coverage(best_single) - lex_beta:
                text = best_single

        rows.append({"ID": rid, "Target": " ".join(str(text).split())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="rover", choices=["rover", "mbr", "lexicon"])
    ap.add_argument("--arm_weights", default=None, help="e.g. trocr=1.0,vlm=1.3,ctc=0.7")
    ap.add_argument("--nbest", type=int, default=1)
    ap.add_argument("--train_csv", default=None, help="Enables train-lexicon guard/rescore")
    ap.add_argument("--lex_alpha", type=float, default=1.0)
    ap.add_argument("--lex_beta", type=float, default=0.0)
    a = ap.parse_args()

    lex = TrainLexicon.fit(a.train_csv) if a.train_csv else None
    out = combine(a.candidates, a.method, parse_arm_weights(a.arm_weights),
                  a.nbest, lex, a.lex_alpha, a.lex_beta)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, index=False)
    log.info("ensemble method=%s rows=%d -> %s", a.method, len(out), a.out)


if __name__ == "__main__":
    main()
