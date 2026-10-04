"""
rover.py -- Word-level ROVER (Recognizer Output Voting Error Reduction).

Core primitive used by both:
  - pseudo_label_filter_v2.py  (builds a better pseudo-label TARGET than a
    raw single-model top-1, by recombining agreeing/disagreeing hypotheses
    instead of hard-accepting/rejecting them)
  - confusion_network.py       (reuses the same alignment machinery but
    swaps majority-vote for MBR-style risk minimization per slot)

Implementation notes:
  - This is a practical, incremental-profile implementation: align
    hypothesis 1 into an empty profile, then align each subsequent
    hypothesis into the growing profile. This is what real ROVER
    implementations (NIST SCTK) do. True N-way multiple alignment is
    NP-hard; incremental/profile alignment is the standard approximation
    (alignment ORDER can matter at the margins -- feed your most reliable
    hypothesis first if you have a strong prior about which one that is).
  - Each hypothesis carries ONE scalar confidence. At every aligned column,
    a hypothesis that has a real token there votes for that token with its
    own confidence; a hypothesis that has a GAP there (because some other
    hypothesis inserted/deleted a word) votes for deletion with that SAME
    confidence -- gaps are not free, they cost votes too. This matters: an
    earlier draft of this that zeroed out gap-votes let any single
    hypothesis's insertion win by default, since nothing competed with it.
"""
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple

GAP = None  # sentinel for "no token aligned here"


def align_two(seq_a: Sequence[str], seq_b: Sequence[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    """Global (Needleman-Wunsch) alignment of two token sequences.

    Returns a list of (tok_a_or_None, tok_b_or_None) pairs.
    Substitution cost = 0 if tokens equal else 1. Gap cost = 1.
    """
    a, b = list(seq_a), list(seq_b)
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub_cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    pairs: List[Tuple[Optional[str], Optional[str]]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub_cost = 0 if a[i - 1] == b[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + sub_cost:
                pairs.append((a[i - 1], b[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((a[i - 1], None))
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            pairs.append((None, b[j - 1]))
            j -= 1
            continue
        raise RuntimeError("alignment backtrace failed (should not happen)")
    pairs.reverse()
    return pairs


class Profile:
    """A growing multi-way alignment: columns of tokens-or-GAP, one slot per
    hypothesis merged so far, plus each hypothesis's own scalar confidence
    (used as the vote weight for whatever that hypothesis has in each column,
    including a GAP)."""

    def __init__(self):
        self.columns: List[List[Optional[str]]] = []
        self.hyp_conf: List[float] = []

    @property
    def n_hyps(self) -> int:
        return len(self.hyp_conf)

    def _column_vote(self, tok: Optional[str], hyp_confidences: Sequence[float]) -> Optional[str]:
        raise NotImplementedError  # replaced by module-level vote fns; kept for clarity

    def add(self, tokens: Sequence[str], confidence: float):
        if self.n_hyps == 0:
            self.columns = [[t] for t in tokens]
            self.hyp_conf = [confidence]
            return

        rep_seq: List[str] = []
        rep_col_idx: List[int] = []
        for idx, col in enumerate(self.columns):
            tok = majority_vote(col, self.hyp_conf)
            if tok is not None:
                rep_seq.append(tok)
                rep_col_idx.append(idx)

        pairs = align_two(rep_seq, list(tokens))

        new_columns: List[List[Optional[str]]] = []
        rep_ptr = 0
        orig_ptr = 0
        pair_i = 0

        def emit_original(idx):
            new_columns.append(self.columns[idx] + [None])

        def emit_new_insertion(new_tok):
            new_columns.append([None] * self.n_hyps + [new_tok])

        def emit_matched(idx, new_tok):
            new_columns.append(self.columns[idx] + [new_tok])

        while orig_ptr < len(self.columns) or pair_i < len(pairs):
            next_rep_col = rep_col_idx[rep_ptr] if rep_ptr < len(rep_col_idx) else len(self.columns)

            if orig_ptr < next_rep_col:
                emit_original(orig_ptr)
                orig_ptr += 1
                continue

            if rep_ptr < len(rep_col_idx):
                while pairs[pair_i][0] is None:
                    emit_new_insertion(pairs[pair_i][1])
                    pair_i += 1
                rep_tok, new_tok = pairs[pair_i]
                assert rep_tok == rep_seq[rep_ptr], "alignment bookkeeping bug"
                emit_matched(next_rep_col, new_tok)
                pair_i += 1
                rep_ptr += 1
                orig_ptr = next_rep_col + 1
                continue

            if pair_i < len(pairs):
                assert pairs[pair_i][0] is None
                emit_new_insertion(pairs[pair_i][1])
                pair_i += 1
                continue
            break

        self.columns = new_columns
        self.hyp_conf.append(confidence)

    def collapse(self, vote_fn=None) -> str:
        vote_fn = vote_fn or majority_vote
        out = []
        for col in self.columns:
            tok = vote_fn(col, self.hyp_conf)
            if tok is not None:
                out.append(tok)
        return " ".join(out)


def majority_vote(col_tokens: Sequence[Optional[str]], hyp_conf: Sequence[float]) -> Optional[str]:
    """Confidence-weighted majority vote for one column. GAP is a real
    candidate (a hypothesis with a gap here is voting to delete, weighted
    by that hypothesis's own confidence). Ties prefer a real word over GAP."""
    tally = {}
    for tok, w in zip(col_tokens, hyp_conf):
        tally[tok] = tally.get(tok, 0.0) + w
    return max(tally.items(), key=lambda kv: (kv[1], kv[0] is not None))[0]


def rover_combine(hyps: List[str], confidences: Optional[List[float]] = None) -> str:
    """Combine >=1 hypothesis strings via word-level ROVER majority voting."""
    hyps = [h for h in hyps if h is not None and h != ""]
    if not hyps:
        return ""
    if len(hyps) == 1:
        return hyps[0]
    if confidences is None:
        confidences = [1.0] * len(hyps)
    prof = Profile()
    for h, c in zip(hyps, confidences):
        prof.add(h.split(), c)
    return prof.collapse(majority_vote)
