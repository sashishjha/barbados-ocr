"""
Automated, generalisable post-processing. Every rule is FITTED ON TRAIN
TRANSCRIPTIONS ONLY and applied uniformly to all rows - no per-image logic, no
hand-written fixes aimed at specific test images. This is what the organizers'
pseudo-labelling ruling requires ("context-based 0->O substitution" style rules
are fine; per-image hardcoded fixes are not).

Components
----------
1. TrainLexicon      - vocabulary, character unigram/bigram stats and a word
                       frequency table, all derived from Train.csv.
2. ConfusionRules    - character substitutions (0/O, 1/l, f/long-s, etc.)
                       accepted only when the substitution turns an
                       out-of-lexicon token into an in-lexicon token that is
                       frequent enough in train.
3. spacing_normalize - whitespace normalization matched to the scorer's
                       tokenization (whitespace split).
4. lexicon_rescore   - for a set of candidates, prefer the one with the best
                       combination of model score and train-lexicon coverage.

DELIBERATE CHOICE: the lexicon is built exclusively from training
transcriptions. An external English/legal dictionary is grey under
"provided dataset only", so we do not use one.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.eval.edit_distance import levenshtein

_WS = re.compile(r"\s+")

# Visually confusable pairs typical of 18th/19th-c. secretary hand and of OCR
# on faded iron-gall ink. Bidirectional; each is only APPLIED when it improves
# train-lexicon membership, so the list being broad is safe.
CANDIDATE_CONFUSIONS: List[Tuple[str, str]] = [
    ("0", "O"), ("O", "0"), ("1", "l"), ("l", "1"), ("1", "I"), ("I", "1"),
    ("5", "S"), ("S", "5"), ("8", "B"), ("B", "8"), ("6", "b"), ("b", "6"),
    ("f", "s"), ("s", "f"),          # long-s (ſ) misread as f
    ("rn", "m"), ("m", "rn"),
    ("cl", "d"), ("d", "cl"),
    ("ii", "u"), ("u", "ii"),
    ("vv", "w"), ("w", "vv"),
    ("nn", "m"), ("y", "g"), ("g", "y"),
    ("c", "e"), ("e", "c"), ("t", "f"), ("i", "j"),
]


@dataclass
class TrainLexicon:
    words: Counter = field(default_factory=Counter)
    charset: set = field(default_factory=set)
    total_words: int = 0

    @classmethod
    def fit(cls, train_csv: str, lowercase_key: bool = True) -> "TrainLexicon":
        df = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
        lex = cls()
        for t in df["Target"].fillna(""):
            lex.charset |= set(t)
            for w in str(t).split():
                lex.words[w.lower() if lowercase_key else w] += 1
        lex.total_words = sum(lex.words.values())
        return lex

    def contains(self, word: str) -> bool:
        return word.lower() in self.words

    def freq(self, word: str) -> int:
        return self.words.get(word.lower(), 0)

    def coverage(self, text: str) -> float:
        toks = str(text).split()
        if not toks:
            return 0.0
        return sum(1 for w in toks if self.contains(w)) / len(toks)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"words": dict(self.words), "charset": sorted(self.charset),
                       "total_words": self.total_words}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "TrainLexicon":
        with open(path) as f:
            d = json.load(f)
        lex = cls(Counter(d["words"]), set(d["charset"]), d["total_words"])
        return lex


@dataclass
class PostprocessConfig:
    enabled: bool = True
    spacing: bool = True
    confusion: bool = True
    min_word_freq: int = 2       # target word must appear >= this often in train
    max_edits_per_word: int = 1  # only single-substitution repairs
    max_word_len_for_fix: int = 24
    strip_prompt_echo: bool = True


PROMPT_ECHO = re.compile(
    r"^(transcribe exactly as seen[.:]?|assistant[:]?|user[:]?|verbatim transcription[:]?)\s*",
    flags=re.IGNORECASE,
)


def spacing_normalize(text: str) -> str:
    """Match the scorer's whitespace tokenization: collapse runs, strip ends,
    and remove spaces immediately before common punctuation."""
    s = _WS.sub(" ", str(text)).strip()
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s


def strip_prompt_echo(text: str) -> str:
    """VLMs occasionally echo the instruction. Purely structural, never
    content-specific."""
    s = str(text)
    for _ in range(3):
        new = PROMPT_ECHO.sub("", s).strip()
        if new == s:
            break
        s = new
    return s


class ConfusionRules:
    """Applies confusion-pair repairs only when they move a token INTO the
    train lexicon. This is the 'generalisable, context-based' form the
    organizers explicitly allowed."""

    def __init__(self, lexicon: TrainLexicon, cfg: PostprocessConfig):
        self.lex = lexicon
        self.cfg = cfg

    def fix_word(self, word: str) -> Tuple[str, Optional[str]]:
        if self.lex.contains(word) or len(word) > self.cfg.max_word_len_for_fix:
            return word, None

        core = word.strip(".,;:()\"'[]")
        prefix = word[: len(word) - len(word.lstrip(".,;:()\"'[]"))]
        suffix = word[len(prefix) + len(core):]
        if not core or self.lex.contains(core):
            return word, None

        best, best_freq, best_rule = None, self.cfg.min_word_freq - 1, None
        for a, b in CANDIDATE_CONFUSIONS:
            if a not in core:
                continue
            # try each occurrence individually (single-substitution repairs)
            start = 0
            while True:
                idx = core.find(a, start)
                if idx < 0:
                    break
                cand = core[:idx] + b + core[idx + len(a):]
                start = idx + 1
                if levenshtein(list(core), list(cand)) > self.cfg.max_edits_per_word:
                    continue
                f = self.lex.freq(cand)
                if f > best_freq:
                    best, best_freq, best_rule = cand, f, f"{a}->{b}"
        if best is None:
            return word, None
        return prefix + best + suffix, best_rule

    def apply(self, text: str) -> Tuple[str, List[str]]:
        out, rules = [], []
        for w in str(text).split():
            fixed, rule = self.fix_word(w)
            out.append(fixed)
            if rule:
                rules.append(f"{w}->{fixed}({rule})")
        return " ".join(out), rules


def build(train_csv: str, cfg: Optional[PostprocessConfig] = None):
    cfg = cfg or PostprocessConfig()
    lex = TrainLexicon.fit(train_csv)
    return lex, ConfusionRules(lex, cfg), cfg


def apply_all(text: str, lex: TrainLexicon, rules: ConfusionRules,
              cfg: PostprocessConfig) -> Tuple[str, List[str]]:
    if not cfg.enabled:
        return str(text), []
    s = str(text)
    applied: List[str] = []
    if cfg.strip_prompt_echo:
        s2 = strip_prompt_echo(s)
        if s2 != s:
            applied.append("prompt_echo_stripped")
        s = s2
    if cfg.spacing:
        s = spacing_normalize(s)
    if cfg.confusion:
        s, r = rules.apply(s)
        applied.extend(r)
    return s, applied


def lexicon_rescore(candidates: List[Tuple[str, float]], lex: TrainLexicon,
                    alpha: float = 1.0, beta: float = 0.5) -> str:
    """Pick the candidate maximising `alpha*normalized_model_score +
    beta*train_lexicon_coverage`. Beam scores are length-normalized log-probs,
    so we min-max them within the candidate set before combining."""
    if not candidates:
        return ""
    scores = [s for _, s in candidates]
    lo, hi = min(scores), max(scores)
    rng = (hi - lo) or 1.0
    best, best_v = candidates[0][0], -1e18
    for text, sc in candidates:
        v = alpha * ((sc - lo) / rng) + beta * lex.coverage(text)
        if v > best_v:
            best, best_v = text, v
    return best


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--train_csv", default="data/derived_checkpoint/train_clean.csv")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--audit_csv", default=None)
    ap.add_argument("--no_confusion", action="store_true")
    ap.add_argument("--min_word_freq", type=int, default=2)
    a = ap.parse_args()

    cfg = PostprocessConfig(confusion=not a.no_confusion, min_word_freq=a.min_word_freq)
    lex, rules, cfg = build(a.train_csv, cfg)
    print(f"[POST] train lexicon: {len(lex.words)} unique words, "
          f"{len(lex.charset)} chars")

    df = pd.read_csv(a.pred_csv, dtype=str, keep_default_na=False)
    audit = []
    fixed = []
    for _, r in df.iterrows():
        new, applied = apply_all(r["Target"], lex, rules, cfg)
        fixed.append(new)
        if applied:
            audit.append({"ID": r["ID"], "before": r["Target"], "after": new,
                          "rules": "; ".join(applied)})
    df["Target"] = fixed

    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    df.to_csv(a.out_csv, index=False)
    print(f"[POST] rows changed: {len(audit)}/{len(df)} -> {a.out_csv}")
    if a.audit_csv:
        pd.DataFrame(audit).to_csv(a.audit_csv, index=False)
        print(f"[POST] audit -> {a.audit_csv}")


if __name__ == "__main__":
    main()
