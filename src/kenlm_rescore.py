#!/usr/bin/env python3
"""
kenlm_rescore.py -- KenLM-based N-best rescoring for OCR/HTR line hypotheses.

This is a corrected version of the original draft. Fixes applied, each tied
to a specific problem raised in cross-verification:

  1. CONSISTENT LOG BASE. kenlm's .score()/.full_scores() return log10
     probabilities. Every LM score is converted to natural log (nats) at the
     point it's read, via LN10, so it can be summed with acoustic scores on
     one consistent scale instead of silently relying on alpha to absorb an
     unexamined ln(10) ~= 2.303 factor.

  2. PER-SOURCE-MODEL ACOUSTIC CALIBRATION. Qwen-VL and TrOCR acoustic
     scores are not on the same scale (different tokenizers, different
     vocab granularity, different generation-time normalization). Pooling
     them with a raw max() lets whichever arm happens to produce less
     negative numbers dominate the fused ranking regardless of correctness.
     `calibrate` fits a per-arm z-score (mu, sigma) so every arm's
     contribution is normalized before comparison.

  3. LENGTH NORMALIZATION OF THE ACOUSTIC TERM. Raw summed log-likelihood
     favors shorter candidates. The acoustic score is normalized by output
     character length (tokenizer-agnostic across arms) before z-scoring,
     so the length effect on the *LM* term (which `beta` is meant to
     correct) isn't tangled up with a separate length effect on the
     *acoustic* term.

  4. HYBRID WORD/CHAR LM SCORING (OOV GATING). Historical ledger text is
     full of rare-but-correct proper nouns (surnames, place names) that a
     ~5K-line word LM has seen 0-1 times. Scoring OOV words with kenlm's
     flat <unk> cost means the LM will happily "correct" a rare-but-right
     name into a common-but-wrong word. `LMScorer.hybrid_word_score`
     detects OOV words via `full_scores()`'s oov flag and substitutes a
     character-level estimate for just that word instead.

  5. A SWAP GUARD. The fused ranking may only override the acoustic-only
     best candidate for a line if the replacement is textually close to it
     (SequenceMatcher ratio) or the fused-score margin -- normalized to
     nats-per-word so it's comparable across line lengths -- clears an
     explicit minimum. This stops the LM from silently trading a correct
     rare token for a fluent wrong one.

  6. HARD FAILURES instead of silent bad defaults: missing `score` column,
     NaN `Target`, or mismatched ID sets across candidate files now raise
     (or require an explicit --allow_id_mismatch opt-in) instead of
     quietly producing a biased or partial fusion.

  7. A BREVITY GUARD against degenerate very-short winners: a non-best
     candidate is only eligible to win if its length isn't a large outlier
     relative to the group's median, unless it's the acoustic-best itself.

  8. SHARED NFC UNICODE NORMALIZATION between corpus-building and
     rescoring, so train-corpus text and inference-time candidate text
     can't silently drift apart in tokenization.

  9. `tune` FITS PARAMETERS AGAINST THE ACTUAL COMPETITION METRIC
     (0.7*(1-CER) + 0.3*(1-WER), each clipped to [0,1] -- confirm this
     matches your exact metric definition before trusting the numbers),
     evaluated on a held-out val set with known references, not against
     LM perplexity. It also reports an acoustic-only baseline so you can
     see whether LM rescoring is actually helping before trusting it.

 10. Output carries every component score (acoustic, LM word/char, guard
     status, normalized margin) for auditability, and because the margin
     is a ready-made confidence signal for a transductive pseudo-labeling
     loop.

 11. Caching (duplicate strings across arms' N-best lists are common)
     and optional multiprocessing across the (expensive, kenlm-touching)
     component-scoring step -- separated from the (cheap, pure-arithmetic)
     param-combination step so Optuna trials don't re-invoke kenlm.

Candidate CSV contract: every file passed to --candidates must have columns
  ID, Target, score, arm
where `score` is that row's acoustic log-probability from whichever arm
produced it, and `arm` identifies the arm (e.g. "qwen", "trocr") so scores
can be calibrated per source.

Usage:
  # 1. Build LMs from train text only.
  python kenlm_rescore.py train --train_csv data/train.csv \
      --word_order 3 --char_order 10 --out_dir data

  # 2. (Recommended first) sweep orders by held-out perplexity on TRAIN
  #    text only -- never val/test -- before trusting a default order.
  python kenlm_rescore.py select_order --text_file data/corpus_words.txt \
      --orders 2 3 4 --out_prefix data/word_sweep

  # 3. Fit per-arm acoustic calibration from a candidates pool.
  python kenlm_rescore.py calibrate \
      --candidates preds/qwen_val_nbest.csv preds/trocr_val_nbest.csv \
      --out data/calibration.json

  # 4. Tune alpha_word / alpha_char / beta / guard params against the
  #    blended metric on a val set with known ground truth.
  python kenlm_rescore.py tune \
      --candidates preds/qwen_val_nbest.csv preds/trocr_val_nbest.csv \
      --refs data/val_refs.csv \
      --word_lm data/word_3gram.bin --char_lm data/char_10gram.bin \
      --calibration data/calibration.json --out data/best_params.json

  # 5. Apply to test.
  python kenlm_rescore.py rescore \
      --candidates preds/qwen_test_nbest.csv preds/trocr_test_nbest.csv \
      --word_lm data/word_3gram.bin --char_lm data/char_10gram.bin \
      --calibration data/calibration.json --params data/best_params.json \
      --out preds/rescored_test.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
from difflib import SequenceMatcher
from multiprocessing import Pool
from typing import Optional

import pandas as pd

try:
    import kenlm
except ImportError:
    kenlm = None

try:
    import optuna
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

LN10 = math.log(10.0)
REQUIRED_CAND_COLS = {"ID", "Target", "score", "arm"}


# --------------------------------------------------------------------------
# Text normalization / edit-distance primitives
# --------------------------------------------------------------------------

def normalize_text(text) -> str:
    """Single shared normalization used at corpus-build time AND rescoring
    time. Keep this identical in both places, or the LM's vocabulary and
    the candidate text will silently drift apart (case, whitespace, Unicode
    composition of accented/superscript characters used in abbreviations)."""
    import unicodedata
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.split())


def levenshtein(a: list, b: list) -> int:
    """Standard O(n*m) DP edit distance over two sequences (chars or words)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def cer(ref: str, hyp: str) -> float:
    ref_chars, hyp_chars = list(ref), list(hyp)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return levenshtein(ref_chars, hyp_chars) / len(ref_chars)


def wer(ref: str, hyp: str) -> float:
    ref_words, hyp_words = ref.split(), hyp.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)


def blended_metric(ref: str, hyp: str, cer_weight: float = 0.7, wer_weight: float = 0.3) -> float:
    """0.7*(1-CER) + 0.3*(1-WER), each clipped to [0,1] before combining so a
    single catastrophic line can't drag the average negative. Confirm this
    matches the competition's exact metric definition (in particular how it
    handles CER/WER > 1) before trusting the tuned parameters."""
    c = min(cer(ref, hyp), 1.0)
    w = min(wer(ref, hyp), 1.0)
    return cer_weight * (1.0 - c) + wer_weight * (1.0 - w)


def text_similarity(a: str, b: str) -> float:
    """1.0 = identical. Cheap proxy used only by the swap guard; the actual
    scored metric always uses the exact `cer`/`wer` edit distance above."""
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


# --------------------------------------------------------------------------
# Corpus prep / LM training / order selection
# --------------------------------------------------------------------------

def prepare_corpus(train_csv: str, word_txt: str, char_txt: str, text_col: str = "Target") -> int:
    df = pd.read_csv(train_csv)
    texts = [normalize_text(t) for t in df[text_col].dropna().astype(str).tolist()]
    texts = [t for t in texts if t]
    if not texts:
        raise ValueError(f"No usable text found in {train_csv!r} column {text_col!r}.")

    os.makedirs(os.path.dirname(word_txt) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(char_txt) or ".", exist_ok=True)
    with open(word_txt, "w", encoding="utf-8") as fw, open(char_txt, "w", encoding="utf-8") as fc:
        for text in texts:
            fw.write(text + "\n")
            char_seq = " ".join(list(text.replace(" ", "<space>")))
            fc.write(char_seq + "\n")

    print(f"[prepare_corpus] {len(texts)} usable lines written from {train_csv}")
    return len(texts)


def build_lm(text_file: str, out_bin: str, order: int,
             lmplz_path: str = "lmplz", build_binary_path: str = "build_binary") -> str:
    """NOTE: kenlm's lmplz implements exactly one smoothing method --
    modified, interpolated Kneser-Ney. There is no Witten-Bell option in
    this tool; getting literal Witten-Bell would mean switching to SRILM or
    IRSTLM. --discount_fallback lets lmplz substitute generic discount
    constants when a small corpus doesn't have enough count-of-count
    statistics at a given order to estimate them from data -- this avoids a
    crash but is a sign that order may be too high for the corpus size; see
    `select_order`."""
    out_arpa = out_bin.replace(".bin", ".arpa")
    print(f"[build_lm] order={order} -> {out_arpa}")
    with open(text_file, "r", encoding="utf-8") as f_in, open(out_arpa, "w", encoding="utf-8") as f_out:
        cmd = [lmplz_path, "-o", str(order), "--discount_fallback"]
        subprocess.run(cmd, stdin=f_in, stdout=f_out, check=True)
    print(f"[build_lm] converting to binary -> {out_bin}")
    subprocess.run([build_binary_path, out_arpa, out_bin], check=True)
    return out_bin


def heldout_split(text_file: str, frac: float = 0.1, seed: int = 13):
    """Split a one-line-per-sentence corpus into train/heldout files, for
    perplexity-based order selection. Always run this on TRAIN text only --
    never on val or test text -- or order selection leaks test-distribution
    information into the arm choice."""
    with open(text_file, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(lines)
    cut = max(1, int(len(lines) * frac))
    heldout, train = lines[:cut], lines[cut:]
    train_path, heldout_path = text_file + ".ho_train", text_file + ".ho_heldout"
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train) + "\n")
    with open(heldout_path, "w", encoding="utf-8") as f:
        f.write("\n".join(heldout) + "\n")
    return train_path, heldout_path


def evaluate_perplexity(lm_bin_path: str, heldout_txt: str) -> float:
    if kenlm is None:
        raise RuntimeError("kenlm is not installed in this environment.")
    arm = kenlm.Model(lm_bin_path)
    with open(heldout_txt, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    if not lines:
        raise ValueError(f"{heldout_txt} has no usable lines.")
    total_logprob10, total_words = 0.0, 0
    for line in lines:
        total_logprob10 += arm.score(line, bos=True, eos=True)
        total_words += len(line.split()) + 1  # +1 for </s>
    return 10.0 ** (-total_logprob10 / total_words)


def select_order(text_file: str, orders: list, out_prefix: str, frac: float = 0.1, seed: int = 13,
                  lmplz_path: str = "lmplz", build_binary_path: str = "build_binary"):
    train_path, heldout_path = heldout_split(text_file, frac=frac, seed=seed)
    results = {}
    for order in orders:
        tmp_bin = f"{out_prefix}.order{order}.tmp.bin"
        build_lm(train_path, tmp_bin, order, lmplz_path, build_binary_path)
        ppl = evaluate_perplexity(tmp_bin, heldout_path)
        results[order] = ppl
        print(f"[select_order] order={order} heldout_perplexity={ppl:.3f}")
    best_order = min(results, key=results.get)
    print(f"[select_order] best order by heldout perplexity: {best_order}")
    return best_order, results


# --------------------------------------------------------------------------
# Candidate loading with hard validation (no silent bad defaults)
# --------------------------------------------------------------------------

def load_candidates(paths: list, allow_id_mismatch: bool = False) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype={"ID": str})
        missing = REQUIRED_CAND_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"{p} is missing required column(s) {sorted(missing)}. Every candidates "
                f"CSV must include ID, Target, score, and arm (arm = which arm "
                f"produced the row, e.g. 'qwen' / 'trocr'). Silently defaulting a "
                f"missing score to 0.0 would bias the fused ranking toward whichever "
                f"file forgot to include scores -- refusing to guess."
            )
        n_before = len(df)
        df = df.dropna(subset=["Target"]).copy()
        if len(df) < n_before:
            print(f"[load_candidates] {p}: dropped {n_before - len(df)} row(s) with NaN Target.")
        df["Target"] = df["Target"].astype(str).map(normalize_text)
        df = df[df["Target"] != ""]
        frames.append(df)

    cand = pd.concat(frames, ignore_index=True)

    id_sets = [set(pd.read_csv(p, dtype={"ID": str})["ID"]) for p in paths]
    common = set.intersection(*id_sets) if id_sets else set()
    union = set.union(*id_sets) if id_sets else set()
    if common != union:
        gaps = {p: sorted(union - s)[:5] for p, s in zip(paths, id_sets) if union - s}
        msg = (
            "ID sets differ across candidate files -- fusion for the mismatched IDs "
            "would silently run on a partial candidate pool instead of pooling all "
            f"arms. Examples of IDs missing per file (first 5 shown): {gaps}."
        )
        if allow_id_mismatch:
            print(f"[load_candidates] WARNING: {msg} Proceeding because --allow_id_mismatch was set.")
        else:
            raise ValueError(msg + " Fix the ID export, or pass --allow_id_mismatch if this is expected.")

    return cand


# --------------------------------------------------------------------------
# Per-source-arm acoustic calibration
# --------------------------------------------------------------------------

def fit_calibration(cand: pd.DataFrame) -> dict:
    """Per source-arm z-score stats over length-normalized acoustic scores
    (raw score / max(1, char length of the text)). Fit independently per
    value of the 'arm' column so heterogeneous arms (different
    tokenizers, different confidence calibration) don't dominate the fused
    ranking purely by scale."""
    stats = {}
    tmp = cand.copy()
    tmp["_norm_score"] = tmp["score"].astype(float) / tmp["Target"].str.len().clip(lower=1)
    for arm_name, g in tmp.groupby("arm"):
        mu = float(g["_norm_score"].mean())
        sigma = float(g["_norm_score"].std(ddof=0))
        if sigma < 1e-8:
            sigma = 1.0  # degenerate-distribution guard
        stats[arm_name] = {"mu": mu, "sigma": sigma}
        print(f"[calibrate] arm={arm_name!r} n={len(g)} mu={mu:.4f} sigma={sigma:.4f}")
    return stats


def calibrated_acoustic(row, calib: dict) -> float:
    norm = float(row["score"]) / max(1, len(row["Target"]))
    st = calib.get(row["arm"])
    if st is None:
        raise ValueError(
            f"No calibration stats for arm {row['arm']!r}; re-run `calibrate` on a "
            f"candidate pool that includes this arm."
        )
    return (norm - st["mu"]) / st["sigma"]


# --------------------------------------------------------------------------
# LM scoring: consistent log base + hybrid word/char OOV gating
# --------------------------------------------------------------------------

class LMScorer:
    """Wraps a word-level and char-level kenlm arm. Converts every score
    to natural log (nats) at the boundary, and substitutes a char-level
    estimate for OOV words in `hybrid_word_score` instead of eating kenlm's
    flat <unk> cost -- see module docstring, fix #4."""

    def __init__(self, word_lm_path: Optional[str], char_lm_path: Optional[str]):
        if kenlm is None:
            raise RuntimeError("kenlm is not installed in this environment.")
        self.word_arm = kenlm.Model(word_lm_path) if word_lm_path else None
        self.char_arm = kenlm.Model(char_lm_path) if char_lm_path else None
        self._char_word_cache: dict = {}
        self._char_line_cache: dict = {}
        self._hybrid_cache: dict = {}

    def _char_word_score(self, word: str) -> float:
        """Score a single word under the char-level LM, no sentence
        boundary context -- just 'how plausible are these letters
        together'."""
        if word not in self._char_word_cache:
            if self.char_arm is None:
                self._char_word_cache[word] = 0.0
            else:
                seq = " ".join(list(word))
                self._char_word_cache[word] = self.char_arm.score(seq, bos=False, eos=False) * LN10
        return self._char_word_cache[word]

    def char_line_score(self, text: str) -> float:
        if text not in self._char_line_cache:
            if self.char_arm is None:
                self._char_line_cache[text] = 0.0
            else:
                seq = " ".join(list(text.replace(" ", "<space>")))
                self._char_line_cache[text] = self.char_arm.score(seq, bos=True, eos=True) * LN10
        return self._char_line_cache[text]

    def hybrid_word_score(self, text: str) -> float:
        if text in self._hybrid_cache:
            return self._hybrid_cache[text]
        if self.word_arm is None:
            self._hybrid_cache[text] = 0.0
            return 0.0
        words = text.split()
        if not words:
            self._hybrid_cache[text] = 0.0
            return 0.0
        full = list(self.word_arm.full_scores(text, bos=True, eos=True))
        # `full` has one entry per word, plus (since eos=True) one trailing
        # entry scoring the </s> transition.
        total = 0.0
        for i, w in enumerate(words):
            prob10, _ngram_len, is_oov = full[i]
            total += self._char_word_score(w) if is_oov else prob10 * LN10
        if len(full) > len(words):
            total += full[-1][0] * LN10  # </s> contribution
        self._hybrid_cache[text] = total
        return total


# --------------------------------------------------------------------------
# Component computation (expensive, touches kenlm) vs. combination (cheap,
# pure arithmetic) -- kept separate so Optuna trials don't re-invoke kenlm.
# --------------------------------------------------------------------------

_cw_scorer: Optional[LMScorer] = None
_cw_calib: Optional[dict] = None


def _init_component_worker(word_lm_path, char_lm_path, calib):
    global _cw_scorer, _cw_calib
    _cw_scorer = LMScorer(word_lm_path, char_lm_path)
    _cw_calib = calib


def _component_worker(chunk_records: list) -> list:
    rows = []
    for r in chunk_records:
        acoustic = calibrated_acoustic(r, _cw_calib)
        lm_word = _cw_scorer.hybrid_word_score(r["Target"])
        lm_char = _cw_scorer.char_line_score(r["Target"])
        rows.append({
            **r,
            "acoustic_calibrated": acoustic,
            "lm_word_nats": lm_word,
            "lm_char_nats": lm_char,
            "n_words": len(r["Target"].split()),
        })
    return rows


def compute_components(cand: pd.DataFrame, calib: dict, word_lm_path: str, char_lm_path: str,
                        n_jobs: int = 1) -> pd.DataFrame:
    records = cand.to_dict("records")
    if n_jobs <= 1:
        _init_component_worker(word_lm_path, char_lm_path, calib)
        rows = _component_worker(records)
    else:
        chunk_size = max(1, len(records) // n_jobs)
        chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
        with Pool(processes=n_jobs, initializer=_init_component_worker,
                  initargs=(word_lm_path, char_lm_path, calib)) as pool:
            chunked = pool.map(_component_worker, chunks)
        rows = [r for chunk in chunked for r in chunk]
    return pd.DataFrame(rows)


def combine_and_select(components: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Pure arithmetic + guard logic, no kenlm calls -- cheap enough to run
    once per Optuna trial. This is the ONLY place the fused score and the
    swap guard are computed, so `tune` and `rescore` can never drift apart."""
    alpha_word = params["alpha_word"]
    alpha_char = params["alpha_char"]
    beta = params["beta"]
    guard_min_margin = params["guard_min_margin"]       # nats per word
    guard_similarity = params["guard_similarity"]        # SequenceMatcher ratio
    brevity_floor = params["brevity_floor"]              # fraction of group median length

    results = []
    for rid, g in components.groupby("ID"):
        rows = g.to_dict("records")
        for r in rows:
            r["total_score"] = (
                r["acoustic_calibrated"]
                + alpha_word * r["lm_word_nats"]
                + alpha_char * r["lm_char_nats"]
                + beta * r["n_words"]
            )

        acoustic_best = max(rows, key=lambda r: r["acoustic_calibrated"])

        lengths = sorted(r["n_words"] for r in rows)
        median_len = lengths[len(lengths) // 2] if lengths else 0
        eligible = [
            r for r in rows
            if r is acoustic_best or median_len == 0 or r["n_words"] >= brevity_floor * median_len
        ]
        if not eligible:
            eligible = rows
        fused_best = max(eligible, key=lambda r: r["total_score"])

        if fused_best["Target"] == acoustic_best["Target"]:
            chosen, guard_triggered = fused_best, False
        else:
            margin_per_word = (fused_best["total_score"] - acoustic_best["total_score"]) / max(1, fused_best["n_words"])
            sim = text_similarity(acoustic_best["Target"], fused_best["Target"])
            if sim >= guard_similarity or margin_per_word >= guard_min_margin:
                chosen, guard_triggered = fused_best, False
            else:
                chosen, guard_triggered = acoustic_best, True

        ranked = sorted(rows, key=lambda r: -r["total_score"])
        runner_up = ranked[1] if len(ranked) > 1 else ranked[0]
        margin_per_word_vs_runner_up = (chosen["total_score"] - runner_up["total_score"]) / max(1, chosen["n_words"])

        results.append({
            "ID": rid,
            "Target": chosen["Target"],
            "arm_source": chosen["arm"],
            "acoustic_calibrated": chosen["acoustic_calibrated"],
            "lm_word_nats": chosen["lm_word_nats"],
            "lm_char_nats": chosen["lm_char_nats"],
            "total_score": chosen["total_score"],
            "margin_per_word_vs_runner_up": margin_per_word_vs_runner_up,
            "acoustic_best_text": acoustic_best["Target"],
            "guard_triggered": guard_triggered,
            "n_candidates": len(rows),
        })
    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# Tuning against the actual competition metric
# --------------------------------------------------------------------------

def load_refs(path: str, id_col: str = "ID", text_col: str = "Target") -> dict:
    df = pd.read_csv(path, dtype={id_col: str})
    return {
        row[id_col]: normalize_text(row[text_col])
        for _, row in df.iterrows() if pd.notna(row[text_col])
    }


def evaluate_params(components: pd.DataFrame, refs: dict, params: dict):
    picked = combine_and_select(components, params)
    scores, missing = [], 0
    for _, row in picked.iterrows():
        ref = refs.get(row["ID"])
        if ref is None:
            missing += 1
            continue
        scores.append(blended_metric(ref, row["Target"]))
    if missing and missing == len(picked):
        raise ValueError("None of the picked IDs matched the references CSV -- check ID formatting.")
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return mean_score, picked


def tune_params(components: pd.DataFrame, refs: dict, n_trials: int = 60, seed: int = 13) -> dict:
    search_space = {
        "alpha_word": (0.0, 3.0),
        "alpha_char": (0.0, 3.0),
        "beta": (-2.0, 2.0),
        "guard_min_margin": (0.0, 5.0),
        "guard_similarity": (0.5, 0.98),
        "brevity_floor": (0.2, 0.9),
    }

    if _HAS_OPTUNA:
        def objective(trial):
            params = {k: trial.suggest_float(k, lo, hi) for k, (lo, hi) in search_space.items()}
            score, _ = evaluate_params(components, refs, params)
            return score

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params, best_score = study.best_params, study.best_value
        print(f"[tune] optuna best blended metric = {best_score:.4f}")
    else:
        print("[tune] optuna not installed -- falling back to random search. "
              "`pip install optuna --break-system-packages` for a much better search.")
        rng = random.Random(seed)
        best_params, best_score = None, float("-inf")
        for _ in range(n_trials):
            params = {k: rng.uniform(lo, hi) for k, (lo, hi) in search_space.items()}
            score, _ = evaluate_params(components, refs, params)
            if score > best_score:
                best_score, best_params = score, params
        print(f"[tune] random-search best blended metric = {best_score:.4f}")

    # Always report the "do nothing" baseline so you can see whether LM
    # rescoring is actually helping before trusting the tuned parameters.
    baseline_params = {"alpha_word": 0.0, "alpha_char": 0.0, "beta": 0.0,
                        "guard_min_margin": 0.0, "guard_similarity": 1.0, "brevity_floor": 0.0}
    baseline_score, _ = evaluate_params(components, refs, baseline_params)
    print(f"[tune] acoustic-only baseline blended metric = {baseline_score:.4f}")
    if best_score <= baseline_score:
        print("[tune] WARNING: tuned LM rescoring did not beat the acoustic-only baseline "
              "on this val set. Consider more val data, a different LM order, or skipping "
              "LM rescoring for this arm.")

    return {**best_params, "_val_blended_metric": best_score, "_baseline_blended_metric": baseline_score}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)

    p_train = sub.add_parser("train", help="Build word- and char-level KenLM binaries from a train CSV.")
    p_train.add_argument("--train_csv", required=True)
    p_train.add_argument("--text_col", default="Target")
    p_train.add_argument("--word_order", type=int, default=3)
    p_train.add_argument("--char_order", type=int, default=10)
    p_train.add_argument("--out_dir", default="data")
    p_train.add_argument("--lmplz_path", default="lmplz")
    p_train.add_argument("--build_binary_path", default="build_binary")

    p_ord = sub.add_parser("select_order", help="Sweep n-gram orders by held-out perplexity, on TRAIN text only.")
    p_ord.add_argument("--text_file", required=True, help="corpus_words.txt or corpus_chars.txt from `train`.")
    p_ord.add_argument("--orders", type=int, nargs="+", required=True)
    p_ord.add_argument("--out_prefix", default="data/order_sweep")
    p_ord.add_argument("--heldout_frac", type=float, default=0.1)
    p_ord.add_argument("--lmplz_path", default="lmplz")
    p_ord.add_argument("--build_binary_path", default="build_binary")

    p_cal = sub.add_parser("calibrate", help="Fit per-source-arm z-score stats for the acoustic score column.")
    p_cal.add_argument("--candidates", nargs="+", required=True)
    p_cal.add_argument("--out", default="data/calibration.json")
    p_cal.add_argument("--allow_id_mismatch", action="store_true")

    p_tune = sub.add_parser("tune", help="Fit alpha/beta/guard params against the blended metric on labeled val data.")
    p_tune.add_argument("--candidates", nargs="+", required=True)
    p_tune.add_argument("--refs", required=True, help="CSV with ID,Target = ground-truth reference text for val.")
    p_tune.add_argument("--word_lm", required=True)
    p_tune.add_argument("--char_lm", required=True)
    p_tune.add_argument("--calibration", required=True)
    p_tune.add_argument("--out", default="data/best_params.json")
    p_tune.add_argument("--n_trials", type=int, default=60)
    p_tune.add_argument("--n_jobs", type=int, default=1)
    p_tune.add_argument("--allow_id_mismatch", action="store_true")

    p_res = sub.add_parser("rescore", help="Apply calibrated, guarded, OOV-gated rescoring to a candidate set.")
    p_res.add_argument("--candidates", nargs="+", required=True)
    p_res.add_argument("--word_lm", required=True)
    p_res.add_argument("--char_lm", required=True)
    p_res.add_argument("--calibration", required=True)
    p_res.add_argument("--params", required=True, help="JSON produced by `tune`.")
    p_res.add_argument("--out", required=True)
    p_res.add_argument("--n_jobs", type=int, default=1)
    p_res.add_argument("--allow_id_mismatch", action="store_true")

    args = parser.parse_args()

    if args.action == "train":
        os.makedirs(args.out_dir, exist_ok=True)
        word_txt = os.path.join(args.out_dir, "corpus_words.txt")
        char_txt = os.path.join(args.out_dir, "corpus_chars.txt")
        prepare_corpus(args.train_csv, word_txt, char_txt, args.text_col)
        build_lm(word_txt, os.path.join(args.out_dir, f"word_{args.word_order}gram.bin"),
                 args.word_order, args.lmplz_path, args.build_binary_path)
        build_lm(char_txt, os.path.join(args.out_dir, f"char_{args.char_order}gram.bin"),
                 args.char_order, args.lmplz_path, args.build_binary_path)
        print("[train] NOTE: these orders are a starting point, not a validated choice -- "
              "run `select_order` against held-out perplexity, especially on a small corpus.")

    elif args.action == "select_order":
        best_order, results = select_order(args.text_file, args.orders, args.out_prefix,
                                            args.heldout_frac, lmplz_path=args.lmplz_path,
                                            build_binary_path=args.build_binary_path)
        print(json.dumps({"best_order": best_order, "perplexity_by_order": results}, indent=2))

    elif args.action == "calibrate":
        cand = load_candidates(args.candidates, args.allow_id_mismatch)
        calib = fit_calibration(cand)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(calib, f, indent=2)
        print(f"[calibrate] wrote {args.out}")

    elif args.action == "tune":
        cand = load_candidates(args.candidates, args.allow_id_mismatch)
        refs = load_refs(args.refs)
        with open(args.calibration) as f:
            calib = json.load(f)
        components = compute_components(cand, calib, args.word_lm, args.char_lm, n_jobs=args.n_jobs)
        best = tune_params(components, refs, n_trials=args.n_trials)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(best, f, indent=2)
        print(f"[tune] wrote {args.out}")
        print(json.dumps(best, indent=2))

    elif args.action == "rescore":
        cand = load_candidates(args.candidates, args.allow_id_mismatch)
        with open(args.calibration) as f:
            calib = json.load(f)
        with open(args.params) as f:
            params = json.load(f)
        params = {k: v for k, v in params.items() if not k.startswith("_")}
        components = compute_components(cand, calib, args.word_lm, args.char_lm, n_jobs=args.n_jobs)
        picked = combine_and_select(components, params)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        picked.to_csv(args.out, index=False)
        n_guarded = int(picked["guard_triggered"].sum())
        print(f"[rescore] wrote {len(picked)} rows to {args.out} "
              f"({n_guarded} candidate(s) reverted to acoustic-best by the swap guard).")


if __name__ == "__main__":
    main()
