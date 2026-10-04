"""Tune the R8 ensemble (method + per-arm weights) on fold-0 val for the LB metric (0.5*CER + 0.5*WER).

Each input file gets its own arm name (the n-best CSVs all say arm=vlm, which would merge them).
Val and test must come from the same model for an arm to be usable in a submission.

  .venv/bin/python -m scripts.tune_ensemble_r8 --dir preds_checkpoint/r8_ens
"""
from __future__ import annotations

import argparse
import itertools
from multiprocessing import Pool
import os
import re
import tempfile

import pandas as pd

from src.ensemble import mbr, rover
from src.eval.edit_distance import levenshtein
from src.eval.eval_metrics import evaluate_set


def load(path: str, arm: str, nbest: int) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ID": str}, keep_default_na=False)
    if "rank" not in df.columns:
        df["rank"] = 0
    df = df[df["rank"] < nbest]
    return df.assign(arm=arm)[["ID", "rank", "Target", "arm"]]


def fuse(cand: pd.DataFrame, weights: dict, method: str) -> pd.DataFrame:
    cand = cand[cand["arm"].isin(weights)]
    fn = rover if method == "rover" else mbr
    rows = []
    for rid, g in cand.groupby("ID"):
        ws = [weights[a] / (1 + r) for a, r in zip(g["arm"], g["rank"])]
        rows.append({"ID": rid, "Target": " ".join(str(fn(g["Target"].astype(str).tolist(), ws)).split())})
    return pd.DataFrame(rows)


def is_runaway(text: str) -> bool:
    """Degenerate repetition of any short phrase 6+ times in a row: '& R . . . .', 'in the in the ...',
    '1666 - 1666 - ...', '100000...' (one such line cost 0.018 on the LB: the official metric does not
    cap per-line error). Fires on 0 of the 4054 train labels; tilde fills ('~ ~ ~') are legitimate."""
    return bool(re.search(r"(.{2,12}?)\1{5,}", text)) and not re.fullmatch(r"[\w\s,.'&^:-]*(~ ?)+", text)


def prefilter_candidates(cand: pd.DataFrame, ratio: float = 2.0, slack: int = 15) -> pd.DataFrame:
    """Drop runaway or overlong candidates per line BEFORE fusion (they slow ROVER's quadratic alignment
    and can leak into the vote); always keeps at least one candidate per line."""
    keep = []
    for _, g in cand.groupby("ID"):
        texts = g["Target"].astype(str)
        ok = ~texts.map(is_runaway)
        med = sorted(texts[ok].str.len())[ok.sum() // 2] if ok.any() else int(texts.str.len().median())
        ok &= texts.str.len() <= ratio * med + slack
        keep.append(g[ok] if ok.any() else g.loc[[texts.str.len().idxmin()]])
    return pd.concat(keep)


def guard_runaway(cand: pd.DataFrame, fused: pd.DataFrame, ratio: float = 2.0, slack: int = 15):
    """Replace a fused line that is a runaway, or far longer than the median candidate, by the
    non-runaway candidate that agrees best with the others. Returns (fixed_df, [(ID, old, new)])."""
    out = fused.set_index("ID")["Target"].copy()
    changed = []
    for rid, g in cand.groupby("ID"):
        texts = [t for t in g["Target"].astype(str) if not is_runaway(t)] or g["Target"].astype(str).tolist()
        med = sorted(len(t) for t in texts)[len(texts) // 2]
        f = out[rid]
        if is_runaway(f) or len(f) > ratio * med + slack:
            best = min(texts, key=lambda t: sum(levenshtein(t, u) for u in texts))
            changed.append((rid, f, best))
            out[rid] = best
    return out.reset_index(), changed


def score(pred: pd.DataFrame, ref_csv: str) -> float:
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        pred.to_csv(f.name, index=False)
    try:
        return evaluate_set(ref_csv, f.name, verbose=False)["final_05_05"]
    finally:
        os.unlink(f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="preds_checkpoint/r8_ens")
    ap.add_argument("--nbest", type=int, default=1)
    a = ap.parse_args()
    d = a.dir
    ref = os.path.join(d, "val_ref_fold0.csv")

    # arm -> (val file, test file or None)
    arms = {
        "qwen3vl8b_r8": ("qwen3vl8b_val_nbest_1387.csv", "qwen3vl8b_test_nbest_1387.csv"),
        "qwen7b_r8b":   ("qwen7b_b_val_nbest_1387.csv", "qwen7b_b_test_nbest_1387.csv"),
        "qwen7b_r8a":   ("qwen7b_a_val_nbest_1387.csv", "qwen7b_a_test_nbest_1387.csv"),  # epoch-3 best
        "qwen_r5":      ("valtop1_qwen25vl_lora_r5.csv", "vlm_test_nbest_r5.csv"),
        "trocr_r5":     ("valtop1_trocr_hw_r5.csv", "trocr_test_nbest_r5.csv"),
        "trocr_r7":     ("valtop1_trocr_hw_r7.csv", None),              # no test preds yet
    }
    val = pd.concat([load(os.path.join(d, v), k, a.nbest) for k, (v, _) in arms.items()])
    submittable = {k for k, (_, t) in arms.items() if t and os.path.exists(os.path.join(d, t))}

    print("single arms (val 0.5/0.5):")
    for k in arms:
        print(f"  {k:14s} {score(fuse(val, {k: 1.0}, 'rover'), ref):.4f}   test={'yes' if k in submittable else 'no'}")

    # Only the sets we can submit now or soon (a full grid is hours of pure-Python ROVER).
    base = ["qwen3vl8b_r8", "qwen7b_r8a", "qwen7b_r8b", "trocr_r5"]
    sets = [base, base + ["qwen_r5"]]
    grid = [0.5, 1.0, 1.5, 2.0]
    jobs = [(dict(zip(sub, (1.0,) + w)), m)            # only weight ratios matter: first arm fixed at 1
            for sub in sets for w in itertools.product(grid, repeat=len(sub) - 1)
            for m in ("rover", "mbr")]
    with Pool(8, initializer=_init, initargs=(val, ref)) as pool:  # macOS spawns: pass data explicitly
        scores = pool.map(_eval, jobs, chunksize=4)
    results = sorted(zip(scores, jobs), key=lambda r: -r[0])
    for sub in sets:
        best = [(s, w, m) for s, (w, m) in results if set(w) == set(sub)][:3]
        tag = "SUBMITTABLE" if set(sub) <= submittable else "not yet (test preds missing)"
        print(f"\n{'+'.join(sub)}  [{tag}]")
        for s, w, m in best:
            print(f"  {s:.4f}  {m:5s}  " + ", ".join(f"{k}={v}" for k, v in w.items()))


_VAL = _REF = None


def _init(val, ref):
    global _VAL, _REF
    _VAL, _REF = val, ref


def _eval(job):
    weights, method = job
    return score(fuse(_VAL, weights, method), _REF)


if __name__ == "__main__":
    main()
