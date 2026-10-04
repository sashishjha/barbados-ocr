"""Compare ensemble variants on fold-0 val (0.5/0.5): character ROVER vs word-level ROVER, and equal votes
vs family-balanced votes (each model family shares one vote), simulating "many seeds of one family" with
three fold-0 Qwen3-VL-8B variants.  .venv/bin/python -m scripts.ens_variants
"""
from __future__ import annotations

import sys
from multiprocessing import Pool

import pandas as pd

from scripts.tune_ensemble_r8 import fuse, load, prefilter_candidates, score
from src.rover import rover_combine

D = "preds_checkpoint/r8_ens"
REF = f"{D}/val_ref_fold0.csv"
FILES = {
    "q32": "qwen3vl32b_r10_val_nbest_1478.csv",
    "q3_r8": "qwen3vl8b_val_nbest_1387.csv", "q3_r9": "qwen3vl8b_r9_val_nbest_1430.csv",
    "q3_r9s7": "qwen3vl8b_r9s7_val_nbest_1430.csv",
    "q7_r8a": "qwen7b_a_val_nbest_1387.csv", "q7_r8b": "qwen7b_b_val_nbest_1387.csv",
    "trocr": "valtop1_trocr_hw_r5.csv",
}
BASE = ["q32", "q3_r8", "q7_r8a", "q7_r8b", "trocr"]
SEEDY = BASE + ["q3_r9", "q3_r9s7"]
FAMILY = {"q32": "q32", "q3_r8": "q3", "q3_r9": "q3", "q3_r9s7": "q3", "q7_r8a": "q7", "q7_r8b": "q7", "trocr": "trocr"}


def family_balanced(arms):
    n = pd.Series([FAMILY[a] for a in arms]).value_counts()
    return {a: 1.0 / n[FAMILY[a]] for a in arms}


def fuse_word(cand, weights):
    rows = []
    for rid, g in cand[cand["arm"].isin(weights)].groupby("ID"):
        g = g.assign(w=g["arm"].map(weights)).sort_values("w", ascending=False)  # most trusted first
        rows.append({"ID": rid, "Target": rover_combine(g["Target"].astype(str).tolist(), g["w"].tolist())})
    return pd.DataFrame(rows)


def run(job):
    name, method, weights = job
    pf = _PF[_PF["arm"].isin(weights)]
    pred = fuse(pf, weights, "rover") if method == "char" else fuse_word(pf, weights)
    return name, score(pred, REF)


_PF = None


def _init(pf):
    global _PF
    _PF = pf


if __name__ == "__main__":
    val = pd.concat([load(f"{D}/{f}", k, 1) for k, f in FILES.items()])
    pf = prefilter_candidates(val)
    jobs = [
        ("5 families, equal, char ROVER", "char", {a: 1.0 for a in BASE}),
        ("5 families, equal, word ROVER", "word", {a: 1.0 for a in BASE}),
        ("7 (3x Qwen3-8B), equal, char", "char", {a: 1.0 for a in SEEDY}),
        ("7 (3x Qwen3-8B), family-balanced, char", "char", family_balanced(SEEDY)),
        ("7 (3x Qwen3-8B), equal, word", "word", {a: 1.0 for a in SEEDY}),
        ("7 (3x Qwen3-8B), family-balanced, word", "word", family_balanced(SEEDY)),
    ]
    with Pool(6, initializer=_init, initargs=(pf,)) as p:
        for name, s in p.map(run, jobs):
            print(f"{s:.4f}  {name}")
    sys.exit(0)
