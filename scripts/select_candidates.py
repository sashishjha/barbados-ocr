"""Per-line candidate selection from rescoring log-probs (scripts/rescore_candidates.py), tuned on fold-0 val
with 5-fold cross-validation grouped by line, then applied to test.

Score of candidate c on line i:   s(c) = sum_m w_m * rel_m(c) + b_top1 * n_top1(c) - lam * d_rover(c)
  rel_m      = model m's log-prob per character of c minus the best candidate's on the same line (<= 0); relative
               values transfer from the fold-0 scorers (val) to their all-data twins (test).
  n_top1     = number of generators whose top-1 reading is c (cross-model consensus).
  d_rover    = 0.5*CER + 0.5*WER of c against the ROVER string (stay close to the vote).
Gate: c replaces ROVER only if s(c) - s(ROVER) > tau.

  .venv/bin/python -m scripts.select_candidates            # CV report on val
  .venv/bin/python -m scripts.select_candidates --apply     # + build test submissions
"""
import argparse
import itertools
import sys

import numpy as np
import pandas as pd

from scripts.tune_ensemble_r8 import is_runaway, score
from src.eval.edit_distance import levenshtein
from src.eval.normalize import get_normalizer

D = "preds_checkpoint/r8_ens"
REF = f"{D}/val_ref_fold0.csv"
NORM = get_normalizer("raw")
# scorer slots: the same families in val (fold-0 models) and test (models of the 0.9221 recipe)
VAL_SCORERS = {"q3": "q3_r8", "q7a": "q7_r8a", "q7b": "q7_r8b", "q32": "q32"}
TEST_SCORERS = {"q3": "q3_all", "q7a": "q7_all", "q7b": "q7_r8b", "q32": "q32"}


def line_err(ref, hyp):
    """(cer, wer, w_char) for one line, as src/eval (length_pow, raw, chars-weighted, per-line cap 1)."""
    r, h = NORM(ref), NORM(hyp)
    cer = min(1.0, levenshtein(list(r), list(h)) / max(1, len(r)))
    rw, hw = r.split(), h.split()
    wer = min(1.0, levenshtein(rw, hw) / max(1, len(rw)))
    return cer, wer, len(r) ** 0.5


def pair_loss(a, b):
    cer, wer, _ = line_err(a, b)
    return 0.5 * cer + 0.5 * wer


def build(split, scorers):
    meta = pd.read_csv(f"{D}/cands_{split}_meta.csv", dtype={"ID": str}, keep_default_na=False)
    meta = meta[meta.Target.str.strip() != ""]
    rover = meta[meta.arm == "rover"].drop_duplicates("ID").set_index("ID").Target
    top1 = meta[(meta["rank"] == 0) & (meta.arm != "rover")]
    c = meta[["ID", "Target"]].drop_duplicates().reset_index(drop=True)
    c["n_top1"] = c.merge(top1.groupby(["ID", "Target"]).arm.nunique().rename("n").reset_index(), how="left",
                          on=["ID", "Target"])["n"].fillna(0).values
    c["is_rover"] = (c.Target.values == rover.reindex(c.ID).values).astype(float)
    c["d_rover"] = [pair_loss(r, t) for r, t in zip(rover.reindex(c.ID).values, c.Target)]
    c["nchar"] = c.Target.str.len().clip(lower=1)
    c["runaway"] = c.Target.map(is_runaway).astype(float)
    for slot, name in scorers.items():
        lp = pd.read_csv(f"preds_checkpoint/lp_{split}_{name}.csv", dtype={"ID": str}, keep_default_na=False)
        lp = lp.drop_duplicates(["ID", "Target"])
        c = c.merge(lp[["ID", "Target", "logp"]].rename(columns={"logp": f"lp_{slot}"}), how="left", on=["ID", "Target"])
        pc = c[f"lp_{slot}"] / c["nchar"]
        c[f"rel_{slot}"] = (pc - pc.groupby(c.ID).transform("max")).fillna(-10.0)
    return c, rover


def attach_loss(c):
    ref = pd.read_csv(REF, dtype=str, keep_default_na=False).set_index("ID").Target
    e = [line_err(ref.get(i, ""), t) for i, t in zip(c.ID, c.Target)]
    c["cer"], c["wer"], c["w"] = zip(*e)
    return c


def choose(c, p, slots):
    s = sum(p["w_" + k] * c["rel_" + k] for k in slots) + p["b"] * c["n_top1"] - p["lam"] * c["d_rover"] - 99 * c["runaway"]
    s_rov = s.where(c.is_rover == 1).groupby(c.ID).transform("max")
    best = s.groupby(c.ID).transform("max")
    keep = (s == best) & (best - s_rov > p["tau"])
    keep |= (c.is_rover == 1) & ~(best - s_rov > p["tau"])
    return c[keep].drop_duplicates("ID")


def fast_score(sel, wsum_c, wsum_w):
    return 1 - 0.5 * (sel.cer * sel.w).sum() / wsum_c - 0.5 * (sel.wer * sel.w).sum() / wsum_w


def grid(slots):
    ws = [0.0, 0.5, 1.0, 2.0]
    for combo in itertools.product(ws, repeat=len(slots)):
        if not any(combo):
            continue
        for b, lam, tau in itertools.product([0.0, 0.05, 0.15], [0.0, 0.5, 2.0], [0.0, 0.05, 0.2]):
            yield {**{"w_" + k: v for k, v in zip(slots, combo)}, "b": b, "lam": lam, "tau": tau}


def cv(c, slots, k=5, seed=0):
    ids = c.ID.unique()
    rng = np.random.default_rng(seed)
    fold = pd.Series(rng.integers(0, k, len(ids)), index=ids)
    c = c.assign(fold=fold.reindex(c.ID).values)
    params = list(grid(slots))
    picks = []
    for f in range(k):
        tr = c[c.fold != f]
        rov_tr = tr[tr.is_rover == 1].drop_duplicates("ID")
        wc, ww = rov_tr.w.sum(), rov_tr.w.sum()
        best = max(params, key=lambda p: fast_score(choose(tr, p, slots), wc, ww))
        picks.append(choose(c[c.fold == f], best, slots))
        print(f"  fold {f}: {best}", flush=True)
    return pd.concat(picks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="q3,q7a,q7b", help="scorer slots to use (q32 once its scores exist)")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    slots = a.slots.split(",")
    c, rover = build("val", {s: VAL_SCORERS[s] for s in slots})
    c = attach_loss(c)
    print(f"val: {c.ID.nunique()} lines, {len(c)} candidates; missing scores: "
          f"{ {s: int(c['lp_' + s].isna().sum()) for s in slots} }")
    rov = c[c.is_rover == 1].drop_duplicates("ID")
    print(f"ROVER {score(rov[['ID', 'Target']], REF):.4f}   oracle {score(c.loc[(0.5 * c.cer + 0.5 * c.wer).groupby(c.ID).idxmin(), ['ID', 'Target']], REF):.4f}")
    for s in slots:   # each scorer alone, no tuning
        p = {**{"w_" + k: float(k == s) for k in slots}, "b": 0.0, "lam": 0.0, "tau": 0.0}
        print(f"  argmax {s:4s} alone: {score(choose(c, p, slots)[['ID', 'Target']], REF):.4f}")
    p = {**{"w_" + k: 1.0 for k in slots}, "b": 0.0, "lam": 0.0, "tau": 0.0}
    print(f"  argmax equal sum:   {score(choose(c, p, slots)[['ID', 'Target']], REF):.4f}")
    oof = cv(c, slots)
    print(f"CV (5-fold, grouped by line) selector: {score(oof[['ID', 'Target']], REF):.4f}  "
          f"override rate {1 - oof.is_rover.mean():.1%}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
