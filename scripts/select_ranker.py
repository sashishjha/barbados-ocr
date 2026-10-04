"""Richer selector: L2-regularised linear regression (numpy) predicting each candidate's loss relative to the line's
ROVER candidate from within-line-relative features; pick argmin predicted loss, gate by margin tau. 5-fold CV by line.
  .venv/bin/python -m scripts.select_ranker --slots q3,q7a,q32"""
import argparse

import numpy as np
import pandas as pd

from scripts.select_candidates import REF, VAL_SCORERS, attach_loss, build, fast_score
from scripts.tune_ensemble_r8 import score


def feats(c, slots, meta):
    f = pd.DataFrame(index=c.index)
    for s in slots:
        f[f"rel_{s}"] = c[f"rel_{s}"]
        f[f"rank_{s}"] = c[f"lp_{s}"].groupby(c.ID).rank(ascending=False, pct=True)
        f[f"sum_{s}"] = (c[f"lp_{s}"] - c[f"lp_{s}"].groupby(c.ID).transform("max")).clip(lower=-30) / 10
    f["n_top1"] = c.n_top1
    sup = meta.groupby(["ID", "Target"]).arm.nunique().rename("n_sup").reset_index()
    f["n_support"] = c[["ID", "Target"]].merge(sup, how="left", on=["ID", "Target"]).n_sup.fillna(0).values
    f["d_rover"] = c.d_rover
    f["is_rover"] = c.is_rover
    L = c.nchar / c.nchar.groupby(c.ID).transform("median")
    f["len_ratio"] = (L - 1).abs()
    f["long"] = (L > 1.15).astype(float)
    f["short"] = (L < 0.85).astype(float)
    f["pool_div"] = c.d_rover.groupby(c.ID).transform("mean")
    f["rel_x_div"] = f[f"rel_{slots[0]}"] * f.pool_div
    return f


def fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0).replace(0, 1)
    Z = np.c_[np.ones(len(X)), ((X - mu) / sd).values]
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ y)
    return lambda Xn: np.c_[np.ones(len(Xn)), ((Xn - mu) / sd).values] @ w


def pick(c, pred, tau):
    p = pd.Series(pred, index=c.index)
    rov = p.where(c.is_rover == 1).groupby(c.ID).transform("min")
    best = p.groupby(c.ID).transform("min")
    take = (p == best) & (rov - best > tau)
    take |= (c.is_rover == 1) & ~(rov - best > tau)
    return c[take].drop_duplicates("ID")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="q3,q7a,q32")
    a = ap.parse_args()
    slots = a.slots.split(",")
    c, _ = build("val", {s: VAL_SCORERS[s] for s in slots})
    c = attach_loss(c)
    meta = pd.read_csv("preds_checkpoint/r8_ens/cands_val_meta.csv", dtype={"ID": str}, keep_default_na=False)
    meta = meta[meta.arm != "rover"]
    X = feats(c, slots, meta)
    loss = 0.5 * c.cer + 0.5 * c.wer
    y = (loss - loss.where(c.is_rover == 1).groupby(c.ID).transform("max")).values       # loss relative to ROVER's
    rng = np.random.default_rng(0)
    ids = c.ID.unique(); fold = pd.Series(rng.integers(0, 5, len(ids)), index=ids).reindex(c.ID).values
    rov = c[c.is_rover == 1].drop_duplicates("ID")
    print(f"ROVER {score(rov[['ID','Target']], REF):.4f}   features: {list(X.columns)}")
    for alpha in (1, 30, 300, 3000):
        for tau in (0.0, 0.01, 0.03):
            picks = []
            for f in range(5):
                tr, te = fold != f, fold == f
                m = fit(X[tr], y[tr], alpha)
                picks.append(pick(c[te], m(X[te]), tau))
            sel = pd.concat(picks)
            print(f"alpha {alpha:5d} tau {tau:.2f}: CV {score(sel[['ID','Target']], REF):.4f}  override {1 - sel.is_rover.mean():.1%}")


if __name__ == "__main__":
    main()
