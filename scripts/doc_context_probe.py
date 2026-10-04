"""Option 3 feasibility probe: can lines of the same page/writer be found by image similarity?

Cheap page-style features per line image (paper colour, ink colour, crop size, ink density, contrast, colour
histogram).  For each train line, take its K nearest train lines in feature space and test whether they share its
RARE words (proper names, odd spellings) much more often than K random lines do.  Strong enrichment => document
structure is recoverable and neighbour vocabulary can steer candidate selection.

  .venv/bin/python -m scripts.doc_context_probe --images <dir with *.jpg>
"""
import argparse
import collections
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def features(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    a = np.asarray(im.resize((max(8, w // 4), max(8, h // 4))), dtype=np.float32) / 255.0
    g = a.mean(2)
    thr = np.quantile(g, 0.15)
    ink, paper = a[g <= thr], a[g >= np.quantile(g, 0.6)]
    hist = np.histogramdd(a.reshape(-1, 3), bins=(4, 4, 4), range=[(0, 1)] * 3)[0].ravel()
    hist = hist / hist.sum()
    return np.concatenate([[w, h, w / h, g.std(), (g < g.mean() - 2 * g.std()).mean()],
                           paper.mean(0), paper.std(0), ink.mean(0), ink.std(0), np.sqrt(hist)])


def words(t):
    return [x.strip(".,;:()[]~^").lower() for x in t.split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    tr = pd.read_csv("Train.csv", dtype=str, keep_default_na=False)
    te = pd.read_csv("Test.csv", dtype=str, keep_default_na=False)
    cache = "preds_checkpoint/doc_feats.npz"
    ids = list(tr.ID) + list(te.ID)
    if os.path.exists(cache):
        X = np.load(cache)["X"]
    else:
        X = np.stack([features(os.path.join(a.images, f"{i}.jpg")) for i in ids])
        np.savez(cache, X=X, ids=np.array(ids))
    Xtr = StandardScaler().fit(X).transform(X)[: len(tr)]

    freq = collections.Counter(w for t in tr.Target for w in set(words(t)))
    rare = [{w for w in words(t) if len(w) >= 4 and freq[w] <= 5} for t in tr.Target]
    bags = [set(words(t)) for t in tr.Target]
    nn = NearestNeighbors(n_neighbors=a.k + 1).fit(Xtr)
    _, idx = nn.kneighbors(Xtr)
    rng = np.random.default_rng(0)
    hit_nn = hit_rand = n = 0
    for i, r in enumerate(rare):
        if not r:
            continue
        nb = [j for j in idx[i] if j != i and tr.Target[j] != tr.Target[i]][: a.k]
        rd = [j for j in rng.choice(len(tr), a.k + 1, replace=False) if j != i][: a.k]
        u_nn = set().union(*(bags[j] for j in nb))
        u_rd = set().union(*(bags[j] for j in rd))
        hit_nn += len(r & u_nn)
        hit_rand += len(r & u_rd)
        n += len(r)
    print(f"lines with rare words: {sum(bool(r) for r in rare)}  rare-word tokens: {n}")
    print(f"rare words found in {a.k} image-nearest lines: {hit_nn / n:.1%}   in {a.k} random lines: {hit_rand / n:.1%}   "
          f"enrichment x{hit_nn / max(1, hit_rand):.1f}")


if __name__ == "__main__":
    main()
