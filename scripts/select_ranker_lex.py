"""select_ranker + lexical features from TRAIN labels only (fold != 0): char n-gram LM (Witten-Bell interpolation),
word OOV rate, capitalised-word OOV rate, unseen '^'-patterns.  Tests whether a prior independent of the VLMs breaks
the generator=scorer circularity.   .venv/bin/python -m scripts.select_ranker_lex"""
import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from scripts.select_candidates import REF, VAL_SCORERS, attach_loss, build
from scripts.select_ranker import feats, fit, pick
from scripts.tune_ensemble_r8 import score

N = 6


class CharLM:
    def __init__(self, texts, n=N):
        self.n = n
        self.cnt = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]      # order k context -> char -> count
        self.V = set()
        for t in texts:
            s = "\x02" * (n - 1) + t + "\x03"
            for i in range(n - 1, len(s)):
                self.V.add(s[i])
                for k in range(n):
                    self.cnt[k][s[i - k:i]][s[i]] += 1
        self.V = len(self.V) + 1

    def p(self, ctx, ch):
        pr = 1.0 / self.V
        for k in range(self.n):
            c = self.cnt[k].get(ctx[len(ctx) - k:] if k else "")
            if not c:
                break
            tot, types = sum(c.values()), len(c)
            lam = tot / (tot + types)
            pr = lam * c.get(ch, 0) / tot + (1 - lam) * pr
        return pr

    def logp(self, t):
        s = "\x02" * (self.n - 1) + t + "\x03"
        return sum(math.log(self.p(s[max(0, i - self.n + 1):i], s[i])) for i in range(self.n - 1, len(s)))


def main():
    slots = ["q3", "q7a", "q32"]
    c, _ = build("val", {s: VAL_SCORERS[s] for s in slots})
    c = attach_loss(c)
    tf = pd.read_csv("preds_checkpoint/r8_ens/train_folds.csv", dtype=str, keep_default_na=False)
    tr = tf[tf.fold != "0"].Target.tolist()
    lm = CharLM(tr)
    vocab = set(w for t in tr for w in t.split()); vocab_l = set(w.lower() for w in vocab)
    caret = set(re.findall(r"\S*\^\S*", " ".join(tr)))
    meta = pd.read_csv("preds_checkpoint/r8_ens/cands_val_meta.csv", dtype={"ID": str}, keep_default_na=False)
    meta = meta[meta.arm != "rover"]
    X = feats(c, slots, meta)
    lp = c.Target.map(lm.logp)
    X["lm_char"] = lp / c.nchar
    X["lm_rel"] = X.lm_char - X.lm_char.groupby(c.ID).transform("max")
    ws = c.Target.str.split()
    X["oov"] = ws.map(lambda w: np.mean([x not in vocab for x in w]) if w else 0)
    X["oov_rel"] = X.oov - X.oov.groupby(c.ID).transform("min")
    X["caps_oov"] = ws.map(lambda w: np.mean([x.lower() not in vocab_l for x in w if x[:1].isupper()]) if any(x[:1].isupper() for x in w) else 0)
    X["caret_new"] = c.Target.map(lambda t: np.mean([x not in caret for x in re.findall(r"\S*\^\S*", t)]) if "^" in t else 0)
    loss = 0.5 * c.cer + 0.5 * c.wer
    y = (loss - loss.where(c.is_rover == 1).groupby(c.ID).transform("max")).values
    ids = c.ID.unique(); rng = np.random.default_rng(0)
    fold = pd.Series(rng.integers(0, 5, len(ids)), index=ids).reindex(c.ID).values
    rov = c[c.is_rover == 1].drop_duplicates("ID")
    print(f"ROVER {score(rov[['ID','Target']], REF):.4f} | LM-only argmax (candidate with best char-{N}gram logp/char): "
          f"{score(c.loc[X.lm_char.groupby(c.ID).idxmax(), ['ID','Target']], REF):.4f}")
    for name, cols in (("base 18 feats", [x for x in X.columns if x not in ("lm_char", "lm_rel", "oov", "oov_rel", "caps_oov", "caret_new")]),
                       ("+ lexical feats", list(X.columns))):
        for alpha in (30, 300):
            for tau in (0.0, 0.01, 0.02):
                picks = []
                for f in range(5):
                    tr_, te_ = fold != f, fold == f
                    m = fit(X.loc[tr_, cols], y[tr_], alpha)
                    picks.append(pick(c[te_], m(X.loc[te_, cols]), tau))
                sel = pd.concat(picks)
                print(f"{name:16s} alpha {alpha:4d} tau {tau:.2f}: CV {score(sel[['ID','Target']], REF):.4f} override {1 - sel.is_rover.mean():.1%}")


if __name__ == "__main__":
    main()
