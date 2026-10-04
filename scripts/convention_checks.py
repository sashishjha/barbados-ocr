"""CPU checks suggested by the external review: (B) error anatomy single model vs ROVER, (C) ROVER word-snap,
(D) within-cluster consistency of AMBIGUOUS word variants (case / caret / punctuation), (E) targeted recasing.
  .venv/bin/python -m scripts.convention_checks"""
import collections, difflib, os, re, sys
import numpy as np
import pandas as pd
from rapidfuzz import distance, process
from scripts.tune_ensemble_r8 import load, fuse, prefilter_candidates, score

D = "preds_checkpoint/r8_ens"; REF = f"{D}/val_ref_fold0.csv"
SCR = os.environ.get("IMG_STATS", "preds_checkpoint/img_stats_train.npy")
tf = pd.read_csv(f"{D}/train_folds.csv", dtype={"ID": str}, keep_default_na=False)
va = tf[tf.fold == 0].set_index("ID")
F = {"q32": "qwen3vl32b_r10_val_nbest_1478.csv", "q3_r8": "qwen3vl8b_val_nbest_1387.csv", "q7_r8a": "qwen7b_a_val_nbest_1387.csv",
     "q7_r8b": "qwen7b_b_val_nbest_1387.csv", "trocr": "valtop1_trocr_hw_r5.csv"}
pf = prefilter_candidates(pd.concat([load(f"{D}/{f}", k, 1) for k, f in F.items()]))
rov = fuse(pf, {k: 1 for k in F}, "rover").set_index("ID").Target.reindex(va.index)
alnum = lambda s: re.sub(r"[^A-Za-z0-9]", "", s)


def anatomy(hyp):
    c = collections.Counter(); n = 0
    for i in va.index:
        ref = va.Target[i].split(); h = hyp[i].split()
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ref, b=h, autojunk=False).get_opcodes():
            if tag == "equal": continue
            R, H = ref[i1:i2], h[j1:j2]; n += len(R)
            if len(R) == len(H):
                for a, b in zip(R, H):
                    if a == b: continue
                    c["case" if a.lower() == b.lower() else "marks" if alnum(a).lower() == alnum(b).lower() else "misread"] += 1
            else: c["split/merge/ins/del"] += len(R)
    tot = sum(len(t.split()) for t in va.Target)
    return n / tot, {k: round(v / n, 3) for k, v in c.items()}


print("== B. error anatomy (WER, share of errors) ==")
for name in ("q32", "q3_r8", "q7_r8a"):
    h = pf[pf.arm == name].set_index("ID").Target.reindex(va.index).fillna("")
    w, s = anatomy(h); print(f"  {name:7s} WER {w:.3%}  {s}")
w, s = anatomy(rov); print(f"  ROVER   WER {w:.3%}  {s}")

print("\n== C. ROVER word-snap (every ROVER word must be a word some member produced on that line) ==")
mem = {k: pf[pf.arm == k].set_index("ID").Target.reindex(va.index).fillna("") for k in F}
base = score(rov.reset_index().rename(columns={"index": "ID"}), REF)
for maxd in (1, 2, 3):
    out = []; ch = 0
    for i in va.index:
        vocab = set(w for k in F for w in mem[k][i].split()); ws = rov[i].split()
        for j, w in enumerate(ws):
            if w not in vocab and vocab:
                m = process.extractOne(w, list(vocab), scorer=distance.Levenshtein.distance)
                if m and m[1] <= maxd: ws[j] = m[0]; ch += 1
        out.append((i, " ".join(ws)))
    print(f"  snap distance<={maxd}: {ch} words changed -> {score(pd.DataFrame(out, columns=['ID', 'Target']), REF):.4f} (ROVER {base:.4f})")

print("\n== D. ambiguous word variants: within-cluster vs global majority (5-fold CV by line, train only) ==")
S = np.load(SCR); Z = (S - S.mean(0)) / (S.std(0) + 1e-9)
def km(X, k, seed=0, it=40):
    r = np.random.default_rng(seed); C = X[r.choice(len(X), k, replace=False)]
    for _ in range(it):
        a = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1); C = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
    return a
tok = []
for li, t in enumerate(tf.Target):
    for w in t.split():
        b = alnum(w).lower()
        if len(b) >= 2: tok.append((li, b, w))
cnt = collections.defaultdict(collections.Counter)
for _, b, w in tok: cnt[b][w] += 1
amb = {b for b, c in cnt.items() if sum(c.values()) >= 25 and c.most_common(1)[0][1] / sum(c.values()) < 0.70}
print(f"  ambiguous bases (>=25 occurrences, top variant <70%): {len(amb)} e.g. {sorted(amb, key=lambda b: -sum(cnt[b].values()))[:10]}")
T = [x for x in tok if x[1] in amb]; lines = np.array([x[0] for x in T]); bases = [x[1] for x in T]; vars_ = [x[2] for x in T]
fold = np.random.default_rng(1).integers(0, 5, len(tf))[lines]
for k in (1, 4, 8, 16):
    cl = km(Z, k)[lines]; ok = 0; okg = 0
    for f in range(5):
        g = collections.defaultdict(collections.Counter); gc = collections.defaultdict(collections.Counter)
        for i in np.where(fold != f)[0]: g[bases[i]][vars_[i]] += 1; gc[(bases[i], cl[i])][vars_[i]] += 1
        for i in np.where(fold == f)[0]:
            pg = g[bases[i]]; okg += pg.most_common(1)[0][0] == vars_[i]
            sc = collections.Counter({v: gc[(bases[i], cl[i])][v] + 2 * pg[v] / sum(pg.values()) for v in pg})
            ok += sc.most_common(1)[0][0] == vars_[i]
    print(f"  k={k:2d}: variant prediction acc {ok / len(T):.3f}  (global majority {okg / len(T):.3f}; n={len(T)})")

print("\n== E. targeted recasing: only where members disagree on the case of the word AND cluster prior >= conf ==")
cl4 = km(Z, 4); clid = dict(zip(tf.ID, cl4))
gl = collections.defaultdict(lambda: [1.0, 1.0]); gc = collections.defaultdict(lambda: [0.0, 0.0])
for i, t, f in zip(tf.ID, tf.Target, tf.fold):
    if f == 0: continue
    for w in t.split():
        c = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", w)
        if len(c) >= 3: gl[c.lower()][int(c[0].isupper())] += 1; gc[(c.lower(), clid[i])][int(c[0].isupper())] += 1
core = lambda w: re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", w)
for conf in (0.7, 0.8, 0.9):
    out = []; ch = 0
    for i in va.index:
        ws = rov[i].split(); mc = collections.defaultdict(set)
        for k in F:
            for w in mem[k][i].split():
                c = core(w)
                if len(c) >= 3: mc[c.lower()].add(c[0].isupper())
        for j, w in enumerate(ws):
            c = core(w)
            if len(c) < 3 or len(mc.get(c.lower(), ())) < 2: continue            # members agree on case -> leave it
            p = np.array(gl[c.lower()]); p /= p.sum(); q = np.array(gc[(c.lower(), clid[i])]) + 2 * p; q /= q.sum()
            cap = int(q.argmax())
            if q[cap] >= conf and cap != int(c[0].isupper()):
                ws[j] = w.replace(c, (c[0].upper() if cap else c[0].lower()) + c[1:], 1); ch += 1
        out.append((i, " ".join(ws)))
    print(f"  conf>={conf}: {ch:3d} words recased -> {score(pd.DataFrame(out, columns=['ID', 'Target']), REF):.4f} (ROVER {base:.4f})")
