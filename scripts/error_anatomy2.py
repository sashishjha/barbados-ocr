"""Finer word-error anatomy of an ensemble output vs fold-0 references.  .venv/bin/python -m scripts.error_anatomy2"""
import re, sys, collections, difflib
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein as L
from scripts.tune_ensemble_r8 import load, fuse, prefilter_candidates
d = "preds_checkpoint/r8_ens"
tf = pd.read_csv(f"{d}/train_folds.csv", dtype={"ID": str}, keep_default_na=False); va = tf[tf.fold == 0].set_index("ID")
F = {"q32": "qwen3vl32b_r10_val_nbest_1478.csv", "q3": "qwen3vl8b_val_nbest_1387.csv", "q7a": "qwen7b_a_val_nbest_1387.csv",
     "q7b": "qwen7b_b_val_nbest_1387.csv", "trocr": "valtop1_trocr_hw_r5.csv"}
pf = prefilter_candidates(pd.concat([load(f"{d}/{f}", k, 1) for k, f in F.items()]))
rov = fuse(pf, {k: 1 for k in F}, "rover").set_index("ID").Target
alpha = lambda s: re.sub(r"[^A-Za-z]", "", s)
def arch(s):  # undo 17th-c. letter conventions: u/v, i/j, ff->f, y->i
    s = alpha(s).lower()
    return re.sub(r"^ff", "f", s).replace("v", "u").replace("j", "i").replace("y", "i")
FILL = {"~", "-", "=", "—", "_", "..", "...", "....", "&c", "&c."}
cls = collections.Counter(); ex = collections.defaultdict(list); last_word_trunc = 0
words = 0
for i in va.index:
    r, h = va.Target[i].split(), rov[i].split(); words += len(r)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=r, b=h, autojunk=False).get_opcodes():
        if tag == "equal": continue
        R, H = r[i1:i2], h[j1:j2]
        if R and H and alpha("".join(R)).lower() == alpha("".join(H)).lower() and len(R) != len(H):
            k = "merge/split (spacing)"; n = max(len(R), len(H)); cls[k] += n; ex[k].append((" ".join(R), " ".join(H))); continue
        if tag == "delete" or tag == "insert":
            toks = R if tag == "delete" else H
            for t in toks:
                k = ("filler/mark dropped" if t in FILL or not alpha(t) else "word dropped") if tag == "delete" else ("filler/mark added" if t in FILL or not alpha(t) else "word added (hallucinated)")
                cls[k] += 1; ex[k].append((t if tag == "delete" else "∅", "∅" if tag == "delete" else t))
            continue
        pairs = list(zip(R, H)); extra = abs(len(R) - len(H))
        if extra: cls["word dropped" if len(R) > len(H) else "word added (hallucinated)"] += extra
        for a, b in pairs:
            if a == b: continue
            A, B = alpha(a).lower(), alpha(b).lower()
            if a.lower() == b.lower(): k = "case only"
            elif A == B: k = "marks/punctuation only (incl. ^ :)"
            elif A and B and (A.startswith(B) or B.startswith(A)):
                k = "truncated/completed word (prefix)"
                if i1 + pairs.index((a, b)) == len(r) - 1: last_word_trunc += 1
            elif A and B and arch(a) == arch(b): k = "archaic letters u/v i/j ff y/i"
            elif A and B and L.distance(A, B) <= 2: k = "spelling variant (<=2 letters)"
            else: k = "misread (different word)"
            cls[k] += 1; ex[k].append((a, b))
tot = sum(cls.values())
print(f"val words {words}; classified word edits {tot} (WER contribution {tot/words:.1%})")
for k, v in cls.most_common():
    print(f"  {v:4d} = {v/tot:5.1%} of edits = {v/words:.2%} WER  {k:38s} e.g. {ex[k][:5]}")
print(f"  (truncated/completed words that are the LAST word of the line: {last_word_trunc})")
