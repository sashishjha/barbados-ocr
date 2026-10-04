"""T2: prior-corrected likelihood test.  Needs preds_checkpoint/{lp_val_q3_r8,t2_refs_normal,t2_pool_blur}.csv"""
import sys, tempfile, os
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from scripts.tune_ensemble_r8 import fuse, load, prefilter_candidates
from src.eval.eval_metrics import evaluate_set
D = "preds_checkpoint"; d = f"{D}/r8_ens"
rd = lambda f: pd.read_csv(f, dtype={"ID": str}, keep_default_na=False)
tf = rd(f"{d}/train_folds.csv"); S = np.load(os.environ.get("IMG_STATS", "preds_checkpoint/img_stats_train.npy"))
tf["h"] = np.exp(S[:, 0]); va = tf[tf.fold == 0]; tall = set(va[va.h > 200].ID); short = set(va.ID) - tall
R = rd(f"{d}/val_ref_fold0.csv"); ref = R.set_index("ID").Target
N = pd.concat([rd(f"{D}/lp_val_q3_r8.csv")[["ID", "Target", "logp"]], rd(f"{D}/t2_refs_normal.csv")[["ID", "Target", "logp"]]]).drop_duplicates(["ID", "Target"])
B = rd(f"{D}/t2_pool_blur.csv")[["ID", "Target", "logp"]].drop_duplicates(["ID", "Target"])
M = N.merge(B, on=["ID", "Target"], suffixes=("_n", "_b"))
key = {(i, t): (n, b) for i, t, n, b in zip(M.ID, M.Target, M.logp_n, M.logp_b)}
meta = rd(f"{d}/cands_val_meta.csv"); rov = meta[meta.arm == "rover"].drop_duplicates("ID").set_index("ID").Target
def sc(pred, ids):
    p = pred[pred.ID.isin(ids)]; r = R[R.ID.isin(ids)]
    a = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False); p.to_csv(a.name, index=False)
    b = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False); r.to_csv(b.name, index=False)
    e = evaluate_set(b.name, a.name, verbose=False)["final_05_05"]; os.unlink(a.name); os.unlink(b.name); return e
print("== classification of ROVER's wrong lines (ref != ROVER) ==")
for nm, ids in (("TALL", tall), ("short", short)):
    c = {"search failure (model prefers the reference)": 0, "prior failure (PMI prefers the reference)": 0, "feature failure": 0}; n = 0
    for i in ids:
        r_, y = ref[i], rov[i]
        if r_ == y or (i, r_) not in key or (i, y) not in key: continue
        n += 1; (rn, rb), (yn, yb) = key[(i, r_)], key[(i, y)]
        if rn > yn: c["search failure (model prefers the reference)"] += 1
        elif (rn - rb) > (yn - yb): c["prior failure (PMI prefers the reference)"] += 1
        else: c["feature failure"] += 1
    print(f"  {nm:5s} n={n}: " + " | ".join(f"{k.split(' (')[0]} {v / n:.0%}" for k, v in c.items()))
print("\n== PMI reranking of the pool: argmax_c [ logP(c|x) - lambda * logP(c|blur) ]  (reference NOT a candidate) ==")
pool = rd(f"{D}/cands_val.csv"); pool = pool[[(i, t) in key for i, t in zip(pool.ID, pool.Target)]].copy()
pool["n"] = [key[(i, t)][0] for i, t in zip(pool.ID, pool.Target)]; pool["b"] = [key[(i, t)][1] for i, t in zip(pool.ID, pool.Target)]
rv = rov.reset_index().rename(columns={"Target": "Target"})[["ID", "Target"]]
print(f"  ROVER                 TALL {sc(rv, tall):.4f}  short {sc(rv, short):.4f}  all {sc(rv, tall | short):.4f}")
for lam in (0.0, 0.3, 0.5, 0.7, 1.0):
    pool["s"] = pool.n - lam * pool.b
    pick = pool.loc[pool.groupby("ID").s.idxmax(), ["ID", "Target"]]
    print(f"  PMI lambda={lam:.1f}      TALL {sc(pick, tall):.4f}  short {sc(pick, short):.4f}  all {sc(pick, tall | short):.4f}")
