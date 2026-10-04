"""Standard evaluation of a new fold-0 model as an ensemble member.
  .venv/bin/python -m scripts.eval_member NAME preds_checkpoint/r8_ens/<arm>_val_nbest_<job>.csv
Reports: standalone (all / tall>200px / short), then the 5-member val ensemble (q32,q3_r8,q7_r8a,q7_r8b,trocr) vs +member vs
swap-for-q3_r8, under sqrt-length and micro weighting.  Decision rule used by the heartbeat: member is worth a submission if
(+member or swap) improves the 5-set by >= +0.003 (sqrt-length) without hurting short lines."""
import sys, tempfile, os
import numpy as np, pandas as pd
from scripts.tune_ensemble_r8 import load, fuse, prefilter_candidates
from src.eval.eval_metrics import evaluate_set
name, path = sys.argv[1], sys.argv[2]
d = "preds_checkpoint/r8_ens"; ref = f"{d}/val_ref_fold0.csv"
R = pd.read_csv(ref, dtype={"ID": str}, keep_default_na=False)
tall = set(pd.read_csv("preds_checkpoint/val_tall_ids.csv", dtype={"ID": str}).ID); short = set(R.ID) - tall
def sc(df, ids, scheme):
    p = df[df.ID.isin(ids)][["ID", "Target"]]; r = R[R.ID.isin(ids)]
    a = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False); p.to_csv(a.name, index=False)
    b = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False); r.to_csv(b.name, index=False)
    e = evaluate_set(b.name, a.name, scheme=scheme, verbose=False)["final_05_05"]; os.unlink(a.name); os.unlink(b.name); return e
F = {"q32": "qwen3vl32b_r10_val_nbest_1478.csv", "q3_r8": "qwen3vl8b_val_nbest_1387.csv", "q7_r8a": "qwen7b_a_val_nbest_1387.csv",
     "q7_r8b": "qwen7b_b_val_nbest_1387.csv", "trocr": "valtop1_trocr_hw_r5.csv"}
cand = pd.concat([load(f"{d}/{f}", k, 1) for k, f in F.items()] + [load(path, name, 1)])
pf = prefilter_candidates(cand)
me = pf[pf.arm == name][["ID", "Target"]]
allids = tall | short
print(f"{name} standalone (top-1): all {sc(me, allids, 'length_pow'):.4f} (micro {sc(me, allids, 'micro'):.4f}) | tall {sc(me, tall, 'length_pow'):.4f} | short {sc(me, short, 'length_pow'):.4f}")
base = ["q32", "q3_r8", "q7_r8a", "q7_r8b", "trocr"]
rows = {"5-set (current)": base, f"+ {name}": base + [name], f"swap q3_r8 -> {name}": [m for m in base if m != "q3_r8"] + [name],
        f"swap q7_r8b -> {name}": [m for m in base if m != "q7_r8b"] + [name]}
print(f"{'ensemble':28s} {'sqrt-len':>9s} {'micro':>8s} {'tall':>8s} {'short':>8s}")
for n, m in rows.items():
    r = fuse(pf[pf.arm.isin(m)], {k: 1 for k in m}, "rover")
    print(f"{n:28s} {sc(r, allids, 'length_pow'):9.4f} {sc(r, allids, 'micro'):8.4f} {sc(r, tall, 'length_pow'):8.4f} {sc(r, short, 'length_pow'):8.4f}")
