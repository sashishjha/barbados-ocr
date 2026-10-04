"""Does averaging candidate log-probs over several image views (scripts/rescore_candidates.py --height/--pad) beat the
single training view?  Builds view-averaged lp files and reruns the select_candidates comparison on fold-0 val.
  .venv/bin/python -m scripts.multiview_eval
"""
import glob
import os
import sys

import pandas as pd

import scripts.select_candidates as sc
from scripts.tune_ensemble_r8 import score

P = "preds_checkpoint"
MODELS = {"q3": "q3_r8", "q7a": "q7_r8a", "q32": "q32"}


def averaged(name):
    base = pd.read_csv(f"{P}/lp_val_{name}.csv", dtype={"ID": str}, keep_default_na=False)[["ID", "Target", "logp"]]
    views = sorted(glob.glob(f"{P}/lpv_val_{name}_*.csv"))
    views = [v for v in views if len(pd.read_csv(v, usecols=["ID"])) >= len(base)]          # only completed views
    frames = [base] + [pd.read_csv(v, dtype={"ID": str}, keep_default_na=False)[["ID", "Target", "logp"]] for v in views]
    m = frames[0].rename(columns={"logp": "l0"})
    for i, f in enumerate(frames[1:], 1):
        m = m.merge(f.drop_duplicates(["ID", "Target"]).rename(columns={"logp": f"l{i}"}), on=["ID", "Target"], how="left")
    cols = [c for c in m.columns if c.startswith("l") and c[1:].isdigit()]
    m["logp"] = m[cols].mean(axis=1)
    m["ntok"] = 0
    m[["ID", "Target", "logp", "ntok"]].to_csv(f"{P}/lp_val_{name}_mv.csv", index=False)
    return [os.path.basename(v) for v in views]


def main():
    used = {k: averaged(n) for k, n in MODELS.items()}
    print("views averaged per model (plus the training view):", used)
    for slots, tag in ((["q3", "q7a", "q32"], "base view"), (["q3", "q7a", "q32"], "view-averaged")):
        sc.VAL_SCORERS = {k: (n + ("_mv" if tag == "view-averaged" else "")) for k, n in MODELS.items()}
        c, _ = sc.build("val", {s: sc.VAL_SCORERS[s] for s in slots})
        c = sc.attach_loss(c)
        print(f"\n== {tag} ==")
        for s in slots:
            p = {**{"w_" + k: float(k == s) for k in slots}, "b": 0.0, "lam": 0.0, "tau": 0.0}
            print(f"  argmax {s:4s} alone: {score(sc.choose(c, p, slots)[['ID', 'Target']], sc.REF):.4f}")
        p = {**{"w_" + k: 1.0 for k in slots}, "b": 0.0, "lam": 0.0, "tau": 0.0}
        print(f"  argmax equal sum:   {score(sc.choose(c, p, slots)[['ID', 'Target']], sc.REF):.4f}")
        oof = sc.cv(c, slots)
        print(f"  CV selector: {score(oof[['ID', 'Target']], sc.REF):.4f}  override {1 - oof.is_rover.mean():.1%}   (ROVER 0.9160)")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
