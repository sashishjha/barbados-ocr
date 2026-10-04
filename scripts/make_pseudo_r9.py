"""R9 pseudo-labels: fused ensemble output on Test, kept only where the arms agree.

Agreement per line = weighted mean normalised edit distance between the fused text and each
arm's top-1. The threshold is chosen on fold-0 val, where we can see how accurate the fused
text is at each agreement level, then applied to Test.

  .venv/bin/python -m scripts.make_pseudo_r9 --dir preds_checkpoint/r8_ens --out <csv>
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from scripts.tune_ensemble_r8 import fuse, load
from src.eval.edit_distance import levenshtein

# arm -> (val file, test file, weight); the best ensemble found by tune_ensemble_r8
ARMS = {
    "qwen3vl8b_r8": ("qwen3vl8b_val_nbest_1387.csv", "qwen3vl8b_test_nbest_1387.csv", 1.0),
    "qwen7b_r8b": ("qwen7b_b_val_nbest_1387.csv", "qwen7b_b_test_nbest_1387.csv", 1.0),
    "qwen_r5": ("valtop1_qwen25vl_lora_r5.csv", "vlm_test_nbest_r5_ensemble.csv", 0.5),
    "trocr_r5": ("valtop1_trocr_hw_r5.csv", "trocr_test_nbest_r5_ensemble.csv", 1.0),
}


def disagreement(cand: pd.DataFrame, fused: pd.DataFrame, weights: dict) -> pd.Series:
    f = fused.set_index("ID")["Target"]
    out = {}
    for rid, g in cand.groupby("ID"):
        t = f[rid]
        num = sum(weights[a] * levenshtein(x, t) / max(1, len(t)) for a, x in zip(g["arm"], g["Target"]))
        out[rid] = num / sum(weights[a] for a in g["arm"])
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="preds_checkpoint/r8_ens")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_line_cer", type=float, default=0.02,
                    help="keep the agreement band whose fused val lines average at most this CER")
    a = ap.parse_args()
    d = a.dir
    arms = {k: v for k, v in ARMS.items() if os.path.exists(os.path.join(d, v[1]))}
    w = {k: v[2] for k, v in arms.items()}
    print("arms:", w)

    val = pd.concat([load(os.path.join(d, v[0]), k, 1) for k, v in arms.items()])
    test = pd.concat([load(os.path.join(d, v[1]), k, 1) for k, v in arms.items()])
    ref = pd.read_csv(os.path.join(d, "val_ref_fold0.csv"), dtype={"ID": str}, keep_default_na=False).set_index("ID")["Target"]

    vf = fuse(val, w, "rover")
    vdis = disagreement(val, vf, w)
    vf = vf.set_index("ID")
    vcer = pd.Series({i: levenshtein(vf.at[i, "Target"], ref[i]) / max(1, len(ref[i])) for i in vf.index})

    # Largest agreement threshold whose kept val lines still average <= max_line_cer.
    order = vdis.sort_values().index
    cum = vcer[order].expanding().mean()
    ok = cum[cum <= a.max_line_cer]
    thr = float(vdis[ok.index[-1]]) if len(ok) else 0.0
    kept = vdis <= thr
    print(f"val: threshold {thr:.4f} keeps {kept.sum()}/{len(vdis)} lines, their mean CER "
          f"{vcer[kept].mean():.4f} (all lines {vcer.mean():.4f}, dropped {vcer[~kept].mean():.4f})")
    for q in (0.0, 0.01, 0.02, 0.05, 0.1):
        m = vdis <= q
        print(f"  dis<={q:<4}: {m.sum():4d} val lines, mean CER {vcer[m].mean() if m.any() else float('nan'):.4f}")

    tf = fuse(test, w, "rover")
    tdis = disagreement(test, tf, w)
    tf = tf.set_index("ID")
    keep_ids = tdis[tdis <= thr].index
    out = pd.DataFrame({"ID": keep_ids, "Target": tf.loc[keep_ids, "Target"].values,
                        "source": "pseudo_r9", "agreement_dis": tdis[keep_ids].round(4).values})
    out = out[out["Target"].str.strip() != ""].sort_values("agreement_dis")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"test: kept {len(out)}/{len(tdis)} lines -> {a.out}")


if __name__ == "__main__":
    main()
