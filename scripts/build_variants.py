"""Build weighted/subset variants of the 0.9221 ROVER recipe as guarded submission files (CPU only).
  .venv/bin/python -m scripts.build_variants NAME [NAME ...]
"""
import subprocess
import sys

import pandas as pd

from scripts.tune_ensemble_r8 import fuse, guard_runaway, is_runaway, load, prefilter_candidates

D = "preds_checkpoint/r8_ens"
FILES = {
    "q32": "qwen3vl32b_r10_test_nbest_1478.csv", "q3_all": "qwen3vl8b_r10all_test_nbest_1478.csv",
    "q7_all": "qwen7b_r10all_test_nbest_1478.csv", "q7_r8b": "qwen7b_b_test_nbest_1387.csv",
    "trocr_r5": "trocr_test_nbest_r5_ensemble.csv", "q3_s7": "qwen3vl8b_r11all_s7_test_nbest_1603.csv",
    "trocr_all": "trocr_r13all_test_nbest_1635.csv", "q7_s7": "qwen7b_r12all_s7_test_nbest_1603.csv",
}
BASE = {"q32": 1, "q3_all": 1, "q7_all": 1, "q7_r8b": 1, "trocr_r5": 1, "q3_s7": 1}      # = 0.9221 file
VARIANTS = {
    "no_trocr":    {k: v for k, v in BASE.items() if k != "trocr_r5"},
    "no_q7r8b":    {k: v for k, v in BASE.items() if k != "q7_r8b"},
    "q32_x2":      {**BASE, "q32": 2},
    "allfam_x1.5": {**BASE, "q3_all": 1.5, "q7_all": 1.5},
    "trocr_x2":    {**BASE, "trocr_r5": 2},
    "q3pair_half": {**BASE, "q3_all": 0.5, "q3_s7": 0.5},
    "trocr_x3":    {**BASE, "trocr_r5": 3},
    "trocr_x2_q32x2": {**BASE, "trocr_r5": 2, "q32": 2},
}


def build(name):
    w = VARIANTS[name]
    cand = pd.concat([load(f"{D}/{FILES[k]}", k, 1) for k in w])
    pf = prefilter_candidates(cand)
    g, changes = guard_runaway(pf, fuse(pf, w, "rover"))
    out = f"{D}/test_var_{name}.csv"
    g.to_csv(out, index=False)
    sub = f"submissions/sub_20261003_var_{name}.csv"
    subprocess.run([sys.executable, "-m", "src.make_submission", "--test_csv",
                    "road-barbados-historic-handwriting-challenge20260703-3588-9x1pqe/Test.csv", "--pred_csv", out,
                    "--out_csv", sub, "--fallbacks", f"{D}/test_top1_qwen3vl8b.csv", "--train_csv", f"{D}/train_and_pseudo_r5.csv",
                    "--method", f"var_{name}", "--notes", f"0.9221 recipe variant {name}: {w}; prefilter+guard"],
                   check=True, capture_output=True)
    best = pd.read_csv("submissions/sub_20261002_rover6_s7.csv", dtype=str, keep_default_na=False).set_index("ID").Target
    n = pd.read_csv(sub, dtype=str, keep_default_na=False).set_index("ID").Target.reindex(best.index)
    print(f"{name}: {sub} | differs from 0.9221 file on {int((n != best).sum())} lines | runaways {int(n.map(is_runaway).sum())} | "
          f">10 chars longer {int(((n.str.len() - best.str.len()) > 10).sum())} | guard {len(changes)}", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or VARIANTS:
        build(name)
