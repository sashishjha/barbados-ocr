"""Final-day builder: the 0.9221 recipe with every Qwen all-data member replaced by its vision-LoRA r128 twin.
  .venv/bin/python -m scripts.build_final
Writes submissions/sub_<date>_final_<tag>.csv (prefilter + runaway guard + format checks).  Never uploads."""
import os, subprocess, sys, time
import pandas as pd
from scripts.build_variants import FILES
from scripts.tune_ensemble_r8 import fuse, guard_runaway, is_runaway, load, prefilter_candidates

D = os.environ.get("MEMBER_DIR", "artifacts/members")   # member test n-best files (committed in artifacts/members)
FILES = {**FILES, "q3_vis": "r17_vis128_all_test_nbest_1719.csv", "q3_vis_s7": "r18_q3_vis128_all_s7_test_nbest_1721.csv",
         "q7_vis": "r18_q7_vis128_all_test_nbest_1721.csv"}
VIS3 = {"q32": 1, "q3_vis": 1, "q7_vis": 1, "q7_r8b": 1, "trocr_r5": 1, "q3_vis_s7": 1}   # 1:1 swaps of q3_all, q7_all, q3_s7
VARIANTS = {"vis3": VIS3, "vis3_trocr2": {**VIS3, "trocr_r5": 2}}
REF = "submissions/sub_20261002_rover6_s7.csv"   # earlier submission to diff against (optional; not in the repo)
best = pd.read_csv(REF, dtype=str, keep_default_na=False).set_index("ID").Target if os.path.exists(REF) else None
for tag, w in VARIANTS.items():
    pf = prefilter_candidates(pd.concat([load(f"{D}/{FILES[k]}", k, 1) for k in w]))
    g, ch = guard_runaway(pf, fuse(pf, w, "rover"))
    out = f"{D}/test_final_{tag}.csv"; g.to_csv(out, index=False)
    sub = f"submissions/sub_{time.strftime('%Y%m%d')}_final_{tag}.csv"
    subprocess.run([sys.executable, "-m", "src.make_submission", "--test_csv", "data/Test.csv",
                    "--pred_csv", out, "--out_csv", sub, "--fallbacks", f"{D}/test_top1_qwen3vl8b.csv", "--train_csv", "data/derived_checkpoint/train_clean.csv",
                    "--method", f"final_{tag}", "--notes", f"0.9221 recipe, Qwen members -> vision-LoRA r128 twins: {w}"], check=True, capture_output=True)
    n = pd.read_csv(sub, dtype=str, keep_default_na=False).set_index("ID").Target
    msg = f"{sub}: rows {len(n)} | empty {int((n.str.strip() == '').sum())} | runaways {int(n.map(is_runaway).sum())} | guard {len(ch)}"
    if best is not None:
        b = best.reindex(n.index)
        msg += f" | differs from {REF} on {int((n != b).sum())} lines | >10 chars longer {int(((n.str.len() - b.str.len()) > 10).sum())}"
    print(msg)
