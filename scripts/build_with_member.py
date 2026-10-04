"""Build a guarded submission = the 0.9221 recipe plus (or with a swapped-in) new member, from its TEST n-best file.
  .venv/bin/python -m scripts.build_with_member TAG preds_checkpoint/r8_ens/<arm>_test_nbest_<job>.csv [--swap q3_all|q3_s7|q7_all|q7_r8b|q32|trocr_r5] [--weight 1]
Writes submissions/sub_<date>_<TAG>.csv (prefilter + guard + checks).  Never uploads."""
import argparse, subprocess, sys, time
import pandas as pd
from scripts.build_variants import BASE, D, FILES
from scripts.tune_ensemble_r8 import fuse, guard_runaway, is_runaway, load, prefilter_candidates
ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("test_nbest"); ap.add_argument("--swap"); ap.add_argument("--weight", type=float, default=1.0); ap.add_argument("--also", action="append", default=[], help="extra member NAME=path_to_test_nbest (repeatable)")
a = ap.parse_args()
w = dict(BASE)
if a.swap: w.pop(a.swap)
w["NEW"] = a.weight
extra = {}
for kv in a.also:
    k, v = kv.split("=", 1); extra[k] = v; w[k] = 1.0
cand = pd.concat([load(f"{D}/{FILES[k]}", k, 1) for k in w if k != "NEW" and k not in extra] + [load(a.test_nbest, "NEW", 1)] + [load(v, k, 1) for k, v in extra.items()])
pf = prefilter_candidates(cand); g, ch = guard_runaway(pf, fuse(pf, w, "rover"))
out = f"{D}/test_member_{a.tag}.csv"; g.to_csv(out, index=False)
sub = f"submissions/sub_{time.strftime('%Y%m%d')}_{a.tag}.csv"
subprocess.run([sys.executable, "-m", "src.make_submission", "--test_csv", "road-barbados-historic-handwriting-challenge20260703-3588-9x1pqe/Test.csv",
                "--pred_csv", out, "--out_csv", sub, "--fallbacks", f"{D}/test_top1_qwen3vl8b.csv", "--train_csv", f"{D}/train_and_pseudo_r5.csv",
                "--method", a.tag, "--notes", f"0.9221 recipe {'minus ' + a.swap if a.swap else ''} plus {a.tag}: {a.test_nbest}"], check=True, capture_output=True)
best = pd.read_csv("submissions/sub_20261002_rover6_s7.csv", dtype=str, keep_default_na=False).set_index("ID").Target
n = pd.read_csv(sub, dtype=str, keep_default_na=False).set_index("ID").Target.reindex(best.index)
print(f"{sub}: rows {len(n)} | differs from 0.9221 file on {int((n != best).sum())} lines | runaways {int(n.map(is_runaway).sum())} | >10 chars longer {int(((n.str.len() - best.str.len()) > 10).sum())} | guard {len(ch)}")
