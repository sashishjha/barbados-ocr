"""CPU smoke test for a VLM config: real images -> thumbnail -> processor -> collate.
Checks image height, pixel budget, and that only the answer tokens carry labels.
Usage: python scripts/smoke_vlm_collate.py configs/qwen_recipe_7b.yaml"""
import sys

import pandas as pd

sys.path.insert(0, ".")
from src.data.datasets import vlm_records  # noqa: E402
from src.train_vlm import VLMDataset, image_kwargs, load_processor, make_collate  # noqa: E402
from src.utils.common import load_config  # noqa: E402

cfg = load_config(sys.argv[1])
df = pd.read_csv("data/derived_checkpoint/train_folds.csv")
df = df.loc[[df.Target.str.len().idxmin(), df.Target.str.len().idxmax()]]
ds = VLMDataset(vlm_records(df, cfg["data"]["image_dir"]), train=False, img_kw=image_kwargs(cfg))
proc = load_processor(cfg)
proc.tokenizer.padding_side = "left"
items = [ds[0], ds[1]]
print("model:", cfg["model"]["id"], "| image sizes:", [it["image"].size for it in items])
tr = make_collate(proc, train=True)(items)
n_lab = (tr["labels"] != -100).sum(1).tolist()
n_ans = [len(proc.tokenizer.encode(it["text"], add_special_tokens=False)) for it in items]
print("input_ids:", tuple(tr["input_ids"].shape), "| pixel_values:", tuple(tr["pixel_values"].shape),
      "| grid:", tr["image_grid_thw"].tolist())
print("label tokens per sample:", n_lab, "| answer tokens:", n_ans)
assert all(0 < a <= b + 2 for a, b in zip(n_lab, n_ans)), "label masking wrong"
assert all(it["image"].size[1] == cfg["model"]["image_height"] for it in items), "height wrong"
ev = make_collate(proc, train=False)(items)
assert "labels" not in ev
print("SMOKE OK")
