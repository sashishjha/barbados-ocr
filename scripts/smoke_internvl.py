"""CPU smoke test of the InternVL3.5 path in src/train_vlm.py (no GPU): build the LoRA model, run one
training step on 2 real lines (finite loss, gradients reach LoRA), then greedy-generate one line.
Use the 1B model so it fits the login node; the 8B differs only in size.
  PYTHONPATH=$PWD/.pylib_qwen3_checkpoint:$PWD HIP_VISIBLE_DEVICES= .venv/bin/python -m scripts.smoke_internvl
"""
import time

import torch

from src.data.datasets import SplitSpec, load_split, vlm_records
from src.train_vlm import OCR_PROMPT, build_model, image_kwargs, make_collate
from src.data.preprocess import vlm_thumbnail
from src.utils.common import load_config, parse_cli_overrides

cfg = load_config("configs/internvl35_8b_r14.yaml", parse_cli_overrides(
    ["model.id=OpenGVLab/InternVL3_5-1B-HF", "model.dtype=float32"]))
assert cfg["model"]["id"].endswith("1B-HF"), cfg["model"]["id"]   # never load the 8B by accident
t0 = time.time()
model, processor = build_model(cfg)
print(f"built {cfg['model']['id']} in {time.time()-t0:.0f}s; image token: {processor.image_token!r}")

d = cfg["data"]
tr, va = load_split(SplitSpec(d["folds_csv"], d["fold"], d["image_dir"]))
recs = vlm_records(tr.head(2), d["image_dir"])
kw = image_kwargs(cfg)
batch = [{"image": vlm_thumbnail(r["image"], **kw), "text": r["text"], "id": r["id"], "loss_weight": 1.0}
         for r in recs]
print("image sizes:", [b["image"].size for b in batch])
enc = make_collate(processor, train=True)(batch)
print("batch keys:", {k: tuple(v.shape) for k, v in enc.items() if torch.is_tensor(v)})
labels = enc.pop("labels"); enc.pop("ids"); enc.pop("texts"); enc.pop("loss_weight")
model.train()
out = model(**enc, labels=labels)
out.loss.backward()
g = [p.grad.abs().sum().item() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
print(f"loss {out.loss.item():.3f} (finite: {torch.isfinite(out.loss).item()}), LoRA tensors with grad: {len(g)}, "
      f"grad>0: {sum(x > 0 for x in g)}")

model.eval()
r = vlm_records(va.head(1), d["image_dir"])[0]
img = vlm_thumbnail(r["image"], **kw)
msgs = [{"role": "user", "content": [{"type": "text", "text": OCR_PROMPT}, {"type": "image", "image": img}]}]
prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp = processor(text=[prompt], images=[img], return_tensors="pt")
with torch.no_grad():
    gen = model.generate(**inp, max_new_tokens=40, do_sample=False)
txt = processor.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print(f"generated (untrained LoRA): {txt!r}\nreference: {r['text']!r}")
print(f"SMOKE OK in {time.time()-t0:.0f}s")
