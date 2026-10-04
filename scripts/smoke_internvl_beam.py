"""Does beam search (num_beams=3, num_return_sequences=3) work with InternVL's multi-tile pixel_values?
Compares greedy vs beam outputs of the UNTRAINED 1B model on 2 real lines; if tiles were scrambled across
beams, the beam outputs would be garbage/unrelated to greedy.  CPU, 1B model, inside our SLURM allocation.
  PYTHONPATH=$PWD/.pylib_qwen3_checkpoint:$PWD HIP_VISIBLE_DEVICES= .venv/bin/python -m scripts.smoke_internvl_beam
"""
import sys

import torch
from transformers import AutoProcessor

from src.data.datasets import SplitSpec, load_split, vlm_records
from src.data.preprocess import vlm_thumbnail
from src.train_vlm import OCR_PROMPT, image_kwargs, load_processor, vlm_class
from src.utils.common import load_config, parse_cli_overrides

cfg = load_config("configs/internvl35_8b_r14.yaml", parse_cli_overrides(
    ["model.id=OpenGVLab/InternVL3_5-1B-HF", "model.dtype=float32"] + sys.argv[1:]))   # extra overrides from argv
assert cfg["model"]["id"].endswith("1B-HF")
model = vlm_class(cfg["model"]["id"]).from_pretrained(cfg["model"]["id"], torch_dtype=torch.float32,
                                                      attn_implementation="eager").eval()
proc = load_processor(cfg)
proc.tokenizer.padding_side = "left"
d = cfg["data"]
tr, va = load_split(SplitSpec(d["folds_csv"], d["fold"], d["image_dir"]))
kw = image_kwargs(cfg)
for r in vlm_records(va.head(2), d["image_dir"]):
    img = vlm_thumbnail(r["image"], **kw)
    prompt = proc.apply_chat_template([{"role": "user", "content": [
        {"type": "text", "text": OCR_PROMPT}, {"type": "image", "image": img}]}], tokenize=False, add_generation_prompt=True)
    enc = proc(text=[prompt], images=[img], padding=True, return_tensors="pt")
    n = enc["input_ids"].shape[1]
    print(f"\nref   : {r['text']!r}  (pixel_values tiles: {enc['pixel_values'].shape[0]})")
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=40, do_sample=False, num_beams=1)
        b = model.generate(**enc, max_new_tokens=40, do_sample=False, num_beams=3, num_return_sequences=3,
                           output_scores=True, return_dict_in_generate=True)
    print("greedy:", repr(proc.batch_decode(g[:, n:], skip_special_tokens=True)[0]))
    for k, t in enumerate(proc.batch_decode(b.sequences[:, n:], skip_special_tokens=True)):
        print(f"beam{k} :", repr(t), f"(score {b.sequences_scores[k].item():.3f})")
print("BEAM SMOKE DONE")
