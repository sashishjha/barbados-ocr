"""Score every candidate transcription of a line under a trained VLM (teacher forcing, no generation).

For each (ID, Target) row of --cands, computes the summed log-probability of the answer tokens (plus the
end-of-turn token, so truncated candidates are penalised) given the line image and the OCR prompt — the same
assistant-only span the model was trained on (src.train_vlm.make_collate).  The ensemble can then pick, per
line, the candidate the models jointly find most likely (scripts/select_candidates.py).

  PYTHONPATH=$PWD/.pylib_qwen3_checkpoint:$PWD python -m scripts.rescore_candidates \
      --config configs/qwen3vl_8b.yaml --adapter runs_checkpoint/BEST_qwen3vl8b_r8_f55_08997 \
      --cands preds_checkpoint/cands_val.csv --out preds_checkpoint/lp_val_q3_r8.csv
"""
import argparse
import copy
import logging
import os
import time

import pandas as pd
import torch

from src.data.preprocess import vlm_thumbnail
from src.train_vlm import image_kwargs, load_processor, make_collate, vlm_class
from src.utils.common import load_config

log = logging.getLogger("rescore")


@torch.no_grad()
def score_line_fast(model, prompt_collate, tok, tail, img, rid, texts, bsz):
    """Encode image + prompt once (prefill), then score all candidates as continuations of its KV cache."""
    enc = prompt_collate([{"image": img, "text": "", "id": rid, "loss_weight": 1.0}])
    enc = {k: v.to("cuda") for k, v in enc.items() if torch.is_tensor(v) and k != "loss_weight"}
    out = model(**enc, use_cache=True)
    first_lp = torch.log_softmax(out.logits[0, -1].float(), -1)          # predicts the first answer token
    past, L = out.past_key_values, enc["input_ids"].shape[1]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    rows = []
    for s in range(0, len(texts), bsz):
        chunk = texts[s:s + bsz]
        ids = [tok.encode(t, add_special_tokens=False) + tail for t in chunk]
        T = max(map(len, ids))
        inp = torch.full((len(ids), T), pad, dtype=torch.long, device="cuda")
        m = torch.zeros((len(ids), T), dtype=torch.long, device="cuda")
        for j, x in enumerate(ids):
            inp[j, :len(x)] = torch.tensor(x, device="cuda")
            m[j, :len(x)] = 1
        cache = copy.deepcopy(past)
        cache.batch_repeat_interleave(len(ids))
        attn = torch.cat([torch.ones((len(ids), L), dtype=torch.long, device="cuda"), m], 1)
        o = model(input_ids=inp, attention_mask=attn, past_key_values=cache, use_cache=True,
                  cache_position=torch.arange(L, L + T, device="cuda"))
        lp_rest = torch.log_softmax(o.logits[:, :-1].float(), -1).gather(-1, inp[:, 1:].unsqueeze(-1)).squeeze(-1)
        lp = first_lp[inp[:, 0]] + (lp_rest * m[:, 1:]).sum(1)
        for t, v, n in zip(chunk, lp.tolist(), m.sum(1).tolist()):
            rows.append({"ID": rid, "Target": t, "logp": v, "ntok": n})
        del cache, o
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--cands", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--attn", default="sdpa", help="sdpa: vision attention per image (eager is quadratic in batch)")
    ap.add_argument("--mode", default="fast", choices=["fast", "slow"],
                    help="fast: image+prompt encoded once per line, candidates scored from its KV cache; "
                         "slow: one full forward per candidate (reference implementation)")
    ap.add_argument("--height", type=int, default=0, help="image view: override line height in px (0 = training value)")
    ap.add_argument("--pad", type=float, default=-1.0, help="image view: override vertical pad fraction (-1 = training value)")
    ap.add_argument("--blur", type=float, default=0.0, help="image view: Gaussian blur sigma (px at the resized height) - destroys letter identity, keeps layout")
    ap.add_argument("--fast_batch", type=int, default=1, help="candidates per cached forward; >1 pads, which shifts scores in transformers 4.49")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N lines (testing)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    from peft import PeftModel
    cfg = load_config(a.config)
    mid = cfg["model"]["id"]
    if "Qwen2.5" in mid:   # its vision attention (eager and, in our transformers, sdpa) is quadratic in images per batch (8 -> 105 GB)
        a.batch_size = min(a.batch_size, 2)
    base = vlm_class(mid).from_pretrained(mid, torch_dtype=getattr(torch, cfg["model"].get("dtype", "bfloat16")),
                                          attn_implementation=a.attn)
    model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload().to("cuda").eval()
    processor = load_processor(cfg)
    processor.tokenizer.padding_side = "right"
    collate = make_collate(processor, train=True)
    kw = image_kwargs(cfg)
    if a.height:
        kw["target_height"] = a.height
    if a.pad >= 0:
        kw["vertical_pad_frac"] = a.pad
    log.info("image view kwargs: %s", kw)
    image_dir = cfg["data"]["image_dir"]

    cands = pd.read_csv(a.cands, dtype={"ID": str}, keep_default_na=False)
    cands = cands[cands["Target"].str.strip() != ""].drop_duplicates(["ID", "Target"]).reset_index(drop=True)
    done = set()
    if os.path.exists(a.out):                      # resume after a crash
        prev = pd.read_csv(a.out, dtype={"ID": str}, keep_default_na=False)
        done = set(zip(prev["ID"], prev["Target"]))
    todo = cands[[(i, t) not in done for i, t in zip(cands["ID"], cands["Target"])]]
    if a.limit:
        todo = todo[todo["ID"].isin(todo["ID"].unique()[:a.limit])]
    prompt_collate = make_collate(processor, train=False)
    tok = processor.tokenizer
    tail = tok.encode("<|im_end|>\n", add_special_tokens=False)   # scored like in training: answer + end of turn
    log.info("%s: %d candidates for %d lines (%d already scored)", a.adapter, len(cands), cands["ID"].nunique(), len(done))

    t0, n_done, first = time.time(), 0, not os.path.exists(a.out)
    for rid, g in todo.groupby("ID", sort=False):
        img = vlm_thumbnail(os.path.join(image_dir, f"{rid}.jpg"), **kw)
        if a.blur > 0:
            from PIL import ImageFilter
            img = img.filter(ImageFilter.GaussianBlur(a.blur))
        rows = []
        texts = g["Target"].tolist()
        if a.mode == "fast":
            rows = score_line_fast(model, prompt_collate, tok, tail, img, rid, texts, a.fast_batch)
        for s in range(0, len(texts) if a.mode == "slow" else 0, a.batch_size):
            chunk = texts[s:s + a.batch_size]
            enc = collate([{"image": img, "text": t, "id": rid, "loss_weight": 1.0} for t in chunk])
            labels = enc.pop("labels").to("cuda")
            for k in ("ids", "texts", "loss_weight"):
                enc.pop(k)
            enc = {k: v.to("cuda") for k, v in enc.items() if torch.is_tensor(v)}
            with torch.no_grad():
                logits = model(**enc).logits[:, :-1].float()
            tgt = labels[:, 1:]
            mask = tgt != -100
            lp = torch.log_softmax(logits, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            lp = (lp * mask).sum(1)
            for t, v, n in zip(chunk, lp.tolist(), mask.sum(1).tolist()):
                rows.append({"ID": rid, "Target": t, "logp": v, "ntok": n})
            del logits
        pd.DataFrame(rows).to_csv(a.out, mode="a", header=first, index=False)
        first = False
        n_done += len(rows)
        if n_done % 500 < len(rows):
            rate = n_done / (time.time() - t0)
            log.info("%d/%d candidates  %.1f/s  ETA %.0f min", n_done, len(todo), rate, (len(todo) - n_done) / rate / 60)
    log.info("RESCORE DONE %s", a.out)


if __name__ == "__main__":
    main()
