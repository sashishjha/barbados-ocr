"""
Unified inference for all three arms. Emits an n-best CSV so the ensemble and
the pseudo-label filter have candidate-level evidence, not just a top-1 string.

Output schema (one row per image per candidate):
    ID, rank, Target, score, arm

Usage:
    python -m src.infer --config configs/trocr_large.yaml \
        --checkpoint runs_checkpoint/trocr_large_checkpoints/best_checkpoint \
        --manifest data/Test.csv --out preds_checkpoint/trocr_test.csv --nbest 5
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.datasets import LineOCRDataset
from src.data.preprocess import PreprocessConfig, vlm_thumbnail
from src.utils.common import get_device, get_logger, load_config, parse_cli_overrides, set_seed

log = get_logger("infer")


def load_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "ID" not in df.columns:
        for c in ("id", "new_id", "trapp_id"):
            if c in df.columns:
                df = df.rename(columns={c: "ID"})
                break
    if "Target" not in df.columns:
        df["Target"] = ""
    return df[["ID", "Target"]]


# ---------------------------------------------------------------- TrOCR
@torch.no_grad()
def infer_trocr(cfg, checkpoint, df, device, nbest, batch_size):
    import transformers.modeling_utils
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    model_id = cfg["model"]["id"]
    try:
        processor = TrOCRProcessor.from_pretrained(model_id)
    except Exception:
        from transformers import AutoImageProcessor, AutoTokenizer
        feature_extractor = AutoImageProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(
        model_id, attn_implementation=cfg["model"].get("attn_implementation", "eager"))

    if cfg["model"].get("use_lora", False):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()   # merge only at inference (storage rule)
    else:
        sd = torch.load(os.path.join(checkpoint, "model.pt"), map_location="cpu")
        model.load_state_dict(sd)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.to(device).eval()
    
    # Meta tensor fix for TrOCR positional embeddings
    from transformers.models.trocr.modeling_trocr import TrOCRSinusoidalPositionalEmbedding
    for module in model.modules():
        if isinstance(module, TrOCRSinusoidalPositionalEmbedding):
            n_pos = module.weights.shape[0] if module.weights is not None else 1026
            module.weights = TrOCRSinusoidalPositionalEmbedding.get_embedding(
                n_pos, module.embedding_dim, module.padding_idx
            ).to(device)
            log.info("infer: recomputed TrOCRSinusoidalPositionalEmbedding.weights on %s", device)


    pre = PreprocessConfig(**cfg["preprocess"])
    ds = LineOCRDataset(df, cfg["data"]["image_dir"], pre, None, None, None, train=False)

    def collate(b):
        return {
            "pv": processor(images=[x["image"] for x in b], return_tensors="pt").pixel_values,
            "ids": [x["id"] for x in b],
        }

    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate,
                        num_workers=cfg["train"].get("num_workers", 8))
    gen = cfg.get("generate", {})
    beams = max(gen.get("num_beams", 5), nbest)

    rows: List[dict] = []
    for batch in loader:
        pv = batch["pv"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model.generate(
                pv, num_beams=beams, num_return_sequences=nbest,
                max_length=gen.get("max_length", 96),
                length_penalty=gen.get("length_penalty", 1.0),
                early_stopping=True, output_scores=True,
                return_dict_in_generate=True,
            )
        texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
        scores = (out.sequences_scores.detach().float().cpu().tolist()
                  if getattr(out, "sequences_scores", None) is not None
                  else [0.0] * len(texts))
        for i, rid in enumerate(batch["ids"]):
            for k in range(nbest):
                j = i * nbest + k
                rows.append({"ID": rid, "rank": k,
                             "Target": " ".join(str(texts[j]).split()),
                             "score": scores[j], "arm": "trocr"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- VLM
@torch.no_grad()
def infer_vlm(cfg, checkpoint, df, device, nbest, batch_size):
    if os.environ.get("GPU_MEM_FRACTION"):   # hard cap on this process's GPU memory when the GPU is shared
        torch.cuda.set_per_process_memory_fraction(float(os.environ["GPU_MEM_FRACTION"]), 0)
    from peft import PeftModel

    from src.data.preprocess import style_tag
    from src.train_vlm import image_kwargs, load_processor, prompt_text, vlm_class

    model_id = cfg["model"]["id"]
    base = vlm_class(model_id).from_pretrained(
        model_id, torch_dtype=getattr(torch, cfg["model"].get("dtype", "float32")),
        attn_implementation=cfg["model"].get("attn_implementation", "eager"),
        trust_remote_code=True)
    model = PeftModel.from_pretrained(base, checkpoint)
    from src.utils.ckpt_manager import load_visual_full
    load_visual_full(model, checkpoint)          # R19 checkpoints carry a fully fine-tuned vision tower
    model = model.merge_and_unload().to(device).eval()
    processor = load_processor(cfg)
    img_kw = image_kwargs(cfg)

    # FIX: Qwen2.5-VL is a decoder-only model. Right-padding causes garbage
    # generation because attention mask is wrong for the prompt tokens.
    # Must use left-padding for batched inference.
    processor.tokenizer.padding_side = "left"

    gen = cfg.get("generate", {})
    beams = max(gen.get("num_beams", 5), nbest)
    image_dir = cfg["data"]["image_dir"]

    rows: List[dict] = []
    ids = df["ID"].tolist()
    for s in range(0, len(ids), batch_size):
        chunk = ids[s:s + batch_size]
        imgs, styles = [], []
        for rid in chunk:
            p = os.path.join(image_dir, f"{rid}.jpg")
            if os.path.exists(p):
                imgs.append(vlm_thumbnail(p, **img_kw))
                styles.append(style_tag(p) if cfg["model"].get("style_prompt") else "")
            else:
                from PIL import Image
                imgs.append(Image.new("RGB", (512, 96), (255, 255, 255)))
                styles.append("")

        prompts = [
            processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "text", "text": prompt_text(st)},
                    {"type": "image", "image": im}]}],
                tokenize=False, add_generation_prompt=True)
            for im, st in zip(imgs, styles)
        ]
        enc = processor(text=prompts, images=imgs, padding=True, return_tensors="pt")
        input_len = enc["input_ids"].shape[1]
        enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
        out = model.generate(
            **enc, max_new_tokens=gen.get("max_new_tokens", 128),
            num_beams=beams, num_return_sequences=nbest, do_sample=False,
            length_penalty=gen.get("length_penalty", 1.0),
            output_scores=True, return_dict_in_generate=True)

        # FIX: strip the input prompt tokens from the generated sequences before
        # decoding. Role-tag splitting was fragile and could eat real OCR content.
        generated_tokens = out.sequences[:, input_len:]
        texts = processor.batch_decode(generated_tokens, skip_special_tokens=True)

        scores = (out.sequences_scores.detach().float().cpu().tolist()
                  if getattr(out, "sequences_scores", None) is not None
                  else [0.0] * len(texts))
        for i, rid in enumerate(chunk):
            for k in range(nbest):
                j = i * nbest + k
                rows.append({"ID": rid, "rank": k, "Target": " ".join(str(texts[j]).split()),
                             "score": scores[j], "arm": "vlm"})
        if (s // batch_size) % 20 == 0:
            log.info("vlm %d/%d", s, len(ids))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- CTC
@torch.no_grad()
def infer_ctc(cfg, checkpoint, df, device, nbest, batch_size):
    from src.models.crnn_ctc import CRNN, CharTokenizer, ctc_beam_search

    tokenizer = CharTokenizer.load(os.path.join(cfg["output"]["dir"], "charset.json"))
    model = CRNN(n_classes=len(tokenizer), hidden=cfg["model"].get("hidden", 384),
                 layers=cfg["model"].get("layers", 3))
    model.load_state_dict(torch.load(os.path.join(checkpoint, "model.pt"), map_location="cpu"))
    model.to(device).eval()

    pre = PreprocessConfig(**cfg["preprocess"])
    ds = LineOCRDataset(df, cfg["data"]["image_dir"], pre, None, None, None, train=False)

    def collate(b):
        xs = [torch.from_numpy(np.asarray(x["image"].convert("L"), np.float32) / 255.0)[None]
              for x in b]
        return {"x": torch.stack(xs), "ids": [x["id"] for x in b]}

    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate,
                        num_workers=cfg["train"].get("num_workers", 8))
    bw = cfg.get("generate", {}).get("beam_width", 10)

    rows: List[dict] = []
    for batch in loader:
        x = batch["x"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lp = model.logprobs_btc(x).float().cpu()
        for i, rid in enumerate(batch["ids"]):
            cands = ctc_beam_search(lp[i], tokenizer, beam_width=bw, top_k=nbest)
            for k, (text, sc) in enumerate(cands):
                rows.append({"ID": rid, "rank": k, "Target": " ".join(text.split()),
                             "score": float(sc), "arm": "ctc"})
    return pd.DataFrame(rows)


ARMS = {"trocr": infer_trocr, "vlm": infer_vlm, "ctc": infer_ctc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True, help="CSV with an ID column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default=None, help="trocr | vlm | ctc (default: cfg.arm)")
    ap.add_argument("--nbest", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--top1_out", default=None, help="Also write an ID,Target top-1 CSV")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, parse_cli_overrides(args.set))
    set_seed(cfg.get("seed", 42))
    arm = args.arm or cfg.get("arm")
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {list(ARMS)} (got {arm})")

    device = get_device()
    df = load_manifest(args.manifest)
    bs = args.batch_size or cfg["train"].get("eval_batch_size", 16)
    log.info("arm=%s  images=%d  nbest=%d  ckpt=%s", arm, len(df), args.nbest, args.checkpoint)

    out = ARMS[arm](cfg, args.checkpoint, df, device, args.nbest, bs)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    log.info("wrote %d candidate rows -> %s", len(out), args.out)

    if args.top1_out:
        top1 = out[out["rank"] == 0][["ID", "Target"]]
        # Guarantee coverage of every manifest ID.
        top1 = df[["ID"]].merge(top1, on="ID", how="left")
        top1["Target"] = top1["Target"].fillna("")
        top1.to_csv(args.top1_out, index=False)
        log.info("wrote top-1 -> %s", args.top1_out)


if __name__ == "__main__":
    main()
