"""
Arm B - open VLM fine-tuned with bf16 LoRA (Qwen2.5-VL-3B-Instruct, Apache-2.0).

This is the organizers' VLM starter, adapted for the ROCm/MI300X cluster:

  DROPPED from the starter   : BitsAndBytesConfig 4-bit QLoRA and
                               prepare_model_for_kbit_training. bitsandbytes
                               4-bit is fragile on ROCm and unnecessary at
                               192 GB HBM per GPU.
  KEPT from the starter      : the data-loading contract, the exact OCR prompt
                               ("Transcribe EXACTLY as seen. ..."), the
                               aspect-ratio thumbnail bucketing, and the
                               assistant-only loss masking idea.
  ADDED                      : plain bf16 peft.LoraConfig + get_peft_model,
                               checkpoint/resume, per-sample loss weighting for
                               pseudo-labels, long-line curriculum, weighted-
                               metric validation, SDPA attention.

LICENSING: base Qwen2.5-VL-3B-Instruct is Apache-2.0. CHURRO / churro-3B is
Qwen Research License and was OFFICIALLY REJECTED by the organizers - it is
never referenced here. Verify the exact checkpoint's weights license before a
final run and record it in report/license_verification.md.

Usage:
    python -m src.train_vlm --config configs/qwen25vl_lora.yaml --fold 0 [--resume]
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.augment import AugmentConfig, Augmenter
from src.data.datasets import SplitSpec, load_split, long_line_sampler, vlm_records
from src.data.preprocess import style_tag, vlm_thumbnail
from src.data.stackmix import StackMixConfig, load_generator
from src.eval.eval_metrics import evaluate_set
from src.utils.ckpt_manager import CheckpointManager, cleanup_hf_cache
from src.utils.common import (config_hash, describe_runtime, get_device, get_logger,
                              load_config, parse_cli_overrides, set_seed)

log = get_logger("vlm")

# Verbatim from the organizers' starter (trainer.py / inference.py). Keeping it
# byte-identical between training and inference is essential.
OCR_PROMPT = (
    "Transcribe EXACTLY as seen. Preserve line breaks, punctuation, and spacing. "
    "Do not add explanations. Return verbatim transcription. No paraphrasing. No normalization."
)


def prompt_text(style: str = "") -> str:
    """Training/inference prompt; the optional style tag (model.style_prompt) is appended after the instruction."""
    return OCR_PROMPT + (" " + style if style else "")


def image_kwargs(cfg: Dict) -> Dict:
    """vlm_thumbnail options from cfg.model (shared by training and inference)."""
    m = cfg["model"]
    return {"vertical_pad_frac": m.get("vertical_pad_frac", 0.20),
            "target_height": m.get("image_height", 0),
            "max_width": m.get("image_max_width", 4096),
            "isolate_band": bool(m.get("isolate_band", False))}


def vlm_class(model_id: str):
    if "Qwen3-VL" in model_id:
        from transformers import Qwen3VLForConditionalGeneration as cls
    elif "InternVL" in model_id:   # InternVL3.5 *-HF checkpoints; needs transformers >= 4.52 (pylib 4.57.1)
        from transformers import InternVLForConditionalGeneration as cls
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration as cls
    return cls


def load_processor(cfg: Dict):
    from transformers import AutoProcessor
    m = cfg["model"]
    processor = AutoProcessor.from_pretrained(m["id"], trust_remote_code=True)
    ip = processor.image_processor
    for key in ("min_pixels", "max_pixels"):
        if m.get(key):
            setattr(ip, key, int(m[key]))
    if m.get("min_pixels") and m.get("max_pixels") and isinstance(getattr(ip, "size", None), dict):
        # newer processors read the pixel budget from size={shortest_edge, longest_edge}
        ip.size = {"shortest_edge": int(m["min_pixels"]), "longest_edge": int(m["max_pixels"])}
    for key, value in (m.get("image_processor_kwargs") or {}).items():
        # model-specific options, e.g. InternVL tiling: crop_to_patches / min_patches / max_patches
        setattr(ip, key, value)
    return processor


class VLMDataset(Dataset):
    def __init__(self, records: List[Dict], aug_cfg=None, stackmix_cfg=None,
                 bank_dir=None, train=True, img_kw=None, style=False):
        self.records = records
        self.style = style
        self.train = train
        self.img_kw = img_kw or {}
        self.augmenter = Augmenter(aug_cfg) if (train and aug_cfg and aug_cfg.enabled) else None
        self.stackmix = (
            load_generator(bank_dir, stackmix_cfg)
            if train and stackmix_cfg and stackmix_cfg.enabled and bank_dir else None
        )

    def __len__(self):
        return len(self.records)

    def set_augment_strength(self, s: float):
        if self.augmenter:
            self.augmenter.cfg.strength = float(s)

    def __getitem__(self, i):
        r = self.records[i]
        if self.stackmix is not None:
            got = self.stackmix.maybe_generate()
            if got is not None:
                img, text, _ = got
                if self.augmenter:
                    img = self.augmenter(img)
                return {"image": img, "text": text, "id": f"stackmix::{r['id']}",
                        "loss_weight": 1.0}

        if not os.path.exists(r["image"]):
            from PIL import Image
            img = Image.new("RGB", (512, 96), (255, 255, 255))
        else:
            img = vlm_thumbnail(r["image"], **self.img_kw)
        if self.augmenter:
            img = self.augmenter(img)
        return {"image": img, "text": r["text"], "id": r["id"],
                "loss_weight": r.get("loss_weight", 1.0),
                "style": style_tag(r["image"]) if (self.style and os.path.exists(r["image"])) else ""}


def build_model(cfg: Dict):
    """bf16 LoRA. NO quantization_config anywhere - see module docstring."""
    from peft import LoraConfig, get_peft_model

    model_id = cfg["model"]["id"]
    dtype = getattr(torch, cfg["model"].get("dtype", "float32"))
    log.info("loading %s in %s (no bitsandbytes, ROCm-safe)", model_id, dtype)

    model = vlm_class(model_id).from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
        device_map=None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if cfg["train"].get("gradient_checkpointing", True):
        # Non-reentrant: with the reentrant default, blocks whose inputs don't
        # require grad (frozen embeddings / vision input) give LoRA no gradients
        # ("None of the inputs have requires_grad=True" in earlier runs).
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    vr = cfg["model"].get("vision_lora_r")      # higher LoRA rank for the vision tower only (R17 feature-side probe)
    vis_re = r"visual\.blocks\.\d+\.(attn\.qkv|attn\.proj|mlp\.linear_fc1|mlp\.linear_fc2|mlp\.gate_proj|mlp\.up_proj|mlp\.down_proj)"
    lc = LoraConfig(
        rank_pattern={vis_re: int(vr)} if vr else {},
        alpha_pattern={vis_re: 2 * int(vr)} if vr else {},
        r=cfg["model"].get("lora_r", 32),
        lora_alpha=cfg["model"].get("lora_alpha", 64),
        lora_dropout=cfg["model"].get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["model"].get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
    )
    model = get_peft_model(model, lc)
    if cfg["model"].get("vision_full_ft"):
        # R19: the whole vision tower (patch embed, all blocks, mergers) is trained in fp32 (bf16 AdamW would drop small
        # updates); forward/backward run under bf16 autocast.  Its weights are saved as visual_full.pt next to the adapter.
        n = 0
        for name, p in model.named_parameters():
            if ".visual." in name and "lora_" not in name:
                p.data = p.data.float(); p.requires_grad_(True); n += p.numel()
        model._vision_full_ft = True
        log.info("vision tower fully trainable in fp32: %.1fM params", n / 1e6)
    model.print_trainable_parameters()

    return model, load_processor(cfg)


def make_collate(processor, train: bool = True):
    tok = processor.tokenizer

    def _find_subseq(hay: List[int], needle: List[int]) -> int:
        n = len(needle)
        if n == 0:
            return -1
        for j in range(len(hay) - n + 1):
            if hay[j:j + n] == needle:
                return j
        return -1

    def collate(batch: List[Dict]):
        images = [b["image"] for b in batch]
        texts = [b["text"] for b in batch]

        msgs = []
        for b, img, label in zip(batch, images, texts):
            content = [{"type": "text", "text": prompt_text(b.get("style", ""))}, {"type": "image", "image": img}]
            m = [{"role": "user", "content": content}]
            if train:
                m.append({"role": "assistant",
                          "content": [{"type": "text", "text": label}]})
            msgs.append(m)

        prompts = [
            processor.apply_chat_template(m, tokenize=False,
                                          add_generation_prompt=not train)
            for m in msgs
        ]
        enc = processor(text=prompts, images=images, padding=True,
                        truncation=True, return_tensors="pt")

        out = dict(enc)
        out["ids"] = [b["id"] for b in batch]
        out["texts"] = texts
        out["loss_weight"] = torch.tensor([b["loss_weight"] for b in batch],
                                          dtype=torch.float)

        if train:
            input_ids = enc["input_ids"]
            labels = input_ids.clone()
            # Assistant-only loss: mask everything before the answer span.
            for i, label in enumerate(texts):
                lab_ids = tok.encode(label, add_special_tokens=False)
                start = _find_subseq(input_ids[i].tolist(), lab_ids)
                if start != -1:
                    labels[i, :start] = -100
            if tok.pad_token_id is not None:
                labels[labels == tok.pad_token_id] = -100
            try:
                img_tok = tok.convert_tokens_to_ids(processor.image_token)
                labels[labels == img_tok] = -100
            except Exception:
                pass
            out["labels"] = labels
        return out

    return collate


@torch.no_grad()
def _autocast(model):
    """bf16 autocast when parts of the model hold fp32 master weights (vision_full_ft); no-op otherwise."""
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(getattr(model, "_vision_full_ft", False)))


def validate(model, processor, loader, device, cfg, out_csv, ref_csv):
    model.eval()
    gen = cfg.get("generate", {})
    rows, refs = [], []
    for batch in loader:
        ids_ = batch.pop("ids")
        texts = batch.pop("texts")
        batch.pop("loss_weight", None)
        batch.pop("labels", None)
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with _autocast(model):
          out = model.generate(
              **inputs,
              max_new_tokens=gen.get("max_new_tokens", 128),
              num_beams=gen.get("num_beams", 5),
              do_sample=False,
              length_penalty=gen.get("length_penalty", 1.0),
          )
        dec = processor.batch_decode(out, skip_special_tokens=True)
        for i, d, t in zip(ids_, dec, texts):
            for tag in ("assistant", "<|assistant|>", "user", "<|user|>"):
                if tag in d:
                    d = d.split(tag)[-1]
            rows.append({"ID": i, "Target": " ".join(str(d).split())})
            refs.append({"ID": i, "Target": t})

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(refs).to_csv(ref_csv, index=False)
    res = evaluate_set(ref_csv, out_csv, "VAL",
                       weight_factor=cfg["eval"].get("weight_factor", 0.5),
                       normalization=cfg["eval"].get("normalization", "raw"),
                       scheme=cfg["eval"].get("scheme", "length_pow"))
    model.train()
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default=None,
                    help="warm-start weights (no optimizer/step) if no checkpoint to resume")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, parse_cli_overrides(args.set))
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold
    set_seed(cfg.get("seed", 42))
    log.info("runtime: %s", describe_runtime())
    log.info("config hash: %s", config_hash(cfg))

    device = get_device()
    tcfg, dcfg = cfg["train"], cfg["data"]
    if os.environ.get("GPU_MEM_FRACTION"):   # hard cap on this process's GPU memory when the GPU is shared
        torch.cuda.set_per_process_memory_fraction(float(os.environ["GPU_MEM_FRACTION"]), 0)
        log.info("GPU memory capped at %.0f%% of the device", 100 * float(os.environ["GPU_MEM_FRACTION"]))

    train_df, val_df = load_split(
        SplitSpec(dcfg["folds_csv"], dcfg["fold"], dcfg["image_dir"],
                  dcfg.get("pseudo_csv"), dcfg.get("pseudo_max_ratio", 0.30),
                  dcfg.get("pseudo_loss_weight", 0.5),
                  use_all_folds=dcfg.get("use_all_folds", False)),
        seed=cfg.get("seed", 42),
    )
    # optional subset / re-weighting of the training and validation lines by ID list (R16: tall-crop experiments)
    def _ids(key):
        return set(pd.read_csv(dcfg[key], dtype={"ID": str})["ID"])
    if dcfg.get("train_ids_csv"):
        train_df = train_df[train_df["ID"].isin(_ids("train_ids_csv"))].reset_index(drop=True)
    if dcfg.get("val_ids_csv"):
        val_df = val_df[val_df["ID"].isin(_ids("val_ids_csv"))].reset_index(drop=True)
    if dcfg.get("boost_ids_csv") and int(dcfg.get("boost_factor", 1)) > 1:
        extra = train_df[train_df["ID"].isin(_ids("boost_ids_csv"))]
        train_df = pd.concat([train_df] + [extra] * (int(dcfg["boost_factor"]) - 1)).reset_index(drop=True)
    log.info("train lines %d, val lines %d (after optional ID filters/boost)", len(train_df), len(val_df))

    aug = AugmentConfig(**{k: v for k, v in cfg["augment"].items() if k != "_rng"})
    sm = StackMixConfig(**cfg["stackmix"]) if cfg.get("stackmix") else None
    bank = cfg.get("stackmix", {}).get("bank_dir")

    img_kw = image_kwargs(cfg)
    train_ds = VLMDataset(vlm_records(train_df, dcfg["image_dir"]), aug, sm, bank, True, img_kw, cfg["model"].get("style_prompt", False))
    val_ds = VLMDataset(vlm_records(val_df, dcfg["image_dir"]), None, None, None, False, img_kw, cfg["model"].get("style_prompt", False))

    model, processor = build_model(cfg)
    # CRITICAL: Qwen2.5-VL is decoder-only and MUST use left-padding.
    # Without this, batched inputs are right-padded, corrupting the causal mask
    # and producing garbage gradients. This caused the Qwen-7B LB regression.
    processor.tokenizer.padding_side = "left"
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"
    model.to(device)

    train_collate = make_collate(processor, train=True)
    val_collate = make_collate(processor, train=False)


    val_loader = DataLoader(val_ds, batch_size=tcfg.get("eval_batch_size", 4),
                            collate_fn=val_collate, shuffle=False,
                            num_workers=tcfg.get("num_workers", 4))

    if getattr(model, "_vision_full_ft", False):
        import re as _re
        vis = [(n, p) for n, p in model.named_parameters() if p.requires_grad and ".visual." in n and "lora_" not in n]
        nb = 1 + max(int(m.group(1)) for n, _ in vis for m in [_re.search(r"\.blocks\.(\d+)\.", n)] if m)
        vlr, dec = tcfg.get("vision_lr", 3e-5), tcfg.get("vision_layer_decay", 0.85)
        by_scale = {}
        for n, p in vis:
            m = _re.search(r"\.blocks\.(\d+)\.", n)
            depth = int(m.group(1)) if m else (-1 if ("patch_embed" in n or "pos_embed" in n) else nb - 1)
            by_scale.setdefault(dec ** (nb - 1 - depth), []).append(p)
        groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and not (".visual." in n and "lora_" not in n)],
                   "lr": tcfg["lr"], "weight_decay": tcfg.get("weight_decay", 0.0)}]
        groups += [{"params": ps, "lr": vlr * sc, "weight_decay": tcfg.get("vision_weight_decay", 0.05)} for sc, ps in by_scale.items()]
        log.info("optimizer: LoRA lr %.1e; vision tower %d blocks, lr %.1e at the top block x %.2f per block down", tcfg["lr"], nb, vlr, dec)
        optim = torch.optim.AdamW(groups)
    else:
        optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    accum = tcfg.get("grad_accum", 2)
    approx_steps = max(1, (len(train_ds) // max(1, tcfg["batch_size"])) // accum)
    total_steps = approx_steps * tcfg["epochs"]
    warmup = int(total_steps * tcfg.get("warmup_ratio", 0.03))

    def lr_lambda(s):
        if s < warmup:
            return s / max(1, warmup)
        p = (s - warmup) / max(1, total_steps - warmup)
        return max(0.02, 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    ckpt = CheckpointManager(cfg["output"]["checkpoint_dir"],
                             keep_last=cfg["output"].get("keep_last", 1),
                             greater_is_better=True, is_peft=True)
    state = ckpt.resume(model, optim, sched) if args.resume else {"epoch": 0, "global_step": 0}
    if args.init_from and not state.get("resumed"):
        ckpt.init_from(model, args.init_from)
    start_epoch, global_step = state["epoch"], state["global_step"]
    # A step-save is mid-epoch: skip the micro-batches already trained (sampler order is seeded per epoch).
    resume_it = int((state.get("extra") or {}).get("resume_it", 0))
    if resume_it:
        log.info("mid-epoch resume: epoch %d, skipping first %d micro-batches", start_epoch + 1, resume_it)

    def validate_and_save(epochs_done):
        res = validate(model, processor, val_loader, device, cfg,
                       os.path.join(cfg["output"]["dir"], f"val_pred_fold{dcfg['fold']}.csv"),
                       os.path.join(cfg["output"]["dir"], f"val_ref_fold{dcfg['fold']}.csv"))
        log.info("epoch %d  VAL final_07_03=%.4f (0.5/0.5=%.4f)",
                 epochs_done, res["final_07_03"], res["final_05_05"])
        ckpt.save(model, optim, sched, epoch=epochs_done, global_step=global_step,
                  metric=res[cfg["eval"].get("primary", "final_05_05")], extra={"val": res})

    if (state.get("extra") or {}).get("pending_val"):
        log.info("resumed from an unvalidated epoch-%d checkpoint: validating it first", start_epoch)
        validate_and_save(start_epoch)

    model.train()
    t0 = time.time()
    cur_start = tcfg.get("long_line_curriculum_start_epoch", 1)

    for epoch in range(start_epoch, tcfg["epochs"]):
        power = 0.0 if epoch < cur_start else tcfg.get("long_line_curriculum_power", 1.0) * (
            (epoch - cur_start + 1) / max(1, tcfg["epochs"] - cur_start))
        train_ds.set_augment_strength(
            min(1.0, tcfg.get("aug_strength_start", 0.5) + epoch * tcfg.get("aug_strength_step", 0.15))
        )
        loader = DataLoader(
            train_ds, batch_size=tcfg["batch_size"], collate_fn=train_collate,
            num_workers=tcfg.get("num_workers", 4), pin_memory=True, drop_last=True,
            sampler=long_line_sampler(train_df, cfg["eval"].get("weight_factor", 0.5),
                                      power=power, seed=cfg.get("seed", 42) + epoch),
        )
        log.info("epoch %d/%d  long_line_power=%.2f", epoch + 1, tcfg["epochs"], power)

        optim.zero_grad(set_to_none=True)
        running, nb = 0.0, 0
        for it, batch in enumerate(loader):
            if epoch == start_epoch and it < resume_it:
                continue
            batch.pop("ids"); batch.pop("texts")
            lw = batch.pop("loss_weight").to(device)
            labels = batch.pop("labels").to(device)
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            with _autocast(model):
                out = model(**inputs)
            logits = out.logits[:, :-1, :]
            tgt = labels[:, 1:]
            lf = torch.nn.CrossEntropyLoss(reduction="none",
                                           label_smoothing=tcfg.get("label_smoothing", 0.0))
            per_tok = lf(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1)).view(tgt.shape)
            mask = (tgt != -100).float()
            per_sample = (per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)
            loss = (per_sample * lw).sum() / lw.sum().clamp(min=1e-6)

            (loss / accum).backward()
            running += float(loss.detach()); nb += 1

            if (it + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    tcfg.get("max_grad_norm", 1.0))
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % tcfg.get("log_every", 25) == 0:
                    log.info("step %d  loss %.4f  lr %.2e  %.1f min",
                             global_step, running / max(nb, 1), sched.get_last_lr()[0],
                             (time.time() - t0) / 60)
                    running, nb = 0.0, 0
                if global_step % tcfg.get("save_every_steps", 300) == 0:
                    ckpt.save(model, optim, sched, epoch=epoch, global_step=global_step,
                              extra={"resume_it": it + 1})

        # Save first (flagged pending_val), so a crash inside validate keeps the epoch
        # and the retry scores it before training on. eval_last_only skips the
        # ~1-2 h beam-search validation on non-final epochs (the last epoch was best in R8).
        if tcfg.get("skip_val", False):
            # All-data models: fold 0 is in training, so validation is meaningless; the
            # final epoch becomes best_checkpoint (metric 0.0 only marks it as scored).
            if epoch + 1 == tcfg["epochs"]:
                ckpt.save(model, optim, sched, epoch=epoch + 1, global_step=global_step,
                          metric=0.0, extra={"val": "skipped (use_all_folds)"})
            else:
                ckpt.save(model, optim, sched, epoch=epoch + 1, global_step=global_step)
        elif epoch + 1 == tcfg["epochs"] or not tcfg.get("eval_last_only", False):
            ckpt.save(model, optim, sched, epoch=epoch + 1, global_step=global_step,
                      extra={"pending_val": True})
            validate_and_save(epoch + 1)
        else:
            ckpt.save(model, optim, sched, epoch=epoch + 1, global_step=global_step)

    if cfg["output"].get("cleanup_hf_cache", False):
        cleanup_hf_cache([cfg["model"]["id"]], dry_run=False)
    log.info("done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
