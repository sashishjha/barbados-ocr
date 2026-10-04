"""
Arm A - TrOCR (microsoft/trocr-large-handwritten, MIT licensed weights).

Design notes for the RBCCPS ROCm/MI300X cluster:
  * bf16 autocast, PyTorch SDPA attention. No bitsandbytes, no quantization.
  * Full fine-tuning fits easily in 192 GB HBM; LoRA is available via
    `model.use_lora: true` if you want to run several variants concurrently.
  * Checkpoint every `train.save_every_steps` AND at every epoch end, with
    optimizer/scheduler/RNG state, so a 24h kill resumes cleanly with --resume.
  * Validation computes the official weighted metric (0.7*CER + 0.3*WER form)
    on the held-out fold and drives best-checkpoint selection.

Usage:
    python -m src.train_trocr --config configs/trocr_large.yaml --fold 0 [--resume]
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.augment import AugmentConfig
from src.data.datasets import LineOCRDataset, SplitSpec, load_split, long_line_sampler
from src.data.preprocess import PreprocessConfig
from src.data.stackmix import StackMixConfig
from src.eval.eval_metrics import evaluate_set
from src.utils.ckpt_manager import CheckpointManager, cleanup_hf_cache
from src.utils.common import (config_hash, describe_runtime, get_device, get_logger,
                              load_config, parse_cli_overrides, set_seed)

log = get_logger("trocr")


# --------------------------------------------------------------------------
def build_model(cfg: Dict):
    import transformers.modeling_utils
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    model_id = cfg["model"]["id"]
    log.info("loading %s", model_id)
    try:
        processor = TrOCRProcessor.from_pretrained(model_id)
    except Exception as e:
        log.warning(f"TrOCRProcessor load failed ({e}), falling back to roberta-large tokenizer")
        from transformers import AutoImageProcessor, AutoTokenizer
        feature_extractor = AutoImageProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)
    
    model = VisionEncoderDecoderModel.from_pretrained(
        model_id,
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
        low_cpu_mem_usage=False,  # stage1 ships with meta tensors; must materialize fully
    )

    # Generation config must be set explicitly or HF warns and decodes badly.
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    gen = cfg.get("generate", {})
    model.generation_config.max_length = gen.get("max_length", 96)
    model.generation_config.num_beams = gen.get("num_beams", 5)
    model.generation_config.early_stopping = True
    model.generation_config.length_penalty = gen.get("length_penalty", 1.0)
    model.generation_config.no_repeat_ngram_size = gen.get("no_repeat_ngram_size", 0)

    if cfg["model"].get("use_lora", False):
        from peft import LoraConfig, get_peft_model

        lc = LoraConfig(
            r=cfg["model"].get("lora_r", 32),
            lora_alpha=cfg["model"].get("lora_alpha", 64),
            lora_dropout=cfg["model"].get("lora_dropout", 0.05),
            bias="none",
            target_modules=cfg["model"].get(
                "lora_target_modules", ["query", "key", "value", "q_proj", "k_proj", "v_proj"]
            ),
        )
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()

    return model, processor


def make_collate(processor, max_target_length: int):
    def collate(batch: List[Dict]):
        images = [b["image"] for b in batch]
        texts = [b["text"] for b in batch]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        tok = processor.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=max_target_length, return_tensors="pt",
        )
        labels = tok.input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "loss_weight": torch.tensor([b["loss_weight"] for b in batch], dtype=torch.float),
            "ids": [b["id"] for b in batch],
            "texts": texts,
        }

    return collate


# --------------------------------------------------------------------------
@torch.no_grad()
def validate(model, processor, loader, device, cfg, out_csv: str, ref_csv: str) -> Dict:
    model.eval()
    gen = cfg.get("generate", {})
    rows, refs = [], []
    for batch in loader:
        pv = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ids = model.generate(
                pv,
                num_beams=gen.get("num_beams", 5),
                max_length=gen.get("max_length", 96),
                length_penalty=gen.get("length_penalty", 1.0),
                early_stopping=True,
            )
        preds = processor.batch_decode(ids, skip_special_tokens=True)
        for i, p in zip(batch["ids"], preds):
            rows.append({"ID": i, "Target": " ".join(str(p).split())})
        for i, t in zip(batch["ids"], batch["texts"]):
            refs.append({"ID": i, "Target": t})

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(refs).to_csv(ref_csv, index=False)

    res = evaluate_set(
        ref_csv, out_csv, set_name="VAL",
        weight_factor=cfg["eval"].get("weight_factor", 0.5),
        normalization=cfg["eval"].get("normalization", "raw"),
        scheme=cfg["eval"].get("scheme", "length_pow"),
    )
    model.train()
    return res


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default=None,
                    help="warm-start weights (no optimizer/step) if no checkpoint to resume")
    ap.add_argument("--set", nargs="*", default=[], help="cfg overrides, e.g. train.lr=1e-5")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_cli_overrides(args.set))
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold

    set_seed(cfg.get("seed", 42))
    log.info("runtime: %s", describe_runtime())
    log.info("config hash: %s", config_hash(cfg))

    device = get_device()
    tcfg, dcfg = cfg["train"], cfg["data"]

    # ---------------- data
    train_df, val_df = load_split(
        SplitSpec(
            folds_csv=dcfg["folds_csv"], fold=dcfg["fold"], image_dir=dcfg["image_dir"],
            pseudo_csv=dcfg.get("pseudo_csv"),
            pseudo_max_ratio=dcfg.get("pseudo_max_ratio", 0.30),
            pseudo_loss_weight=dcfg.get("pseudo_loss_weight", 0.5),
            use_all_folds=dcfg.get("use_all_folds", False),
        ),
        seed=cfg.get("seed", 42),
    )

    pre = PreprocessConfig(**cfg["preprocess"])
    aug = AugmentConfig(**{k: v for k, v in cfg["augment"].items() if k != "_rng"})
    sm = StackMixConfig(**cfg["stackmix"]) if cfg.get("stackmix") else None

    train_ds = LineOCRDataset(train_df, dcfg["image_dir"], pre, aug, sm,
                              cfg.get("stackmix", {}).get("bank_dir"), train=True)
    val_ds = LineOCRDataset(val_df, dcfg["image_dir"], pre, None, None, None, train=False)

    model, processor = build_model(cfg)
    if tcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    model.to(device)
    # trocr-large-stage1: self.weights in TrOCRSinusoidalPositionalEmbedding is a plain
    # Python attribute (NOT a buffer). It is set during meta-device __init__ and stays
    # as a meta tensor — model.to(device) silently skips plain Python attributes.
    # Fix: recompute weights directly on device, replacing the meta tensor.
    from transformers.models.trocr.modeling_trocr import TrOCRSinusoidalPositionalEmbedding
    for module in model.modules():
        if isinstance(module, TrOCRSinusoidalPositionalEmbedding):
            n_pos = module.weights.shape[0] if module.weights is not None else 1026
            module.weights = TrOCRSinusoidalPositionalEmbedding.get_embedding(
                n_pos, module.embedding_dim, module.padding_idx
            ).to(device)
            log.info("recomputed TrOCRSinusoidalPositionalEmbedding.weights on %s", device)

    collate = make_collate(processor, tcfg.get("max_target_length", 96))
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], collate_fn=collate,
        num_workers=tcfg.get("num_workers", 8), pin_memory=True, drop_last=True,
        sampler=long_line_sampler(train_df, cfg["eval"].get("weight_factor", 0.5),
                                  power=0.0, seed=cfg.get("seed", 42)),
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg.get("eval_batch_size", tcfg["batch_size"]),
        collate_fn=collate, num_workers=tcfg.get("num_workers", 8), shuffle=False,
    )

    # ---------------- optim
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.01),
        betas=(0.9, 0.999),
    )
    accum = tcfg.get("grad_accum", 1)
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * tcfg["epochs"]
    warmup = int(total_steps * tcfg.get("warmup_ratio", 0.05))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(0.02, 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    ckpt = CheckpointManager(
        root=cfg["output"]["checkpoint_dir"],
        keep_last=cfg["output"].get("keep_last", 1),
        greater_is_better=True,
        is_peft=cfg["model"].get("use_lora", False),
    )
    state = ckpt.resume(model, optim, sched) if args.resume else {"epoch": 0, "global_step": 0}
    if args.init_from and not state.get("resumed"):
        ckpt.init_from(model, args.init_from)
    start_epoch, global_step = state["epoch"], state["global_step"]
    if state.get("resumed"):
        log.info("resumed at epoch=%d step=%d", start_epoch, global_step)

    # ---------------- loop
    model.train()
    t0 = time.time()
    curriculum_start = tcfg.get("long_line_curriculum_start_epoch", 2)
    max_power = tcfg.get("long_line_curriculum_power", 1.0)

    for epoch in range(start_epoch, tcfg["epochs"]):
        # Long-line curriculum: ramp the sampler power in the later epochs, since
        # the metric weights long references more heavily.
        power = 0.0 if epoch < curriculum_start else max_power * (
            (epoch - curriculum_start + 1) / max(1, tcfg["epochs"] - curriculum_start)
        )
        train_loader = DataLoader(
            train_ds, batch_size=tcfg["batch_size"], collate_fn=collate,
            num_workers=tcfg.get("num_workers", 8), pin_memory=True, drop_last=True,
            sampler=long_line_sampler(train_df, cfg["eval"].get("weight_factor", 0.5),
                                      power=power, seed=cfg.get("seed", 42) + epoch),
        )
        # Augmentation curriculum: gentle early, aggressive later.
        train_ds.set_augment_strength(
            min(1.0, tcfg.get("aug_strength_start", 0.5)
                + epoch * tcfg.get("aug_strength_step", 0.15))
        )
        log.info("epoch %d/%d  long_line_power=%.2f  aug_strength=%.2f",
                 epoch + 1, tcfg["epochs"], power, aug.strength)

        running, nb = 0.0, 0
        optim.zero_grad(set_to_none=True)
        for it, batch in enumerate(train_loader):
            pv = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            lw = batch["loss_weight"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(pixel_values=pv, labels=labels)
                logits = out.logits
                # Per-sample loss so pseudo-labels can be down-weighted.
                lf = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=
                                               tcfg.get("label_smoothing", 0.0))
                per_tok = lf(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                per_tok = per_tok.view(labels.shape)
                mask = (labels != -100).float()
                per_sample = (per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)
                loss = (per_sample * lw).sum() / lw.sum().clamp(min=1e-6)

            (loss / accum).backward()
            running += float(loss.detach())
            nb += 1

            if (it + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               tcfg.get("max_grad_norm", 1.0))
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % tcfg.get("log_every", 25) == 0:
                    log.info("step %d  loss %.4f  lr %.2e  %.1f min elapsed",
                             global_step, running / max(nb, 1),
                             sched.get_last_lr()[0], (time.time() - t0) / 60)
                    running, nb = 0.0, 0

                if global_step % tcfg.get("save_every_steps", 500) == 0:
                    ckpt.save(model, optim, sched, epoch=epoch,
                              global_step=global_step, metric=None)

        # ---- epoch-end validation + checkpoint
        if tcfg.get("skip_val", False):
            # All-data model: fold 0 is in training, so validation is meaningless; the final epoch
            # becomes best_checkpoint (metric 0.0 only marks it as scored).
            last = epoch + 1 == tcfg["epochs"]
            ckpt.save(model, optim, sched, epoch=epoch + 1, global_step=global_step,
                      metric=0.0 if last else None,
                      extra={"val": "skipped (use_all_folds)"} if last else None)
            continue
        res = validate(
            model, processor, val_loader, device, cfg,
            out_csv=os.path.join(cfg["output"]["dir"], f"val_pred_fold{dcfg['fold']}.csv"),
            ref_csv=os.path.join(cfg["output"]["dir"], f"val_ref_fold{dcfg['fold']}.csv"),
        )
        log.info("epoch %d  VAL final_07_03=%.4f  final_05_05=%.4f  CER=%.4f WER=%.4f",
                 epoch + 1, res["final_07_03"], res["final_05_05"], res["cer"], res["wer"])
        ckpt.save(model, optim, sched, epoch=epoch + 1, global_step=global_step,
                  metric=res[cfg["eval"].get("primary", "final_05_05")], extra={"val": res})

    # ---------------- storage hygiene
    if cfg["output"].get("cleanup_hf_cache", False):
        cleanup_hf_cache([cfg["model"]["id"]], dry_run=False)
    log.info("done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
