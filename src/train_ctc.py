"""
Arm C trainer - CRNN/CTC from scratch on the provided data.

Cheap (a few hours on one MI300X), architecturally diverse from the two
transformer arms, and a strong voter in ROVER-style character fusion.

Usage:
    python -m src.train_ctc --config configs/crnn_ctc.yaml --fold 0 [--resume]
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.augment import AugmentConfig
from src.data.datasets import LineOCRDataset, SplitSpec, load_split, long_line_sampler
from src.data.preprocess import PreprocessConfig
from src.data.stackmix import StackMixConfig
from src.eval.eval_metrics import evaluate_set
from src.models.crnn_ctc import CRNN, CharTokenizer
from src.utils.ckpt_manager import CheckpointManager
from src.utils.common import (config_hash, describe_runtime, get_device, get_logger,
                              load_config, parse_cli_overrides, set_seed)

log = get_logger("ctc")


def make_collate(tokenizer: CharTokenizer):
    def collate(batch):
        imgs = []
        for b in batch:
            a = np.asarray(b["image"].convert("L"), dtype=np.float32) / 255.0
            imgs.append(torch.from_numpy(a)[None])  # (1, H, W)
        x = torch.stack(imgs)
        targets = [torch.tensor(tokenizer.encode(b["text"]), dtype=torch.long) for b in batch]
        return {
            "x": x,
            "targets": torch.cat(targets) if targets else torch.zeros(0, dtype=torch.long),
            "target_lengths": torch.tensor([len(t) for t in targets], dtype=torch.long),
            "loss_weight": torch.tensor([b["loss_weight"] for b in batch], dtype=torch.float),
            "ids": [b["id"] for b in batch],
            "texts": [b["text"] for b in batch],
        }

    return collate


@torch.no_grad()
def validate(model, tokenizer, loader, device, cfg, out_csv, ref_csv):
    model.eval()
    rows, refs = [], []
    for batch in loader:
        x = batch["x"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lp = model(x)                     # (T, B, C)
        ids = lp.argmax(-1).permute(1, 0).cpu().tolist()
        for i, seq, t in zip(batch["ids"], ids, batch["texts"]):
            rows.append({"ID": i, "Target": tokenizer.decode_greedy(seq)})
            refs.append({"ID": i, "Target": t})
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(refs).to_csv(ref_csv, index=False)
    res = evaluate_set(ref_csv, out_csv, "VAL",
                       weight_factor=cfg["eval"].get("weight_factor", 0.5),
                       normalization=cfg["eval"].get("normalization", "raw"),
                       scheme=cfg["eval"].get("scheme", "length_pow"))
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
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

    train_df, val_df = load_split(
        SplitSpec(dcfg["folds_csv"], dcfg["fold"], dcfg["image_dir"],
                  dcfg.get("pseudo_csv"), dcfg.get("pseudo_max_ratio", 0.30),
                  dcfg.get("pseudo_loss_weight", 0.5)),
        seed=cfg.get("seed", 42))

    # Charset from TRAIN transcriptions only (never test, never external).
    tok_path = os.path.join(cfg["output"]["dir"], "charset.json")
    if os.path.exists(tok_path):
        tokenizer = CharTokenizer.load(tok_path)
    else:
        tokenizer = CharTokenizer.from_texts(train_df["Target"].tolist())
        tokenizer.save(tok_path)
    log.info("charset size: %d (from training transcriptions only)", len(tokenizer))

    pre = PreprocessConfig(**cfg["preprocess"])
    aug = AugmentConfig(**{k: v for k, v in cfg["augment"].items() if k != "_rng"})
    sm = StackMixConfig(**cfg["stackmix"]) if cfg.get("stackmix") else None

    train_ds = LineOCRDataset(train_df, dcfg["image_dir"], pre, aug, sm,
                              cfg.get("stackmix", {}).get("bank_dir"), train=True)
    val_ds = LineOCRDataset(val_df, dcfg["image_dir"], pre, None, None, None, train=False)

    collate = make_collate(tokenizer)
    val_loader = DataLoader(val_ds, batch_size=tcfg.get("eval_batch_size", 32),
                            collate_fn=collate, num_workers=tcfg.get("num_workers", 8))

    model = CRNN(n_classes=len(tokenizer),
                 hidden=cfg["model"].get("hidden", 384),
                 layers=cfg["model"].get("layers", 3),
                 dropout=cfg["model"].get("dropout", 0.15)).to(device)

    ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    optim = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                              weight_decay=tcfg.get("weight_decay", 0.01))
    steps = max(1, len(train_ds) // max(1, tcfg["batch_size"])) * tcfg["epochs"]
    warm = int(steps * tcfg.get("warmup_ratio", 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: s / max(1, warm) if s < warm
        else max(0.02, 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm)))))
    )

    ckpt = CheckpointManager(cfg["output"]["checkpoint_dir"],
                             keep_last=cfg["output"].get("keep_last", 1),
                             greater_is_better=True, is_peft=False)
    state = ckpt.resume(model, optim, sched) if args.resume else {"epoch": 0, "global_step": 0}
    start_epoch, global_step = state["epoch"], state["global_step"]

    model.train()
    t0 = time.time()
    cur_start = tcfg.get("long_line_curriculum_start_epoch", 5)

    for epoch in range(start_epoch, tcfg["epochs"]):
        power = 0.0 if epoch < cur_start else tcfg.get("long_line_curriculum_power", 1.0) * (
            (epoch - cur_start + 1) / max(1, tcfg["epochs"] - cur_start))
        train_ds.set_augment_strength(
            min(1.0, tcfg.get("aug_strength_start", 0.6) + epoch * tcfg.get("aug_strength_step", 0.05)))
        loader = DataLoader(
            train_ds, batch_size=tcfg["batch_size"], collate_fn=collate,
            num_workers=tcfg.get("num_workers", 8), pin_memory=True, drop_last=True,
            sampler=long_line_sampler(train_df, cfg["eval"].get("weight_factor", 0.5),
                                      power=power, seed=cfg.get("seed", 42) + epoch))

        running, nb = 0.0, 0
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            tgt = batch["targets"].to(device)
            tl = batch["target_lengths"].to(device)
            lw = batch["loss_weight"].to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lp = model(x)
            lp = lp.float()
            il = torch.full((lp.size(1),), lp.size(0), dtype=torch.long, device=device)
            per_sample = ctc(lp, tgt, il, tl)
            loss = (per_sample * lw).sum() / lw.sum().clamp(min=1e-6)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("max_grad_norm", 5.0))
            optim.step(); sched.step()
            global_step += 1
            running += float(loss.detach()); nb += 1

            if global_step % tcfg.get("log_every", 50) == 0:
                log.info("step %d  loss %.4f  lr %.2e  %.1f min", global_step,
                         running / max(nb, 1), sched.get_last_lr()[0], (time.time() - t0) / 60)
                running, nb = 0.0, 0
            if global_step % tcfg.get("save_every_steps", 1000) == 0:
                ckpt.save(model, optim, sched, epoch=epoch, global_step=global_step)

        res = validate(model, tokenizer, val_loader, device, cfg,
                       os.path.join(cfg["output"]["dir"], f"val_pred_fold{dcfg['fold']}.csv"),
                       os.path.join(cfg["output"]["dir"], f"val_ref_fold{dcfg['fold']}.csv"))
        log.info("epoch %d  VAL final_07_03=%.4f", epoch + 1, res["final_07_03"])
        ckpt.save(model, optim, sched, epoch=epoch + 1, global_step=global_step,
                  metric=res["final_07_03"], extra={"val": res})

    log.info("done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
