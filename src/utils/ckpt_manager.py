"""
Checkpoint save/resume that survives a 24h debug-queue kill.

Cluster rules honoured here:
  * every checkpoint directory name contains the literal word "checkpoint"
    so the cluster's cleanup sweep does not delete it;
  * we keep at most `keep_last` + best, to stay inside the 200 GB quota;
  * we store LoRA adapters (tens of MB), never merged full weights, until the
    final inference merge step.

Saved state: model (or adapter), optimizer, LR scheduler, epoch, global step,
best metric, and RNG states for python/numpy/torch/torch.cuda.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
from typing import Dict, Optional

import torch

from .common import get_logger, restore_rng_state, rng_state

log = get_logger("checkpoint")


def _assert_checkpoint_name(d: str) -> None:
    if "checkpoint" not in os.path.basename(os.path.normpath(d)).lower():
        raise ValueError(
            f"Checkpoint dir '{d}' must contain the word 'checkpoint' or the "
            "cluster cleanup sweep will delete it."
        )


def load_visual_full(model, path: str) -> int:
    """Copy a saved fully fine-tuned vision tower (visual_full.pt) into a PEFT-wrapped model, keeping each param's dtype."""
    vf = os.path.join(path, "visual_full.pt")
    if not os.path.exists(vf):
        return 0
    sd = torch.load(vf, map_location="cpu")
    named = dict(model.named_parameters())
    missing = [k for k in sd if k not in named]
    if missing:
        raise RuntimeError(f"visual_full.pt: {len(missing)} tensors do not match the model, e.g. {missing[:2]}")
    with torch.no_grad():
        for k, t in sd.items():
            named[k].data.copy_(t.to(named[k].dtype))
    log.info("loaded fully fine-tuned vision tower: %d tensors from %s", len(sd), vf)
    return len(sd)


class CheckpointManager:
    def __init__(
        self,
        root: str,
        keep_last: int = 1,
        greater_is_better: bool = True,
        is_peft: bool = False,
    ):
        _assert_checkpoint_name(root)
        self.root = os.path.abspath(root)
        self.keep_last = keep_last
        self.greater_is_better = greater_is_better
        self.is_peft = is_peft
        os.makedirs(self.root, exist_ok=True)
        self.best_dir = os.path.join(self.root, "best_checkpoint")

    # -------------------------------------------------- save
    def save(
        self,
        model,
        optimizer=None,
        scheduler=None,
        *,
        epoch: int,
        global_step: int,
        metric: Optional[float] = None,
        extra: Optional[Dict] = None,
        scaler=None,
    ) -> str:
        # Write into a temp dir and rename it into place at the end, so a crash (OOM,
        # SIGTERM, node fault) mid-save can never leave a half-written "latest" checkpoint.
        step_dir = os.path.join(self.root, f"checkpoint_step{global_step:08d}")
        final_dir = step_dir
        step_dir = os.path.join(self.root, f"partial_checkpoint_step{global_step:08d}")
        shutil.rmtree(step_dir, ignore_errors=True)
        os.makedirs(step_dir, exist_ok=True)

        if self.is_peft:
            # LoRA adapter only: small, and the base model is re-downloaded
            # (or read from the shared HF cache) at load time.
            model.save_pretrained(step_dir)
            if getattr(model, "_vision_full_ft", False):   # fully fine-tuned vision tower (fp32 master weights)
                torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if ".visual." in n and "lora_" not in n},
                           os.path.join(step_dir, "visual_full.pt"))
        else:
            torch.save(model.state_dict(), os.path.join(step_dir, "model.pt"))

        trainer_state = {
            "epoch": epoch,
            "global_step": global_step,
            "metric": metric,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": rng_state(),
            "extra": extra or {},
        }
        torch.save(trainer_state, os.path.join(step_dir, "trainer_state.pt"))

        with open(os.path.join(step_dir, "meta.json"), "w") as f:
            json.dump(
                {"epoch": epoch, "global_step": global_step, "metric": metric,
                 "is_peft": self.is_peft, **(extra or {})},
                f, indent=2, default=str,
            )

        old_dir = final_dir + ".old"
        if os.path.isdir(final_dir):           # same step saved twice (pre-validate save, then epoch-end)
            shutil.rmtree(old_dir, ignore_errors=True)
            os.replace(final_dir, old_dir)
        os.replace(step_dir, final_dir)
        shutil.rmtree(old_dir, ignore_errors=True)
        step_dir = final_dir

        log.info("saved checkpoint %s (step=%d metric=%s)", step_dir, global_step, metric)
        self._update_best(step_dir, metric)
        self._prune()
        return step_dir

    def _update_best(self, step_dir: str, metric: Optional[float]) -> None:
        if metric is None:
            return
        prev = self._best_metric()
        better = (
            prev is None
            or (self.greater_is_better and metric > prev)
            or (not self.greater_is_better and metric < prev)
        )
        if better:
            # Weights + meta only: inference and --init-from never read the optimizer state
            # (trainer_state.pt = ~1.3 GB per LoRA arm, ~4 GB for TrOCR); only the rolling
            # checkpoints need it, for resume. Copy aside, then swap, so a crash mid-copy
            # can never leave us without a best checkpoint.
            partial = os.path.join(self.root, "partial_best_checkpoint")
            old = self.best_dir + ".old"
            shutil.rmtree(partial, ignore_errors=True)
            shutil.copytree(step_dir, partial, ignore=shutil.ignore_patterns("trainer_state.pt"))
            shutil.rmtree(old, ignore_errors=True)
            if os.path.isdir(self.best_dir):
                os.replace(self.best_dir, old)
            os.replace(partial, self.best_dir)
            shutil.rmtree(old, ignore_errors=True)
            log.info("new best checkpoint: metric=%.6f -> %s", metric, self.best_dir)

    def _best_metric(self) -> Optional[float]:
        meta = os.path.join(self.best_dir, "meta.json")
        if not os.path.exists(meta):
            return None
        with open(meta) as f:
            return json.load(f).get("metric")

    def _prune(self) -> None:
        dirs = sorted(glob.glob(os.path.join(self.root, "checkpoint_step*")))
        for d in dirs[: max(0, len(dirs) - self.keep_last)]:
            shutil.rmtree(d, ignore_errors=True)
            log.info("pruned old checkpoint %s (200 GB quota discipline)", d)

    # -------------------------------------------------- load
    def latest(self) -> Optional[str]:
        dirs = sorted(glob.glob(os.path.join(self.root, "checkpoint_step*")))
        return dirs[-1] if dirs else None

    def resume(self, model, optimizer=None, scheduler=None, scaler=None,
               path: Optional[str] = None) -> Dict:
        path = path or self.latest()
        if path is None:
            log.info("no checkpoint found in %s; starting fresh", self.root)
            return {"epoch": 0, "global_step": 0, "metric": None, "resumed": False}

        log.info("resuming from %s", path)
        if self.is_peft:
            from peft import PeftModel  # noqa: F401  (adapter loaded by caller)
            model.load_adapter(path, adapter_name="default", is_trainable=True)
            load_visual_full(model, path)
        else:
            sd = torch.load(os.path.join(path, "model.pt"), map_location="cpu")
            model.load_state_dict(sd)

        st = torch.load(os.path.join(path, "trainer_state.pt"), map_location="cpu",
                        weights_only=False)
        if optimizer is not None and st.get("optimizer"):
            optimizer.load_state_dict(st["optimizer"])
        if scheduler is not None and st.get("scheduler"):
            scheduler.load_state_dict(st["scheduler"])
        if scaler is not None and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        restore_rng_state(st.get("rng", {}))

        return {
            "epoch": st.get("epoch", 0),
            "global_step": st.get("global_step", 0),
            "metric": st.get("metric"),
            "extra": st.get("extra", {}),
            "resumed": True,
        }

    def init_from(self, model, path: str) -> None:
        """Warm start: load WEIGHTS ONLY from `path` (e.g. a golden BEST_* dir).

        Optimizer, scheduler, epoch and step stay fresh, so a new round trains a
        full schedule on top of the previous best. Only used when this run has
        no checkpoint_step* of its own; a requeued job resumes its own progress.
        """
        if not os.path.isdir(path):
            raise FileNotFoundError(f"init_from checkpoint not found: {path}")
        log.info("warm start: loading weights only from %s", path)
        if self.is_peft:
            # load_adapter silently skips keys that don't match, so load explicitly and check.
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            raw = load_file(os.path.join(path, "adapter_model.safetensors"))
            # adapters saved under transformers 4.57 (pylib) use 'model.visual...' names, 4.49 ones use 'visual...':
            # try the keys as saved first, then the converted ones, and keep whichever matches the model.
            for sd in (raw, _qwen_vl_keys_v5_to_v4(raw)):
                res = set_peft_model_state_dict(model, sd, adapter_name="default")
                bad = [k for k in res.unexpected_keys if "lora_" in k]
                if not bad:
                    break
            if bad:
                raise RuntimeError(f"{len(bad)} adapter weights did not match the model, e.g. {bad[:3]}")
            log.info("warm start: loaded %d adapter tensors", len(sd))
        else:
            sd = torch.load(os.path.join(path, "model.pt"), map_location="cpu")
            model.load_state_dict(_vit_keys_v5_to_v4(sd))


# transformers 5.x renamed the ViT encoder layers; the golden TrOCR checkpoints
# (R2-R5) were saved under 5.17 but the project is pinned to 4.49.
_VIT_V5_TO_V4 = [
    (r"^encoder\.layers\.(\d+)\.attention\.q_proj\.", r"encoder.encoder.layer.\1.attention.attention.query."),
    (r"^encoder\.layers\.(\d+)\.attention\.k_proj\.", r"encoder.encoder.layer.\1.attention.attention.key."),
    (r"^encoder\.layers\.(\d+)\.attention\.v_proj\.", r"encoder.encoder.layer.\1.attention.attention.value."),
    (r"^encoder\.layers\.(\d+)\.attention\.o_proj\.", r"encoder.encoder.layer.\1.attention.output.dense."),
    (r"^encoder\.layers\.(\d+)\.mlp\.fc1\.", r"encoder.encoder.layer.\1.intermediate.dense."),
    (r"^encoder\.layers\.(\d+)\.mlp\.fc2\.", r"encoder.encoder.layer.\1.output.dense."),
    (r"^encoder\.layers\.(\d+)\.", r"encoder.encoder.layer.\1."),  # layernorm_before/after
]


def _vit_keys_v5_to_v4(sd: Dict) -> Dict:
    if not any(k.startswith("encoder.layers.") for k in sd):
        return sd
    import re
    out = {}
    for k, v in sd.items():
        for pat, rep in _VIT_V5_TO_V4:
            k2 = re.sub(pat, rep, k)
            if k2 != k:
                k = k2
                break
        out[k] = v
    log.info("converted transformers-5.x ViT key names to 4.x layout")
    return out


def _qwen_vl_keys_v5_to_v4(sd: Dict) -> Dict:
    """transformers 5.x nests Qwen2.5-VL as model.language_model / model.visual;
    4.49 uses model.layers / visual. The golden LoRA adapters were saved under 5.17."""
    out = {}
    for k, v in sd.items():
        k = k.replace("base_model.model.model.language_model.", "base_model.model.model.", 1)
        k = k.replace("base_model.model.model.visual.", "base_model.model.visual.", 1)
        out[k] = v
    return out


def cleanup_hf_cache(keep_substrings, cache_dir: Optional[str] = None, dry_run: bool = True):
    """Prune HuggingFace hub snapshots for models we are no longer using.

    Called at the end of each SLURM script. `keep_substrings` is a list of
    model-id fragments to preserve, e.g. ["trocr-large-handwritten"].
    """
    cache_dir = cache_dir or os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface")
    )
    hub = os.path.join(cache_dir, "hub")
    if not os.path.isdir(hub):
        return []
    removed = []
    for entry in os.listdir(hub):
        if not entry.startswith("models--"):
            continue
        if any(s.lower().replace("/", "--") in entry.lower() for s in keep_substrings):
            continue
        target = os.path.join(hub, entry)
        removed.append(target)
        if not dry_run:
            shutil.rmtree(target, ignore_errors=True)
    log.info("HF cache prune (dry_run=%s): %d candidate dirs", dry_run, len(removed))
    return removed
