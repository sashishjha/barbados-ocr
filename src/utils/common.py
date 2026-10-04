"""Config loading, seeding, device pinning, logging. Shared by every entrypoint."""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
from typing import Any, Dict, Optional

import numpy as np
import yaml


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def _deep_merge(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, overrides: Optional[Dict] = None) -> Dict:
    """Load a YAML config. Supports `inherit: other.yaml` (relative to this file)."""
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    parent = cfg.pop("inherit", None)
    if parent:
        parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(path), parent)
        cfg = _deep_merge(load_config(parent_path), cfg)

    if overrides:
        cfg = _deep_merge(cfg, overrides)
    cfg["_config_path"] = path
    return cfg


def _coerce(val: str):
    """YAML 1.1 does not parse '1e-5' as a float (it needs '1.0e-5'), which is a
    classic silent-string-LR bug. Try float first, then YAML."""
    try:
        f = float(val)
        return int(f) if (f.is_integer() and "." not in val
                          and "e" not in val.lower()) else f
    except (TypeError, ValueError):
        pass
    try:
        return yaml.safe_load(val)
    except Exception:
        return val


def parse_cli_overrides(items) -> Dict:
    """Turn ['train.lr=1e-4', 'data.fold=2'] into a nested dict."""
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be key.path=value")
        key, val = item.split("=", 1)
        val = _coerce(val)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out


def config_hash(cfg: Dict) -> str:
    """Stable short hash of a config, used to tag submissions in the log."""
    import hashlib

    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # ROCm honours these; full determinism is not guaranteed for all
            # conv/attention kernels, which we accept and document.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def rng_state() -> Dict:
    """Capture RNG state for checkpointing."""
    state = {"python": random.getstate(), "numpy": np.random.get_state()}
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def restore_rng_state(state: Dict) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch

        if "torch" in state:
            torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Device (ROCm)
# --------------------------------------------------------------------------
def get_device(prefer: str = "auto"):
    """ROCm exposes AMD GPUs through the torch.cuda API. HIP_VISIBLE_DEVICES
    (set by the SLURM launcher) already restricts which physical GPU we see, so
    we always address it as cuda:0 inside the process."""
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_runtime() -> Dict:
    info = {"python": sys.version.split()[0]}
    try:
        import torch

        info["torch"] = torch.__version__
        info["hip"] = getattr(torch.version, "hip", None)
        info["cuda"] = getattr(torch.version, "cuda", None)
        info["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        info["gpus"] = [
            torch.cuda.get_device_name(i) for i in range(info["gpu_count"])
        ]
        info["bf16_supported"] = (
            torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        )
    except Exception as e:  # pragma: no cover
        info["torch_error"] = str(e)
    info["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES")
    info["git_commit"] = _git_commit()
    return info


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str = "barbados", logfile: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
