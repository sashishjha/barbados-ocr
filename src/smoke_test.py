"""
ROCm / MI300X environment smoke test. RUN THIS BEFORE COMMITTING TO ANY
ARCHITECTURE - it is cheap and it is the difference between discovering a
broken stack in 5 minutes and discovering it 20 hours into a training job.

Checks, in order:
  1. torch sees a GPU through the HIP runtime; report torch.version.hip
  2. bf16 is supported and a bf16 matmul runs
  3. SDPA attention works (we do NOT depend on flash-attention)
  4. transformers / peft / trl import and report versions
  5. bitsandbytes is REPORTED BUT NOT REQUIRED - we deliberately avoid it on
     ROCm; the check exists only to document its state
  6. per-candidate-model forward AND backward pass with a LoRA adapter attached
  7. a tiny end-to-end pass of the eval harness on synthetic strings
  8. peak memory report, so batch sizes can be set from evidence

Usage:
    python -m src.smoke_test --models trocr vlm ctc
    python -m src.smoke_test --models trocr --images data/images --n 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np


def ok(msg):
    print(f"  [OK]   {msg}")


def bad(msg):
    print(f"  [FAIL] {msg}")


def info(msg):
    print(f"  [info] {msg}")


def check_torch():
    print("\n=== 1-2. torch / ROCm / bf16 ===")
    import torch

    info(f"torch {torch.__version__}")
    info(f"torch.version.hip  = {getattr(torch.version, 'hip', None)}")
    info(f"torch.version.cuda = {getattr(torch.version, 'cuda', None)}")
    if getattr(torch.version, "hip", None) is None:
        bad("This torch build has no HIP. Install a ROCm wheel:\n"
            "       pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch")
    if not torch.cuda.is_available():
        bad("No GPU visible. Are you on a compute node? Did you module load rocm/6.2.4?")
        return False
    n = torch.cuda.device_count()
    ok(f"{n} GPU(s) visible; HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        info(f"  gpu{i}: {p.name}  {p.total_memory/1e9:.0f} GB")
    if not torch.cuda.is_bf16_supported():
        bad("bf16 unsupported - the whole plan assumes bf16. Stop and investigate.")
        return False
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    t0 = time.time()
    (a @ a).float().sum().item()
    ok(f"bf16 matmul fine ({time.time()-t0:.2f}s)")
    return True


def check_sdpa():
    print("\n=== 3. SDPA attention (flash-attention intentionally optional) ===")
    import torch
    import torch.nn.functional as F

    try:
        q = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
        out = F.scaled_dot_product_attention(q, q, q)
        ok(f"SDPA works, output {tuple(out.shape)}")
    except Exception as e:
        bad(f"SDPA failed: {e}")
    try:
        import flash_attn  # noqa: F401
        info("flash-attn IS importable - optional, only use if a real forward pass works")
    except Exception:
        info("flash-attn not available - fine, we use SDPA by design")


def check_libs():
    print("\n=== 4-5. library versions ===")
    for mod in ("transformers", "peft", "trl", "datasets", "accelerate",
                "scipy", "pandas", "PIL"):
        try:
            m = __import__(mod)
            info(f"{mod:<14} {getattr(m, '__version__', 'n/a')}")
        except Exception as e:
            bad(f"{mod:<14} MISSING ({e})")
    try:
        import bitsandbytes  # noqa: F401
        info("bitsandbytes importable - we still DO NOT use it (ROCm-fragile, "
             "and unnecessary at 192 GB HBM)")
    except Exception:
        ok("bitsandbytes absent - as intended for this ROCm plan")


def check_eval_harness():
    print("\n=== 7. eval harness self-test ===")
    import pandas as pd

    from src.eval.eval_metrics import evaluate_set

    tmp = "runs_checkpoint/_smoke"
    os.makedirs(tmp, exist_ok=True)
    ref = pd.DataFrame({"ID": ["a", "b"],
                        "Target": ["To all Christian people to whome these presents shall come",
                                   "Sealed and delivered in the presence of"]})
    hyp = pd.DataFrame({"ID": ["a", "b"],
                        "Target": ["To all Christian people to whome these presents shall come",
                                   "Sealed and deliuered in the prsence of"]})
    rp, hp = os.path.join(tmp, "ref.csv"), os.path.join(tmp, "hyp.csv")
    ref.to_csv(rp, index=False); hyp.to_csv(hp, index=False)
    r = evaluate_set(rp, hp, "SMOKE", verbose=False)
    assert 0.0 <= r["final_07_03"] <= 1.0
    ok(f"harness runs; a perfect+one-noisy pair gives final_07_03={r['final_07_03']:.4f}")
    info("(sanity only - NOT a model result)")


def check_trocr(train_step: bool = True):
    print("\n=== 6a. TrOCR forward + backward ===")
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    mid = "microsoft/trocr-large-handwritten"
    try:
        proc = TrOCRProcessor.from_pretrained(mid)
        model = VisionEncoderDecoderModel.from_pretrained(mid, attn_implementation="eager").cuda()
        model.config.decoder_start_token_id = proc.tokenizer.cls_token_id
        model.config.pad_token_id = proc.tokenizer.pad_token_id
        from PIL import Image
        img = Image.fromarray((np.random.rand(96, 512, 3) * 255).astype(np.uint8))
        pv = proc(images=[img, img], return_tensors="pt").pixel_values.cuda()
        lab = proc.tokenizer(["hello world", "test line"], padding=True,
                             return_tensors="pt").input_ids.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(pixel_values=pv, labels=lab)
        ok(f"forward loss={float(out.loss):.4f}")
        if train_step:
            out.loss.backward()
            ok("backward pass fine")
        torch.cuda.empty_cache()
    except Exception:
        bad("TrOCR check failed:")
        traceback.print_exc()


def check_vlm():
    print("\n=== 6b. VLM bf16 LoRA forward + backward (NO bitsandbytes) ===")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    mid = os.environ.get("SMOKE_VLM_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
    try:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as C
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration as C
            info("using Qwen2VL class (transformers too old for Qwen2.5-VL)")

        model = C.from_pretrained(mid, torch_dtype=torch.bfloat16,
                                  attn_implementation="sdpa",
                                  trust_remote_code=True).cuda()
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        model.print_trainable_parameters()

        proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        from PIL import Image
        img = Image.fromarray((np.random.rand(96, 512, 3) * 255).astype(np.uint8))
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "Transcribe EXACTLY as seen."},
            {"type": "image", "image": img}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc = proc(text=[text], images=[img], return_tensors="pt", padding=True)
        enc = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in enc.items()}
        enc["labels"] = enc["input_ids"].clone()
        out = model(**enc)
        ok(f"forward loss={float(out.loss):.4f}")
        out.loss.backward()
        ok("backward pass fine (bf16 LoRA, no quantization config)")
        info(f"peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
        torch.cuda.empty_cache()
    except Exception:
        bad("VLM check failed:")
        traceback.print_exc()
        info("If this is an OOM, lower model.max_pixels in the config. If it is a "
             "class-not-found error, upgrade transformers.")


def check_ctc():
    print("\n=== 6c. CRNN/CTC forward + backward ===")
    import torch

    from src.models.crnn_ctc import CRNN, CharTokenizer

    try:
        tok = CharTokenizer.from_texts(["To all Christian people", "Sealed and delivered"])
        model = CRNN(n_classes=len(tok)).cuda()
        x = torch.randn(4, 1, 96, 1024, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp = model(x)
        lp = lp.float()
        tgt = torch.cat([torch.tensor(tok.encode("hello")) for _ in range(4)]).cuda()
        tl = torch.full((4,), 5, dtype=torch.long).cuda()
        il = torch.full((4,), lp.size(0), dtype=torch.long).cuda()
        loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)(lp, tgt, il, tl)
        ok(f"forward loss={float(loss):.4f}, timesteps={lp.size(0)} (need > max label len 120)")
        loss.backward()
        ok("backward pass fine")
        torch.cuda.empty_cache()
    except Exception:
        bad("CTC check failed:")
        traceback.print_exc()


def check_pipeline(image_dir: str, n: int):
    print(f"\n=== 8. preprocessing + augmentation on {n} real images ===")
    from src.data.augment import AugmentConfig, Augmenter
    from src.data.preprocess import PreprocessConfig, preprocess_image

    if not os.path.isdir(image_dir):
        info(f"{image_dir} not present - skipping (run on the cluster with data staged)")
        return
    files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")][:n]
    if not files:
        info("no jpgs found - skipping")
        return
    pre = PreprocessConfig()
    aug = Augmenter(AugmentConfig(seed=0))
    t0 = time.time()
    for f in files:
        img = preprocess_image(os.path.join(image_dir, f), pre)
        aug(img)
    dt = time.time() - t0
    ok(f"{len(files)} images in {dt:.1f}s ({dt/len(files)*1000:.0f} ms/img)")
    info(f"at 8 dataloader workers that is ~{8*len(files)/max(dt,1e-6):.0f} img/s; "
         "raise num_workers if the GPU is starving")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["trocr", "ctc"],
                    choices=["trocr", "vlm", "ctc"])
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()

    print("=" * 70)
    print("  ROCm / MI300X SMOKE TEST")
    print("=" * 70)

    if not check_torch():
        print("\nAborting: no usable GPU.")
        sys.exit(1)
    check_sdpa()
    check_libs()
    check_eval_harness()
    if "trocr" in a.models:
        check_trocr()
    if "vlm" in a.models:
        check_vlm()
    if "ctc" in a.models:
        check_ctc()
    check_pipeline(a.images, a.n)

    print("\n" + "=" * 70)
    print("Smoke test finished. Read every [FAIL] above before submitting a")
    print("real training job - you only get 2 concurrent jobs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
