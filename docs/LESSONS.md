# Lessons learned — Barbados OCR

### [Qwen2.5-VL Tokenizer instantiation crash]
- **Context**: Loading the `Qwen2.5-VL` model for training or inference in the VLM arm.
- **Problem/Mistake**: Triggers `ValueError: Couldn't instantiate the backend tokenizer... You need to have sentencepiece or tiktoken installed`.
- **Verified Solution/Rule**: The Qwen2.5-VL tokenizer relies on `tiktoken`, which must be explicitly added to `requirements.txt`.
- **Date**: 2026-09-18

### [trocr-large-stage1: meta tensor crash on first forward pass]
- **Context**: Loading `microsoft/trocr-large-stage1` and running training on the cluster.
- **Problem/Mistake**: In recent transformers versions, `TrOCRSinusoidalPositionalEmbedding.__init__` sets `self.weights = self.get_embedding(...)` as a **plain Python attribute, NOT a buffer or parameter**. When HuggingFace loads large models via the meta-device path, `__init__` is called on the meta device, making `self.weights` a meta tensor. `model.to(device)` ONLY moves registered parameters and buffers — it silently skips plain Python attributes. So `self.weights` stays as a meta tensor. In `forward()`, `x = self.weights.index_select(...)` returns a meta tensor, and the decoder's subsequent `embed_pos.to(inputs_embeds.device)` crashes with `NotImplementedError: Cannot copy out of meta tensor; no data!`.
- **Verified Solution/Rule**: After `model.to(device)`, call `get_embedding(...).to(device)` to recompute the sinusoidal table directly on GPU, replacing the meta tensor:
  ```python
  from transformers.models.trocr.modeling_trocr import TrOCRSinusoidalPositionalEmbedding
  for module in model.modules():
      if isinstance(module, TrOCRSinusoidalPositionalEmbedding):
          n_pos = module.weights.shape[0] if module.weights is not None else 1026
          module.weights = TrOCRSinusoidalPositionalEmbedding.get_embedding(
              n_pos, module.embedding_dim, module.padding_idx
          ).to(device)
  ```
- **Dead ends** (tried and failed): `low_cpu_mem_usage=False`, `gradient_checkpointing_enable()` ordering, `register_buffer("_float_tensor", ...)`, `setattr(module, buf_name, buf.to(device))`, `module.weights = None` (triggers recompute on CPU → device mismatch crash).
- **Date**: 2026-09-21

### Missing Dependencies when Venv Exists
- **Context**: Cluster setup script (`slurm/00_setup_smoke_test.sbatch`).
- **Problem/Mistake**: The script skipped `pip install -r requirements.txt` if the `.venv` directory already existed, causing `TrOCRProcessor` to crash due to missing `sentencepiece` and `tiktoken`. Also, `protobuf` was completely missing from `requirements.txt`, which `sentencepiece` requires silently to load models.
- **Verified Solution/Rule**: [PENDING VERIFICATION] Extracted the `uv pip install` step outside the `if [ ! -d "$VENV" ]` block so dependencies are always updated. Added `protobuf` to `requirements.txt`.

### Transformers 5.x Tokenizer Bug with TrOCR
- **Context**: `TrOCRProcessor.from_pretrained` during smoke test.
- **Problem/Mistake**: `transformers>=4.49.0` resolved to the newer `5.17.0`, which has a bug attempting to instantiate slow tokenizers, crashing with `Couldn't instantiate the backend tokenizer`.
- **Verified Solution/Rule**: [PENDING VERIFICATION] Pinned `transformers==4.49.0` in `requirements.txt` to avoid the tokenizer conversion bug.

### Missing pseudo_label assertion
- **Context**: Unit tests (`test_pseudo_label_tripwire`).
- **Problem/Mistake**: `ImportError: cannot import name 'assert_no_manual_overrides' from 'src.pseudo_label'` caused the pytest step in the smoke test to fail.
- **Verified Solution/Rule**: [PENDING VERIFICATION] Added `assert_no_manual_overrides` to `src/pseudo_label.py` to check for `manual_labels.csv`.

### HF Transfer Package Missing
- **Context**: Cluster setup script downloading models via Hugging Face.
- **Problem/Mistake**: `slurm/common.sh` sets `export HF_HUB_ENABLE_HF_TRANSFER=1` to speed up large downloads, but `hf_transfer` was missing from `requirements.txt`, causing an immediate `ValueError` during download.
- **Verified Solution/Rule**: [PENDING VERIFICATION] Added `hf_transfer` to `requirements.txt`.

### TrOCR Attention Implementation Crash
- **Context**: `VisionEncoderDecoderModel.from_pretrained` for TrOCR.
- **Problem/Mistake**: Hardcoding `attn_implementation="sdpa"` caused a `ValueError` because TrOCRForCausalLM does not yet support `sdpa` in transformers.
- **Verified Solution/Rule**: [PENDING VERIFICATION] Set `attn_implementation="eager"` for TrOCR. This is already handled correctly in `configs/trocr_large.yaml`, but the smoke test script needed the same fix.

### Unbound Variable Crash on Cleanup
- **Context**: `slurm/00_setup_smoke_test.sbatch` cleanup.
- **Problem/Mistake**: The script executes `rm -rf "$TMPDIR"` at the very end. But `TMPDIR` is only set if the script creates the venv. If the venv already exists, `TMPDIR` is unbound. Because `slurm/common.sh` turns on `set -euo pipefail`, the unbound variable causes the entire job to exit with code 1, silently failing the dependency chain despite the smoke test succeeding!
- **Verified Solution/Rule**: [PENDING VERIFICATION] Use `${TMPDIR:-}` to safely reference potentially unbound variables under `set -u`.

### Trap Abort via Pipeline Failure under set -e
- **Context**: `cleanup_storage` in `slurm/common.sh` executed via `trap EXIT`.
- **Problem/Mistake**: The pipeline `find ... | while read ... done` failed silently (either because `find` encountered unreadable subdirectories and returned 1, or because the `while` loop evaluated to 1 on the last iteration). Because the function was running under `set -e` in an exit trap, it aborted the trap and changed the script's exit code to 1, causing the SLURM job to fail despite a successful run.
- **Verified Solution/Rule**: [PENDING VERIFICATION] Added `|| true` to the end of the `done` statement in the pipeline to suppress the error and allow the trap to finish successfully.

### [R6] TrOCR Crashed Silently in Parallel Job — Left Stale Checkpoint
- **Context**: `12_parallel_arms.sbatch` running TrOCR + VLM + CTC in parallel (Jobs 1234-1236).
- **Problem/Mistake**: TrOCR arm (arm 0) crashed immediately with `ValueError: TrOCRForCausalLM does not support SDPA` even though `attn_implementation: eager` was set in the YAML. The crash happened because the resume path loaded from an older stale checkpoint that didn't carry the eager config forward. The log was only 50 lines. The `best_checkpoint` left behind was from a much earlier/weaker training run. We then used this stale checkpoint for test inference and scored 0.862 instead of 0.898.
- **Verified Solution/Rule**: ALWAYS verify training logs show actual epoch completion before submitting inference. Check log line count and look for `epoch N VAL` lines. A log with <100 lines after a training job almost certainly means a crash.
- **Date**: 2026-09-26

### [R6] Training from Scratch Instead of from R5 Checkpoint — Discarded 5 Rounds of Pseudo-Labels
- **Context**: R6 parallel training (12_parallel_arms.sbatch) with `--resume` flag.
- **Problem/Mistake**: `--resume` looks for a checkpoint in `output.checkpoint_dir`. When that dir is empty (cleared for R6), it starts fresh from HuggingFace pretrained weights. This means R6 discarded 5 rounds of pseudo-label fine-tuning and started from scratch with only 3,243 base training rows. The R5 best weights were safely in `BEST_qwen7b_r5_val0910/` and `BEST_trocr_hw_r5_val0878/` but we never copied them into the checkpoint_dir before resuming.
- **Verified Solution/Rule**: When starting a new round, ALWAYS copy the best checkpoint from `BEST_*_checkpoint/` into the working `*_checkpoints/` dir before submitting training with `--resume`. The golden rule is: **new rounds build on the previous best, never from scratch.**
- **Date**: 2026-09-26

### [R6] Using nbest>1 in ROVER Inference Dramatically Hurts Score
- **Context**: Ensemble inference in `30_inference_ensemble.sbatch`.
- **Problem/Mistake**: Setting `--nbest 5` makes ROVER vote across 10 strings (5 from TrOCR + 5 from VLM) per image. The lower-ranked beam candidates are noisy garbage strings that corrupt the character vote and pivot selection. nbest=5 gave LB 0.872 vs nbest=1 giving 0.862, both dramatically worse than R5 nbest=1 (0.898).
- **Verified Solution/Rule**: For ROVER ensemble, ALWAYS use `--nbest 1` (top-1 only from each arm). Using nbest>1 makes the character voting noise-dominated. Only use nbest>1 for MBR decoding (which was also abandoned at 0.887).
- **Date**: 2026-09-26

### [R6] Adding a Weak Arm to ROVER Hurts Score
- **Context**: Adding CTC arm (OOF=0.8615) to ROVER alongside TrOCR (0.8838) and VLM (0.8797).
- **Problem/Mistake**: Even with low arm_weight=0.7, CTC's garbled predictions corrupted ROVER character votes for 843/1374 rows. Score dropped from 0.898 (2-arm) to 0.890 (3-arm).
- **Verified Solution/Rule**: Never add an arm to ROVER unless its OOF score is within ~0.01 of the other arms. A weak arm always hurts ROVER regardless of arm_weight because ROVER is character-level voting, not score-level selection.
- **Date**: 2026-09-26


### [R7] Seeding golden weights as `best_checkpoint/` does NOT warm-start — resume only globs `checkpoint_step*`
- **Context**: R7 launch (`slurm/25_train_r7.sbatch`) copying `BEST_*_r5_*` into `<ckpt_dir>/best_checkpoint`, then `--resume`. Refines the R6 "copy best checkpoint before resuming" rule above.
- **Problem/Mistake**: `CheckpointManager.latest()` (src/utils/ckpt_manager.py:130) only globs `checkpoint_step*`, so a `best_checkpoint/` folder is ignored → "no checkpoint found ... starting fresh" (seen in job 1351 for both arms). `WARM_START=1` is not read by any code. Also: the script `rm -rf`s the R7 ckpt dirs at every launch, which wipes progress if the job is requeued after the 24 h limit.
- **Verified Solution/Rule**: [PENDING VERIFICATION]
- **Date**: 2026-09-27

### [R7] `set -e` (from slurm/common.sh) + bare `wait` orphaned an arm on the GPU → node DRAIN
- **Context**: Multi-arm sbatch scripts that background several trainers and `wait` on them.
- **Problem/Mistake**: Job 1351: Qwen OOM'd at 16:58:35; `wait "$QWEN_PID"` returned non-zero and `set -e` (enabled by common.sh) exited the script at 16:58:46 while TrOCR still held its GPU. SLURM couldn't kill the orphan within UnkillableStepTimeout (60 s) → "Kill task failed" → iisc drained. The OOM itself came from other users' non-SLURM processes (~180 GB) on the same GPU. Also: the script hard-coded ROCR_VISIBLE_DEVICES=0/1, overriding SLURM's allocation.
- **Verified Solution/Rule**: [PENDING VERIFICATION] `wait "$PID" || STATUS=$?` for every arm; trap TERM/INT to forward to arms and wait; select GPUs with HIP_VISIBLE_DEVICES (relative to SLURM's allocation); preflight free-VRAM check before launch.
- **Date**: 2026-09-27

### [R7] Golden checkpoints saved under transformers 5.17 don't load under the 4.49 pin — and PEFT fails SILENTLY
- **Context**: Warm-starting R7 from `BEST_*_r5_*` after `transformers` was pinned 5.17 → 4.49 on 25 Sep.
- **Problem/Mistake**: 5.x renamed layers. TrOCR ViT: `encoder.layers.N.attention.q_proj` vs 4.49 `encoder.encoder.layer.N.attention.attention.query` → `load_state_dict` raised (job 1357). Qwen2.5-VL LoRA: `model.language_model.layers` / `model.visual` vs 4.49 `model.layers` / `visual` → `load_adapter` skipped every key WITHOUT error; loss at step 25 was the same as from scratch (3.96).
- **Verified Solution/Rule**: [PENDING VERIFICATION] `ckpt_manager.init_from` renames keys (`_vit_keys_v5_to_v4`, `_qwen_vl_keys_v5_to_v4`) and loads the adapter via `set_peft_model_state_dict`, raising if any `lora_` key is unexpected. Rule: after any warm start, compare the step-25 loss with a from-scratch run.
- **Date**: 2026-09-29

### [R7] `srun --jobid=<id> --overlap` defaults to several tasks → duplicate training runs
- **Context**: Starting an extra process inside an existing allocation.
- **Problem/Mistake**: Without `-n1`, the step ran with 2 tasks (job had 32 CPUs, `--cpus-per-task=16`), so two TrOCR copies trained into the same checkpoint dir.
- **Verified Solution/Rule**: Always pass `-n1` (`--ntasks=1`) to `srun` steps that launch a single program.
- **Date**: 2026-09-29

### Small val gains from ensemble tweaks did not transfer to the LB
- **Context**: Choosing which ensemble variant to submit (weights, adding a similar arm).
- **Problem/Mistake**: 5-arm ROVER scored +0.0022 over the 4-arm on fold-0 val (0.9130 vs 0.9108) but -0.0021 on the public LB (0.9133 vs 0.9154). With 811 val lines and a partial public test set, differences of ~0.002 are noise; spending a submission on it wasted one of 5 daily slots.
- **Verified Solution/Rule**: Only submit a new ensemble variant when its val gain is >= ~0.005 or it adds a genuinely stronger model; treat smaller deltas as ties and prefer the simpler / already-proven submission.
- **Date**: 2026-09-30

### Self-training on your own test predictions gives val gains that vanish on the LB
- **Context**: Pseudo-labelling Test with the current best ensemble, retraining, ensembling again.
- **Problem/Mistake**: R9 models (trained on 972 test lines labelled by sub1's ensemble) scored +0.004 on fold-0 val and the R9 ensemble +0.006 on val (0.9171), but the LB went 0.9154 -> 0.9141. On the pseudo-labelled test lines the R9 models reproduce sub1 (84-86% identical vs 72% for R8), so their test outputs add no independent information; fold-0 val (never pseudo-labelled) can't show that.
- **Verified Solution/Rule**: Treat val gains from test-pseudo-label rounds as non-transferable. Check agreement of the new model with its own pseudo-label source on the pseudo lines before trusting it. Real gains must come from independent information: real labels (train on ALL labelled data incl. fold 0), or genuinely different/stronger models.
- **Date**: 2026-09-30

### One runaway line from one ensemble member can sink the LB score
- **Context**: ROVER ensembles on Test; the official metric may not cap per-line error at 1.
- **Problem/Mistake**: Adding Qwen3-VL-32B (val +0.0044 in ROVER) gave LB 0.9005 vs 0.9154. Its test output had a degenerate repetition ('& R . . . .', ~300 chars for a ~19-char line) and ROVER kept it. Our metric copy caps per-line CER at 1, so validation could never show the damage.
- **Verified Solution/Rule**: VERIFIED 2026-10-01: replacing that ONE line moved LB 0.9005 -> **0.9187** (new best; WER-w 1.453, CER-w 2.286) — the official metric does not cap per-line error. Before submitting any ensemble, scan for repetition/runaway lines (same token 6+ times, or > 1.5x the median candidate length + 10) and replace them with the best-agreeing non-runaway candidate.
- **Date**: 2026-10-01

### load_config overrides must be nested (or built with parse_cli_overrides), dotted keys are ignored
- **Context**: Overriding a config in a script, e.g. a smoke test switching model.id.
- **Problem/Mistake**: load_config(path, {"model.id": "...-1B-HF"}) silently ignored the dotted key, so the "1B" smoke test loaded the 8B model and downloaded 16 GB, which filled the 200 GB quota and broke a running job's checkpoint save.
- **Verified Solution/Rule**: Build overrides with parse_cli_overrides(["model.id=..."]) (as the CLI does) and print the resolved value before loading anything big; check quota headroom before any download.
- **Date**: 2026-10-02

## 4 Oct 2026 — overlap steps die with the batch script (cluster)
- Problem: I launched extra training as `srun --jobid=J --overlap` steps inside a running job and assumed they would live to the job's 24 h limit. The job's sbatch script only waited for its own arms; when they finished the script exited and SLURM tore down the whole allocation, killing all my steps (3 runs interrupted, ~1 h GPU time lost, overnight).
- Rule: overlap/head-start steps are only safe while the batch script is guaranteed to outlive them. For anything long, run it as a CHILD of an sbatch script that `wait`s for all of its arms (or add `sleep infinity`-style guard until a sentinel file), and check `sacct`/`squeue` for the real end time.

## 4 Oct — big-gap competitions: test the radical hypotheses early, not on the last day
When the leader is far ahead (+0.026) and our ensemble/selection work saturates, stop polishing it. Spend the first days on cheap
falsification probes (1-3 GPU-h each) of qualitatively different ideas: other pretrained starting points (dedicated OCR / historical-text
VLMs, check licences), full fine-tuning with an LR sweep, document/writer context (cluster test lines into pages). We spent ~3 days on
ensembling, rescoring and selectors (ceiling val ~.918) and only tried full vision FT (R19/R20) in the final 24 h, with no time to tune it;
document clustering was listed but never probed.
