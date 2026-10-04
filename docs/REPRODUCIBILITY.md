# Reproducing the two selected submissions

Both files were uploaded to Zindi and are the two best-scoring submissions (the intended final selection). Everything below is taken from the run logs. Two levels of reproduction:

1. **Exact (fusion only):** the six members' test n-best files are in `artifacts/members/`. Put the competition `Test.csv` at `data/Test.csv` and run `python -m scripts.build_final`; the two output CSVs match the SHA-256 values below (verified on a clean copy). The finished files are also in `artifacts/submissions/`.
2. **Models:** the adapters / weights are on Hugging Face (see *Weights*). To regenerate the n-best files, download them, then run `src/infer.py` as in the README. Retraining from the configs gives scores near, not exactly on, these (see notes).

| Zindi submission | File | Public score | Weights | SHA-256 |
|---|---|---|---|---|
| 6XGecxkf | `sub_20261004_final_vis3_trocr2.csv` | 0.92417 | TrOCR ×2, others ×1 | `71104873e829d5ebcf028e355456b5fe24163cf28e3ac3d62c522b816bc6b1b6` |
| PY73CCA7 | `sub_20261004_final_vis3.csv` | 0.92300 | all ×1 | `e0ae53ff36b10803dde5816c30a2441daf79d955a665e44f6aba3a3785990514` |

## How a file is produced
1. Each member's test-set n-best CSV (3 beams, `--nbest 3`, `generate.num_beams=3`) is read with **top-1 only** (`load(path, name, 1)`).
2. `prefilter_candidates` drops degenerate candidates; `fuse(..., 'rover')` runs the character-level vote with the weights above; `guard_runaway` replaces runaway lines.
3. `src/make_submission.py` writes and checks the CSV (fallback `test_top1_qwen3vl8b.csv`, train lexicon `train_and_pseudo_r5.csv`).

Command: `python -m scripts.build_final` (variants `vis3` and `vis3_trocr2`).

## Members (line height 256 px for the Qwen models; effective batch 16 = 2 × 8 accumulation; lr 1e-4 unless noted)

| Member | Model / recipe | Config | Launcher | Checkpoint dir | Seed | Test n-best file (SHA-256) |
|---|---|---|---|---|---|---|
| `q32` | Qwen3-VL-32B-Instruct, LoRA r32, 2 epochs, train folds (fold 0 held out), no pseudo-labels | `configs/qwen3vl_32b_r10.yaml` | `slurm/29_train_r10.sbatch` | `runs_checkpoint/qwen3vl32b_r10_checkpoints (banked: BEST_qwen3vl32b_r10_f55_09002)` | 42 | `qwen3vl32b_r10_test_nbest_1478.csv` `d6df324556a467c05b34c87541170d2b1bc449d60b4b540a83d9fc8319af2ca0` |
| `q3_vis` | Qwen3-VL-8B-Instruct, LoRA r32 + vision-tower LoRA r128, 3 epochs, ALL labelled data | `configs/qwen3vl_8b_r17_vis128_all.yaml` | `slurm/48_r17_trio.sbatch` | `runs_checkpoint/qwen3vl8b_r17_vis128_all_checkpoints` | 42 | `r17_vis128_all_test_nbest_1719.csv` `dc4833b40c651d1f5da1674295c425928efc13de5fc8ce69fac728d6f57f3b0a` |
| `q3_vis_s7` | same as q3_vis, seed 7 | `configs/qwen3vl_8b_r18_vis128_all_s7.yaml` | `slurm/49_r18_trio.sbatch` | `runs_checkpoint/qwen3vl8b_r18_vis128_all_s7_checkpoints` | 7 | `r18_q3_vis128_all_s7_test_nbest_1721.csv` `0ecfdc0ba0266e1e8ee61be9242d851cff7919c75efbb7d520069b55068423b9` |
| `q7_vis` | Qwen2.5-VL-7B-Instruct, LoRA r32 + vision-tower LoRA r128, 3 epochs, ALL labelled data | `configs/qwen_recipe_7b_r18_vis128_all.yaml` | `slurm/49_r18_trio.sbatch` | `runs_checkpoint/qwen7b_r18_vis128_all_checkpoints` | 42 | `r18_q7_vis128_all_test_nbest_1721.csv` `6659cfca7d4d0b59f834bfb2262f622b0a11997e460245bbb794bd5166ca7b31` |
| `q7_r8b` | Qwen2.5-VL-7B-Instruct, LoRA r32, lr 5e-5, StackMix synthetic lines, 3 epochs, fold 0 held out, no pseudo-labels | `configs/qwen_recipe_7b_b.yaml` | `slurm/26_train_vlm_trio.sbatch (arm A3)` | `runs_checkpoint/qwen7b_r8b_checkpoints (banked: BEST_qwen7b_r8b_f55_08829)` | 7 | `qwen7b_b_test_nbest_1387.csv` `d20be892b65b7f38896e3d6ce3b7884c5cb7bfe1bb8ecc1607405f9381b20a9e` |
| `trocr_r5` | TrOCR-large-handwritten fine-tuned on train + round-5 pseudo-labelled test lines, 12 epochs | `configs/trocr_large.yaml + train.pseudo_csv=data/derived_checkpoint/train_and_pseudo_r5.csv` | `slurm/23_pseudo_r5.sbatch (needs pseudo rounds 1-5: slurm/20..23)` | `runs_checkpoint/trocr_large_checkpoints (banked: BEST_trocr_hw_r5_val0878)` | 42 | `trocr_test_nbest_r5_ensemble.csv` `8726a29a583888300f125716f6d013f1e9edc2bdf225e548ac95031647290971` |

## Notes for a re-run
- Approximate training time on one MI300X (shared node): 8B / 7B vision-LoRA all-data models 4.5–6 h each; 32B about 10 h; TrOCR under 1 h.
- Qwen3-VL needs `transformers` 4.57.1 (separate `PYTHONPATH` directory); Qwen2.5-VL and TrOCR ran on 4.49.0.
- The TrOCR member depends on self-training rounds 1–5 (pseudo-labels from earlier ensembles on the test lines); it is the hardest member to rebuild exactly. Replacing it changes the vote (weight 2 in the 0.92417 file).
- Level 1 above is exact; for level 2 GPU training is not bit-exact across hardware or software versions even with fixed seeds, so a rebuilt ensemble will land near, not exactly on, these scores. Compare against the hashes above only if the n-best files themselves are available.
- Earlier-round scores for context: `docs/RESULTS.md`.

## Weights (Hugging Face)
Repo: [`Sashish823/barbados_ocr_adapters`](https://huggingface.co/Sashish823/barbados_ocr_adapters) (private; make it public when you publish this repository). Uploaded with `scripts/upload_members_to_hf.sh`.
Base models come from their original Hugging Face pages (`Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-VL-32B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct`, `microsoft/trocr-large-handwritten`).

| Folder in the repo | Member | File | Size | SHA-256 |
|---|---|---|---|---|
| `q32/` | Qwen3-VL-32B LoRA | `adapter_model.safetensors` | 1.1 GB | `3548d4dfbc78da3717501381a5fca2c46c58076949f180ff99ada1576c92289e` |
| `q3_vis/` | Qwen3-VL-8B vision LoRA, seed 42 | `adapter_model.safetensors` | 577 MB | `bb858f2be967d00ae18519fe7b523669058e67d00f89fbbb454f35d83245d838` |
| `q3_vis_s7/` | Qwen3-VL-8B vision LoRA, seed 7 | `adapter_model.safetensors` | 577 MB | `e489539c914f9837117364eef314e16fbc58755420865ee85613d4292740f459` |
| `q7_vis/` | Qwen2.5-VL-7B vision LoRA | `adapter_model.safetensors` | 651 MB | `6d9cc1c7df55a9ad217cfc2217cd51304202f23f8f5d67841e707587e87a8d37` |
| `q7_r8b/` | Qwen2.5-VL-7B LoRA (fold 0 held out) | `adapter_model.safetensors` | 396 MB | `08848c51297e9217d49bfa981f3ca84f818dc3264f6fe92d5ea240d561cca2cb` |
| `trocr_r5/` | TrOCR-large (full fine-tuned weights) | `model.pt` | 2.1 GB | `f56e078ecdcc7da8d236a00538a7a8edffc422d2437c1b658a98ea8f501e7a1b` |
