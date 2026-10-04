# Handwriting OCR for historic Barbados records (Zindi R.O.A.D. challenge)

Line-level transcription of 17th–18th-century Barbados deeds, wills and inventories with fine-tuned vision-language models
(Qwen3-VL, Qwen2.5-VL), TrOCR, and a character-level ROVER ensemble. Final public leaderboard score **0.92417**
(`1 − (0.5·CER + 0.5·WER)`; the top score was 0.9485). Results, what worked and what did not: [`docs/RESULTS.md`](docs/RESULTS.md).
Engineering lessons: [`docs/LESSONS.md`](docs/LESSONS.md). Exact recipe behind the two selected submissions: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

**Not included:** the competition data (see [`data/README.md`](data/README.md)). The final members' test predictions are in `artifacts/` (the two selected submissions can be rebuilt exactly with `python -m scripts.build_final`), and their weights are on Hugging Face (see `docs/REPRODUCIBILITY.md`).

## Pipeline
1. **Data and folds** — `src/data/` (`folds.py`, `exclusions.py`, `datasets.py`); `slurm/01_prepare_data.sbatch` writes `data/derived_checkpoint/`.
2. **Train a reader** — `src/train_vlm.py` (Qwen-VL LoRA; optional vision-tower LoRA via `model.vision_lora_r`) or `src/train_trocr.py`.
   Configs are in `configs/` and inherit from `base.yaml`; `--set a.b=c` overrides one value, all overrides go after a single `--set`.
3. **N-best inference** — `src/infer.py` (beam search, `--nbest 3`) writes `ID,Target,...` n-best CSVs.
4. **Ensemble** — character-level ROVER in `scripts/tune_ensemble_r8.py` (`prefilter_candidates`, `fuse`, `guard_runaway`);
   `scripts/build_final.py` builds the final submission files; `src/make_submission.py` checks and logs them.
5. **Analysis** — `scripts/eval_member.py` (does a new model help the ensemble on validation), `scripts/error_anatomy2.py`,
   `scripts/rescore_candidates.py` (teacher-forced candidate scoring), `scripts/doc_context_probe.py`.

## Reproducing the final submission
```bash
# 1. environment (AMD ROCm shown; any CUDA/ROCm PyTorch works): see requirements.txt
#    Qwen3-VL needs transformers >= 4.57 (installed into .pylib_qwen3_checkpoint and put first on PYTHONPATH); Qwen2.5-VL uses 4.49
# 2. data: see data/README.md, then
sbatch slurm/01_prepare_data.sbatch

# 3. train the members (each writes <output.checkpoint_dir>/best_checkpoint; tune the config for your GPU)
python -m src.train_vlm --config configs/qwen3vl_8b_r17_vis128_all.yaml --resume          # Qwen3-VL-8B, vision LoRA r128, all data
python -m src.train_vlm --config configs/qwen3vl_8b_r18_vis128_all_s7.yaml --resume       # same, seed 7
python -m src.train_vlm --config configs/qwen_recipe_7b_r18_vis128_all.yaml --resume      # Qwen2.5-VL-7B, vision LoRA r128, all data
python -m src.train_vlm --config configs/qwen3vl_32b_r10.yaml --resume                    # Qwen3-VL-32B
python -m src.train_vlm --config configs/qwen_recipe_7b_b.yaml --resume                   # Qwen2.5-VL-7B (r32)
python -m src.train_trocr --config configs/trocr_large.yaml --resume                      # TrOCR-large

# 4. n-best test predictions per member, e.g.
python -m src.infer --config configs/qwen3vl_8b_r17_vis128_all.yaml --checkpoint <ckpt_dir>/best_checkpoint \
    --arm vlm --manifest data/Test.csv --nbest 3 --set generate.num_beams=3 --out preds_checkpoint/<name>_test_nbest.csv

# 5. fuse (edit the file map in scripts/build_final.py to your prediction files), guard, write submission
python -m scripts.build_final
```
`slurm/` holds the sbatch/launcher scripts used on a shared 3-GPU AMD MI300X node (retry-from-checkpoint loops,
free-memory preflight, 24 h job limit handling). Seeds are fixed in the configs.

## Layout
```
src/        training, inference, data, metric (src/eval), ensembling and submission code
configs/    one YAML per experiment (r8 … r20 = experiment rounds; *_all = trained on all labelled data)
scripts/    ensembling, evaluation and analysis tools
slurm/      cluster job scripts
tests/      pipeline tests
docs/       results and lessons
```

## Notes
- Metric: the official leaderboard metric does not cap per-line error; `src/eval/` follows the competition's definition,
  and every submission builder guards against degenerate (runaway) lines.
- No license has been chosen yet; add one before making the repository public.
